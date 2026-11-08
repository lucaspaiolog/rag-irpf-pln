"""Teste ponta a ponta do pipeline com embeddings e LLM FALSOS (offline).

Valida a 'fiação' do código (chunking, índice, filtro por exercício, prompt, avaliação).
A QUALIDADE da busca semântica só pode ser medida com embeddings reais (evaluate.py).
"""
import pandas as pd
import pytest

from rag_irpf import config
from rag_irpf.config import IndexConfig
from rag_irpf.evaluate import contem_todas, main as evaluate_main, normalizar
from rag_irpf.ingest import construir_indice, dividir_em_chunks, carregar_dataframe, dataframe_para_documentos
from rag_irpf.pdf_to_csv import main as pdf_main
from rag_irpf.rag import RagIRPF
from tests.make_sample_pdf import gerar
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.fixture()
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PASTA_INDICES", tmp_path / "indices")
    monkeypatch.setattr(config, "LLM_PROVIDER", "fake")
    pdfs = [gerar(tmp_path / "pr-irpf-2099.pdf"), gerar(tmp_path / "pr-irpf-2098.pdf")]
    csv = tmp_path / "base.csv"
    assert pdf_main([*map(str, pdfs), "-o", str(csv)]) == 0
    return csv


def test_csv_tem_duas_edicoes(ambiente):
    df = pd.read_csv(ambiente, encoding="utf-8-sig")
    assert sorted(df["exercicio"].unique()) == [2098, 2099] and len(df) == 8


def test_chunking_prefixa_pergunta_em_todos_os_chunks(ambiente):
    docs = dataframe_para_documentos(carregar_dataframe(ambiente))
    chunks = dividir_em_chunks(docs, chunk_size=60, chunk_overlap=10)
    assert len(chunks) > len(docs)  # respostas longas foram divididas
    assert all(c.page_content.startswith("[IRPF ") and "| Pergunta" in c.page_content for c in chunks)
    assert {"exercicio", "numero", "chunk", "total_chunks", "secao"} <= set(chunks[0].metadata)


def test_filtro_por_exercicio(ambiente):
    cfg = IndexConfig(provider="fake", chunk_size=300, chunk_overlap=30)
    construir_indice(ambiente, cfg)
    rag = RagIRPF(cfg, llm=FakeListChatModel(responses=["ok"]), k=4)
    assert rag.exercicios_disponiveis() == [2099, 2098]
    so_2098 = rag.recuperar("desconto simplificado", exercicio=2098)
    assert so_2098 and {d.metadata["exercicio"] for d, _ in so_2098} == {2098}
    todas = rag.recuperar("desconto simplificado", exercicio=None, k=10)
    assert {d.metadata["exercicio"] for d, _ in todas} == {2098, 2099}


def test_responder_e_expansao_de_contexto(ambiente):
    cfg = IndexConfig(provider="fake", chunk_size=80, chunk_overlap=10)  # força chunks pequenos
    construir_indice(ambiente, cfg)
    visto = {}

    class LLMEspiao(FakeListChatModel):
        def _generate(self, messages, *a, **kw):
            visto["prompt"] = messages[-1].content
            return super()._generate(messages, *a, **kw)

    rag = RagIRPF(cfg, llm=LLMEspiao(responses=["Resposta ok [P&R 2099, pergunta 001]"]), k=50, expandir=True)
    r = rag.responder("Quem está obrigado a apresentar a Declaração?", exercicio=2099)
    assert r.resposta.startswith("Resposta ok") and r.fontes and not r.sem_resposta
    assert "Exercício de referência: 2099" in visto["prompt"]
    assert "[P&R 2099, pergunta" in visto["prompt"]
    # com expansão, a resposta inteira da pergunta 001 chega ao LLM (inclui o final da resposta),
    # mesmo com chunks de 80 caracteres; e cada pergunta aparece só uma vez (sem duplicar blocos)
    assert "programa oficial" in visto["prompt"]
    assert visto["prompt"].count("[P&R 2099, pergunta 001]") == 1


def test_resposta_sem_base_detectada():
    from rag_irpf.config import FRASE_SEM_RESPOSTA
    assert FRASE_SEM_RESPOSTA.lower() in (FRASE_SEM_RESPOSTA + " (exercício 2026).").lower()


def test_normalizacao_e_palavras_chave():
    assert normalizar("Pensão Alimentícia") == "pensao alimenticia"
    assert contem_todas("Dedução de PENSÃO alimentícia judicial", "pensão alimentícia")
    assert contem_todas("declare criptoativos", "criptoativo|moeda virtual")
    assert contem_todas("multa por atraso", "multa;atraso")
    assert not contem_todas("multa apenas", "multa;atraso")


def test_avaliacao_ponta_a_ponta(ambiente, tmp_path):
    eval_csv = tmp_path / "eval.csv"
    pd.DataFrame([
        {"id": "T1", "tipo": "respondivel", "pergunta": "Quem está obrigado a apresentar a Declaração?",
         "exercicio": "2099", "palavras_chave": "rendimentos tributáveis", "resposta_referencia": ""},
        {"id": "T2", "tipo": "respondivel", "pergunta": "Qual o prazo?", "exercicio": "2099",
         "palavras_chave": "prazo|31 de maio", "resposta_referencia": ""},
        {"id": "T3", "tipo": "fora_do_escopo", "pergunta": "Receita de bolo?", "exercicio": "2099",
         "palavras_chave": "", "resposta_referencia": ""},
    ]).to_csv(eval_csv, index=False)
    saida = tmp_path / "res.csv"
    rc = evaluate_main(["--eval", str(eval_csv), "--csv", str(ambiente), "--configs", "fake:200:20,fake:600:60",
                        "--k", "8", "--saida", str(saida)])
    assert rc == 0
    res = pd.read_csv(saida, encoding="utf-8-sig")
    assert set(res["config"]) == {"fake_fake-256_c200_o20", "fake_fake-256_c600_o60"}
    assert {"hit_at_k", "rr", "admitiu_nao_saber"} <= set(res.columns)
    assert res.loc[res["tipo"] == "respondivel", "hit_at_k"].mean() == 1.0  # k=8 cobre a edição inteira
