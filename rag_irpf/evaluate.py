"""Avaliação do sistema e comparação de configurações (chunking / embeddings).

Para cada pergunta do arquivo eval/perguntas_avaliacao.csv:
  * perguntas 'respondivel': mede se a RECUPERAÇÃO trouxe o trecho certo.
        - hit@k : algum dos k chunks recuperados contém TODAS as palavras-chave esperadas
        - MRR   : 1 / posição do primeiro chunk relevante (0 se nenhum)
  * perguntas 'fora_do_escopo': roda o RAG completo e confere se o sistema ADMITE que
        não sabe (a resposta contém a frase padrão de "não encontrei").
  * com --gerar-respostas, salva também a resposta do LLM para você julgar a FIDELIDADE
        manualmente (coluna 'fiel_ao_contexto' para preencher no CSV de saída).

Formato das palavras-chave (coluna 'palavras_chave'):
    termos separados por ';' são TODOS obrigatórios; alternativas dentro de um termo com '|'.
    Ex.: "multa;atraso"  |  "criptoativo|moeda virtual|criptomoeda"
A comparação ignora acentos e maiúsculas/minúsculas.

Uso:
    python -m rag_irpf.evaluate --configs hf:1000:150,hf:500:100 --k 5
    python -m rag_irpf.evaluate --configs hf:1000:150,openai:1000:150 --gerar-respostas
"""
from __future__ import annotations

import argparse
import time
import unicodedata
from pathlib import Path

import pandas as pd

from . import config
from .config import FRASE_SEM_RESPOSTA, IndexConfig
from .ingest import construir_indice
from .rag import RagIRPF

EVAL_PADRAO = config.RAIZ / "eval" / "perguntas_avaliacao.csv"


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sem_acento.lower()


def contem_todas(texto: str, palavras_chave: str) -> bool:
    alvo = normalizar(texto)
    for termo in str(palavras_chave).split(";"):
        alternativas = [normalizar(a).strip() for a in termo.split("|") if a.strip()]
        if alternativas and not any(a in alvo for a in alternativas):
            return False
    return True


def avaliar_configuracao(cfg: IndexConfig, perguntas: pd.DataFrame, csv_base: Path, k: int,
                         gerar_respostas: bool, exercicio_padrao: int) -> pd.DataFrame:
    construir_indice(csv_base, cfg)  # cria se ainda não existir
    rag = RagIRPF(cfg, k=k)
    linhas = []
    for p in perguntas.to_dict("records"):
        exercicio = int(p.get("exercicio") or exercicio_padrao)
        t0 = time.perf_counter()
        achados = rag.recuperar(p["pergunta"], exercicio, k)
        t_busca = time.perf_counter() - t0
        linha = {"config": cfg.nome, "id": p["id"], "tipo": p["tipo"], "pergunta": p["pergunta"],
                 "tempo_busca_s": round(t_busca, 3),
                 "top1": f"{achados[0][0].metadata['exercicio']}-{achados[0][0].metadata['numero']:03d}" if achados else ""}

        if p["tipo"] == "respondivel":
            posicao = next((i + 1 for i, (d, _) in enumerate(achados) if contem_todas(d.page_content, p["palavras_chave"])), 0)
            linha.update({"hit_at_k": int(posicao > 0), "rr": round(1 / posicao, 3) if posicao else 0.0,
                          "posicao_relevante": posicao})
        if gerar_respostas or p["tipo"] == "fora_do_escopo":
            r = rag.responder(p["pergunta"], exercicio)
            linha["resposta"] = r.resposta
            linha["admitiu_nao_saber"] = int(FRASE_SEM_RESPOSTA.lower() in r.resposta.lower())
            linha["fiel_ao_contexto"] = ""  # preencher manualmente (1 = fiel, 0 = inventou)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def resumir(resultados: pd.DataFrame, k: int) -> pd.DataFrame:
    resp = resultados[resultados["tipo"] == "respondivel"].groupby("config").agg(
        **{f"hit@{k}": ("hit_at_k", "mean"), "MRR": ("rr", "mean"), "tempo_busca_s": ("tempo_busca_s", "mean")})
    if "admitiu_nao_saber" in resultados:
        fora = resultados[resultados["tipo"] == "fora_do_escopo"].groupby("config")["admitiu_nao_saber"].mean()
        resp["admite_nao_saber"] = fora
    return resp.round(3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=EVAL_PADRAO, dest="eval_csv")
    ap.add_argument("--csv", type=Path, default=config.CSV_PADRAO, help="CSV da base")
    ap.add_argument("--configs", default="hf:1000:150,hf:500:100",
                    help="lista 'provider:chunk:overlap[:modelo]' separada por vírgula")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--exercicio", type=int, default=2026, help="edição usada quando a linha não define uma")
    ap.add_argument("--gerar-respostas", action="store_true", help="chama o LLM também nas perguntas respondíveis")
    ap.add_argument("--saida", type=Path, default=None)
    args = ap.parse_args(argv)

    perguntas = pd.read_csv(args.eval_csv, encoding="utf-8-sig", dtype=str).fillna("")
    cfgs = [IndexConfig.de_texto(c.strip()) for c in args.configs.split(",") if c.strip()]

    partes = [avaliar_configuracao(c, perguntas, args.csv, args.k, args.gerar_respostas, args.exercicio) for c in cfgs]
    resultados = pd.concat(partes, ignore_index=True)

    saida = args.saida or config.RAIZ / "eval" / f"resultados_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    resultados.to_csv(saida, index=False, encoding="utf-8-sig")
    print("\n=== RESUMO POR CONFIGURAÇÃO ===")
    print(resumir(resultados, args.k).to_string())
    print(f"\nDetalhes por pergunta salvos em: {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
