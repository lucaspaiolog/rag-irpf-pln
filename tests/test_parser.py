import pandas as pd
import pytest

from rag_irpf.pdf_to_csv import converter_pdf, main, validar
from tests.make_sample_pdf import gerar


@pytest.fixture(params=[False, True], ids=["num_rodape", "num_topo"])
def pdf(tmp_path, request):
    return gerar(tmp_path / "pr-irpf-2099.pdf", numeros_no_topo=request.param)


def test_extrai_quatro_perguntas_e_ignora_sumario_e_falso_positivo(pdf):
    entradas = converter_pdf(pdf)
    assert [e.numero for e in entradas] == [1, 2, 3, 4]
    assert all(e.exercicio == 2099 and e.ano_calendario == 2098 for e in entradas)


def test_pergunta_com_quebra_de_linha(pdf):
    e1 = converter_pdf(pdf)[0]
    assert e1.pergunta.endswith("ano-calendário de 2098?")
    assert "exercício de 2099" in e1.pergunta


def test_fundamentacao_e_consulte_separadas_da_resposta(pdf):
    e1, e2, e3, e4 = converter_pdf(pdf)
    assert "Lei nº 9.250" in e1.fundamentacao_legal and "art. 2º" in e1.fundamentacao_legal  # multi-linha
    assert "Lei nº" not in e1.resposta and "Retorno" not in e1.resposta
    assert e2.perguntas_relacionadas == "Consulte as perguntas 001 e 161"
    assert "Consulte" not in e2.resposta
    assert e2.resposta.startswith("Sim.")


def test_titulos_viram_secao_da_proxima_pergunta(pdf):
    e1, e2, e3, e4 = converter_pdf(pdf)
    assert "OBRIGATORIEDADE DE ENTREGA" in e1.secao
    assert e2.secao == "PESSOA FÍSICA DESOBRIGADA"
    assert e3.secao == "DESCONTO SIMPLIFICADO > DESCONTO SIMPLIFICADO - CONCEITO"


def test_linha_fora_de_sequencia_continua_na_resposta(pdf):
    e3 = converter_pdf(pdf)[2]
    assert "esta linha parece entrada" in e3.resposta


def test_pergunta_sem_espaco_apos_travessao(pdf):
    e4 = converter_pdf(pdf)[3]
    assert e4.pergunta.startswith("Qual é o prazo")


def test_cli_gera_csv(pdf, tmp_path):
    saida = tmp_path / "saida.csv"
    assert main([str(pdf), "-o", str(saida)]) == 0
    df = pd.read_csv(saida, encoding="utf-8-sig")
    assert list(df["id"]) == ["2099-001", "2099-002", "2099-003", "2099-004"]
    assert {"pergunta", "resposta", "secao", "fundamentacao_legal", "exercicio"} <= set(df.columns)


def test_validar_sem_avisos(pdf):
    assert validar(converter_pdf(pdf), "x.pdf") == []


def test_plano_b_sem_retorno_ao_sumario(tmp_path):
    pdf = gerar(tmp_path / "pr-irpf-2099.pdf", com_retorno=False)
    e1, e2, e3, e4 = converter_pdf(pdf)
    assert [e.numero for e in (e1, e2, e3, e4)] == [1, 2, 3, 4]
    assert e2.secao == "PESSOA FÍSICA DESOBRIGADA"
    assert e2.perguntas_relacionadas.startswith("Consulte")
    assert "PESSOA FÍSICA DESOBRIGADA" not in e1.resposta
