"""Gera um PDF de teste imitando o layout do 'Perguntas e Respostas IRPF' oficial.

O conteúdo é fictício/resumido: serve só para testar o parser sem depender de rede.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PAGINAS = [
    ["PERGUNTAS E RESPOSTAS", "Exercício de 2099", "Ano-calendário de 2098"],
    ["SUMÁRIO", "OBRIGATORIEDADE DE ENTREGA", "Obrigatoriedade 001", "Pessoa Física desobrigada 002",
     "DESCONTO SIMPLIFICADO", "Desconto simplificado - conceito 003", "Prazo de entrega 004"],
    [
        "OBRIGATORIEDADE DE ENTREGA",
        "OBRIGATORIEDADE",
        "001 — Quem está obrigado a apresentar a Declaração de Ajuste Anual relativa ao exercício de 2099,",
        "ano-calendário de 2098?",
        "Está obrigada a apresentar a Declaração a pessoa física residente no Brasil que, no ano-calendário:",
        "1 - recebeu rendimentos tributáveis cuja soma foi superior a R$ 100.000,00;",
        "2 - recebeu rendimentos isentos superiores a R$ 200.000,00.",
        "Atenção:",
        "A declaração deve ser feita pelo programa oficial.",
        "(Lei nº 9.250, de 26 de dezembro de 1995, art. 7º; e",
        "Instrução Normativa RFB nº 9.999, de 1º de janeiro de 2099, art. 2º)",
        "Retorno ao sumário",
        "PESSOA FÍSICA DESOBRIGADA",
        "002 — Pessoa física desobrigada pode apresentar a Declaração de Ajuste Anual (DAA)?",
        "Sim. A pessoa física, ainda que desobrigada, pode apresentar a DAA.",
        "(Instrução Normativa RFB nº 9.999, de 1º de janeiro de 2099, art. 2º, § 2º)",
        "Consulte as perguntas 001 e 161",
        "Retorno ao sumário",
    ],
    [
        "DESCONTO SIMPLIFICADO",
        "DESCONTO SIMPLIFICADO - CONCEITO",
        "003 — O que se considera como opção pelo desconto simplificado?",
        "A opção pelo desconto simplificado implica a substituição de todas as deduções admitidas na",
        "legislação tributária, correspondente à dedução de 20% do valor dos rendimentos tributáveis.",
        "O valor de R$ 100 - mil não é uma nova pergunta.",
        "100 - esta linha parece entrada mas está fora da sequência e deve ser ignorada.",
        "(Lei nº 9.250, de 26 de dezembro de 1995, art.10, inciso IX)",
        "Retorno ao sumário",
        "PRAZO DE ENTREGA",
        "004 —Qual é o prazo de apresentação da Declaração de Ajuste Anual?",
        "A Declaração deve ser apresentada de 16 de março a 31 de maio.",
        "Retorno ao sumário",
    ],
]


def gerar(caminho: Path, numeros_no_topo: bool = False, com_retorno: bool = True) -> Path:
    c = canvas.Canvas(str(caminho), pagesize=A4)
    largura, altura = A4
    for n, linhas in enumerate(PAGINAS, start=1):
        y_num = altura - 30 if numeros_no_topo else 30
        if numeros_no_topo:
            c.setFont("Helvetica", 9); c.drawString(largura / 2, y_num, str(n))
        y = altura - 60
        c.setFont("Helvetica", 9)
        for linha in linhas:
            if not com_retorno and linha == "Retorno ao sumário":
                continue
            c.drawString(40, y, linha)
            y -= 13
        if not numeros_no_topo:
            c.drawString(largura / 2, y_num, str(n))
        c.showPage()
    c.save()
    return caminho


if __name__ == "__main__":
    print(gerar(Path("/tmp/pr-irpf-2099.pdf")))
