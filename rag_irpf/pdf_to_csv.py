"""Converte o PDF oficial "Perguntas e Respostas IRPF" (Receita Federal) em CSV.

Estrutura observada nos PDFs oficiais (edições 2016-2026):

    SEÇÃO EM MAIÚSCULAS
    TÓPICO EM MAIÚSCULAS
    001 — Texto da pergunta (pode quebrar em várias linhas)?
    Texto da resposta...
    Atenção: ...
    (Lei nº ..., art. ...; Instrução Normativa RFB nº ...)        <- fundamentação legal
    Consulte as perguntas 001 e 161                               <- perguntas relacionadas
    Retorno ao sumário

Uso:
    python -m rag_irpf.pdf_to_csv data/raw/*.pdf -o data/processed/perguntas_respostas_irpf.csv

O ano-exercício é inferido do nome do arquivo (ex.: p-r-irpf-2026-v1-00.pdf -> 2026).
Se o nome não tiver o ano, informe com --exercicio (vale para todos os PDFs da chamada).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

RE_ENTRADA = re.compile(r"^\s*(\d{3})\s*[—–-]\s*(\S.*)$")
RE_RETORNO = re.compile(r"^\s*Retorno\s+ao\s+sum[áa]rio\s*$", re.IGNORECASE)
RE_CONSULTE = re.compile(r"^\s*Consulte\b", re.IGNORECASE)
RE_NUM_PAGINA = re.compile(r"^\s*\d{1,3}\s*$")


@dataclass
class Entrada:
    exercicio: int
    ano_calendario: int
    numero: int
    secao: str
    pergunta: str
    resposta: str
    fundamentacao_legal: str
    perguntas_relacionadas: str
    fonte_pdf: str

    @property
    def id(self) -> str:
        return f"{self.exercicio}-{self.numero:03d}"


# --------------------------------------------------------------------------- utilidades
def _proporcao_maiusculas(linha: str) -> float:
    letras = [c for c in linha if c.isalpha()]
    if not letras:
        return 0.0
    return sum(c.isupper() for c in letras) / len(letras)


def _eh_titulo(linha: str) -> bool:
    """Linhas de seção/tópico: quase só letras maiúsculas (ex.: 'DEDUÇÕES - DEPENDENTES')."""
    s = linha.strip()
    return len(s) >= 4 and _proporcao_maiusculas(s) >= 0.8 and not RE_ENTRADA.match(s)


def _limpar(texto: str) -> str:
    texto = texto.replace("\u00ad", "").replace("\xa0", " ")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return "\n".join(l.rstrip() for l in texto.splitlines()).strip()


def inferir_exercicio(caminho: Path, forcado: int | None = None) -> int:
    if forcado:
        return forcado
    m = re.search(r"irpf[-_ ]?(20\d{2})", caminho.name, re.IGNORECASE) or re.search(r"(20\d{2})", caminho.name)
    if not m:
        raise ValueError(f"Não consegui inferir o exercício de '{caminho.name}'. Use --exercicio.")
    return int(m.group(1))


# --------------------------------------------------------------------------- leitura do PDF
def extrair_linhas(caminho: Path) -> list[str]:
    """Lê o PDF página a página, removendo números de página soltos no topo/rodapé."""
    leitor = PdfReader(str(caminho))
    todas: list[str] = []
    for pagina in leitor.pages:
        linhas = (pagina.extract_text() or "").splitlines()
        # número de página aparece sozinho numa linha, no início ou no fim da página
        for pos in (0, -1):
            if linhas and RE_NUM_PAGINA.match(linhas[pos]):
                linhas.pop(pos)
        todas.extend(linhas)
    return todas


# --------------------------------------------------------------------------- parsing
def _localizar_entradas(linhas: list[str]) -> list[tuple[int, int]]:
    """Retorna [(índice_da_linha, número)] mantendo só uma sequência crescente plausível.

    O filtro de sequência evita falsos positivos (ex.: uma linha de resposta que
    comece com '100 - ...') e ignora o sumário, cujas linhas são 'Título 001'.
    """
    candidatos = [(i, int(m.group(1))) for i, l in enumerate(linhas) if (m := RE_ENTRADA.match(l))]
    escolhidos: list[tuple[int, int]] = []
    anterior: int | None = None
    for i, num in candidatos:
        if anterior is None:
            if num <= 3:
                escolhidos.append((i, num))
                anterior = num
        elif anterior < num <= anterior + 5:
            escolhidos.append((i, num))
            anterior = num
    return escolhidos


def _separar_pergunta(linhas_entrada: list[str]) -> tuple[str, list[str]]:
    """A primeira linha traz 'NNN — início da pergunta'. A pergunta vai até o '?'."""
    primeira = RE_ENTRADA.match(linhas_entrada[0]).group(2).strip()
    partes = [primeira]
    resto = linhas_entrada[1:]
    consumidas = 0
    while not partes[-1].rstrip().endswith("?") and consumidas < min(6, len(resto)):
        partes.append(resto[consumidas].strip())
        consumidas += 1
    if not partes[-1].rstrip().endswith("?"):  # não achou '?': considera só a primeira linha
        return primeira, resto
    return " ".join(partes), resto[consumidas:]


def _extrair_fundamentacao(linhas: list[str]) -> tuple[list[str], str]:
    """Separa o último bloco '( ... )' (base legal) do fim da resposta, se existir."""
    sem_vazias = [l for l in linhas if l.strip()]
    if not sem_vazias or not sem_vazias[-1].rstrip().endswith(")"):
        return linhas, ""
    saldo = 0
    for idx in range(len(sem_vazias) - 1, max(-1, len(sem_vazias) - 13), -1):
        saldo += sem_vazias[idx].count(")") - sem_vazias[idx].count("(")
        if saldo <= 0 and sem_vazias[idx].lstrip().startswith("("):
            bloco = " ".join(l.strip() for l in sem_vazias[idx:])
            return sem_vazias[:idx], bloco.strip("() ").strip()
    return linhas, ""


def _extrair_consulte(linhas: list[str]) -> tuple[list[str], str]:
    """Separa 'Consulte as perguntas ...' (sempre no fim da resposta)."""
    for idx in range(len(linhas) - 1, max(-1, len(linhas) - 5), -1):
        if RE_CONSULTE.match(linhas[idx]):
            return linhas[:idx], " ".join(l.strip() for l in linhas[idx:] if l.strip())
    return linhas, ""


def _remover_titulos_do_fim(linhas: list[str]) -> tuple[list[str], list[str]]:
    """Retira do fim do corpo as linhas de 'Retorno ao sumário' e os títulos da próxima entrada."""
    linhas = list(linhas)
    # Caso normal: existe "Retorno ao sumário" -> o que vem depois são títulos da próxima entrada.
    for idx in range(len(linhas) - 1, -1, -1):
        if RE_RETORNO.match(linhas[idx]):
            titulos_apos = [l.strip() for l in linhas[idx + 1:] if l.strip()]
            return linhas[:idx], titulos_apos
    # Plano B (edições sem "Retorno ao sumário"): descarta linhas em maiúsculas no fim do bloco.
    while linhas and not linhas[-1].strip():
        linhas.pop()
    titulos: list[str] = []
    while linhas and (_eh_titulo(linhas[-1]) or not linhas[-1].strip()):
        l = linhas.pop()
        if l.strip():
            titulos.insert(0, l.strip())
    return linhas, titulos


def parsear(linhas: list[str], exercicio: int, fonte_pdf: str) -> list[Entrada]:
    marcas = _localizar_entradas(linhas)
    if not marcas:
        return []

    # títulos da primeira entrada: sobem a partir da linha dela enquanto forem maiúsculas
    ini = marcas[0][0]
    titulos_anteriores: list[str] = []
    j = ini - 1
    while j >= 0 and len(titulos_anteriores) < 4 and (_eh_titulo(linhas[j]) or not linhas[j].strip()):
        if linhas[j].strip():
            titulos_anteriores.insert(0, linhas[j].strip())
        j -= 1

    entradas: list[Entrada] = []
    for pos, (inicio, numero) in enumerate(marcas):
        fim = marcas[pos + 1][0] if pos + 1 < len(marcas) else len(linhas)
        bloco = linhas[inicio:fim]

        corpo, titulos_proxima = _remover_titulos_do_fim(bloco)
        pergunta, resposta_linhas = _separar_pergunta(corpo)
        resposta_linhas, consulte = _extrair_consulte(resposta_linhas)
        resposta_linhas, fundamentacao = _extrair_fundamentacao(resposta_linhas)

        entradas.append(
            Entrada(
                exercicio=exercicio,
                ano_calendario=exercicio - 1,
                numero=numero,
                secao=" > ".join(titulos_anteriores),
                pergunta=_limpar(pergunta),
                resposta=_limpar("\n".join(resposta_linhas)),
                fundamentacao_legal=_limpar(fundamentacao),
                perguntas_relacionadas=_limpar(consulte),
                fonte_pdf=fonte_pdf,
            )
        )
        titulos_anteriores = titulos_proxima
    return entradas


def converter_pdf(caminho: Path, exercicio: int | None = None) -> list[Entrada]:
    ex = inferir_exercicio(caminho, exercicio)
    return parsear(extrair_linhas(caminho), ex, caminho.name)


# --------------------------------------------------------------------------- validação / CLI
def validar(entradas: list[Entrada], nome: str) -> list[str]:
    avisos: list[str] = []
    if not entradas:
        return [f"{nome}: nenhuma pergunta encontrada (o layout do PDF pode ser diferente)."]
    numeros = [e.numero for e in entradas]
    faltando = sorted(set(range(1, max(numeros) + 1)) - set(numeros))
    if faltando:
        avisos.append(f"{nome}: números ausentes na sequência: {faltando[:20]}{'...' if len(faltando) > 20 else ''}")
    sem_resposta = [e.numero for e in entradas if len(e.resposta) < 10]
    if sem_resposta:
        avisos.append(f"{nome}: respostas vazias/curtas nas perguntas {sem_resposta[:20]}")
    sem_interrogacao = [e.numero for e in entradas if not e.pergunta.rstrip().endswith("?")]
    if sem_interrogacao:
        avisos.append(f"{nome}: perguntas sem '?' (conferir quebra de linha): {sem_interrogacao[:20]}")
    return avisos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", type=Path, help="PDFs do Perguntas e Respostas IRPF")
    ap.add_argument("-o", "--saida", type=Path, default=Path("data/processed/perguntas_respostas_irpf.csv"))
    ap.add_argument("--exercicio", type=int, default=None, help="força o ano-exercício (se não estiver no nome do arquivo)")
    args = ap.parse_args(argv)

    todas: list[Entrada] = []
    for pdf in args.pdfs:
        entradas = converter_pdf(pdf, args.exercicio)
        print(f"{pdf.name}: {len(entradas)} perguntas (exercício {entradas[0].exercicio if entradas else '?'})")
        for aviso in validar(entradas, pdf.name):
            print("  AVISO:", aviso)
        todas.extend(entradas)

    if not todas:
        print("Nada extraído. Confira se o PDF tem texto selecionável (não é imagem escaneada).", file=sys.stderr)
        return 1

    df = pd.DataFrame([{"id": e.id, **asdict(e)} for e in todas])
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida, index=False, encoding="utf-8-sig")  # utf-8-sig abre certo no Excel
    print(f"\nCSV salvo em {args.saida} com {len(df)} linhas.")
    print(df.groupby("exercicio").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
