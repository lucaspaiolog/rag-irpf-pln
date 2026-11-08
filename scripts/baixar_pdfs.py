"""Baixa os PDFs oficiais do 'Perguntas e Respostas IRPF' (gov.br/receitafederal).

    python scripts/baixar_pdfs.py

Se o download automático falhar (o gov.br às vezes bloqueia scripts), baixe manualmente em
https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/perguntas-e-respostas/dirpf
e salve em data/raw/ mantendo o ano no nome (ex.: pr-irpf-2024.pdf, p-r-irpf-2026.pdf).
"""
from pathlib import Path

import requests

BASE = "https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/perguntas-e-respostas/dirpf/"
PDFS = {
    "p-r-irpf-2026-v1-00-2026-04-23.pdf": BASE + "p-r-irpf-2026-v1-00-2026-04-23.pdf",  # exercício 2026 (v1.0)
    "pr-irpf-2024.pdf": BASE + "pr-irpf-2024.pdf",  # exercício 2024
}
DESTINO = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nome, url in PDFS.items():
        destino = DESTINO / nome
        if destino.exists():
            print(f"[ok] {nome} já existe")
            continue
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0 (trabalho-academico-rag)"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("a resposta não é um PDF")
            destino.write_bytes(r.content)
            print(f"[baixado] {nome} ({len(r.content) / 1e6:.1f} MB)")
        except Exception as e:
            print(f"[FALHOU] {nome}: {e}\n  -> baixe manualmente: {url}")


if __name__ == "__main__":
    main()
