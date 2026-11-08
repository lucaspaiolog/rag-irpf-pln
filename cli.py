"""Interface de linha de comando.

    python cli.py "Quem é obrigado a declarar?"          # pergunta única
    python cli.py                                          # modo interativo
    python cli.py --exercicio 2024 --k 8 "Qual o prazo?"
    python cli.py --exercicio todos "..."                  # busca em todas as edições
"""
import argparse

from rag_irpf import config
from rag_irpf.config import IndexConfig
from rag_irpf.rag import RagIRPF


def imprimir(r):
    print(f"\n{r.resposta}\n")
    if r.fontes:
        print("Fontes recuperadas:")
        vistos = set()
        for f in r.fontes:
            chave = (f["exercicio"], f["numero"])
            if chave in vistos:
                continue
            vistos.add(chave)
            print(f"  - P&R {f['exercicio']}, pergunta {f['numero']:03d} (distância {f['score']:.3f}): {f['pergunta'][:90]}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Assistente RAG - dúvidas sobre o Imposto de Renda")
    ap.add_argument("pergunta", nargs="?", help="se omitida, abre o modo interativo")
    ap.add_argument("--exercicio", default="auto", help="ano-exercício (ex.: 2026), 'todos' ou 'auto' (mais recente)")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--config", type=IndexConfig.de_texto, default=IndexConfig(),
                    help="provider:chunk:overlap[:modelo] (padrão: .env)")
    args = ap.parse_args()

    exercicio = None if args.exercicio == "todos" else ("auto" if args.exercicio == "auto" else int(args.exercicio))
    rag = RagIRPF(args.config, k=args.k)
    print(f"Edições na base: {rag.exercicios_disponiveis()}")

    if args.pergunta:
        imprimir(rag.responder(args.pergunta, exercicio))
        return
    print("Digite sua dúvida sobre o IRPF (ou 'sair').")
    while True:
        p = input("\nPergunta> ").strip()
        if p.lower() in ("sair", "exit", "q"):
            break
        if p:
            imprimir(rag.responder(p, exercicio))


if __name__ == "__main__":
    main()
