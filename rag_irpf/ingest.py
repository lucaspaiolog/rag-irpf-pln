"""Ingestão: CSV -> Documents -> chunks -> embeddings -> índice vetorial Chroma.

Etapas (correspondem às etapas mínimas exigidas no trabalho):
  1. Leitura do CSV (base real gerada a partir do PDF oficial da Receita)
  2. Pré-processamento (limpeza, remoção de vazios/duplicados, tipos)
  3. Chunking (RecursiveCharacterTextSplitter)
  4. Geração de embeddings
  5. Armazenamento vetorial (Chroma, persistido em disco)

Uso:
    python -m rag_irpf.ingest                       # configuração padrão do .env
    python -m rag_irpf.ingest --config hf:500:100   # outra configuração (provider:chunk:overlap)
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .config import IndexConfig
from .models import get_embeddings

COLUNAS_OBRIGATORIAS = {"exercicio", "numero", "pergunta", "resposta"}


# 1 + 2) leitura e pré-processamento ----------------------------------------------------
def carregar_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")
    faltando = COLUNAS_OBRIGATORIAS - set(df.columns)
    if faltando:
        raise ValueError(f"CSV sem as colunas obrigatórias: {sorted(faltando)}")

    for col in df.columns:
        df[col] = df[col].map(lambda t: re.sub(r"[ \t]+", " ", t).strip())
    df["exercicio"] = df["exercicio"].astype(int)  # int: o filtro por metadado compara por igualdade
    df["numero"] = df["numero"].astype(int)

    antes = len(df)
    df = df[(df["pergunta"] != "") & (df["resposta"].str.len() >= 10)]
    df = df.drop_duplicates(subset=["exercicio", "numero"])
    print(f"Pré-processamento: {antes} linhas lidas -> {len(df)} válidas.")
    return df.reset_index(drop=True)


def dataframe_para_documentos(df: pd.DataFrame) -> list[Document]:
    docs = []
    for r in df.to_dict("records"):
        docs.append(
            Document(
                page_content=r["resposta"],
                metadata={
                    "exercicio": r["exercicio"],
                    "ano_calendario": int(r.get("ano_calendario") or r["exercicio"] - 1),
                    "numero": r["numero"],
                    "pergunta": r["pergunta"],
                    "secao": r.get("secao", ""),
                    "fundamentacao_legal": r.get("fundamentacao_legal", ""),
                    "perguntas_relacionadas": r.get("perguntas_relacionadas", ""),
                    "fonte": r.get("fonte_pdf", ""),
                },
            )
        )
    return docs


# 3) chunking ------------------------------------------------------------------------------
def dividir_em_chunks(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Divide cada RESPOSTA em pedaços e prefixa cada pedaço com a pergunta.

    O prefixo '[IRPF 2026 | Pergunta 001] <pergunta>' garante que todo chunk carregue o
    contexto de qual pergunta/edição ele pertence (um pedaço solto da resposta, sozinho,
    perde o sentido e piora a busca semântica).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in docs:
        m = doc.metadata
        cabecalho = f"[IRPF {m['exercicio']} | Pergunta {m['numero']:03d}] {m['pergunta']}"
        partes = splitter.split_text(doc.page_content) or [""]
        for i, parte in enumerate(partes):
            chunks.append(
                Document(
                    page_content=f"{cabecalho}\n{parte}",
                    metadata={**m, "chunk": i, "total_chunks": len(partes)},
                )
            )
    return chunks


# 4 + 5) embeddings e armazenamento vetorial -----------------------------------------------
def _abrir_chroma(cfg: IndexConfig) -> Chroma:
    return Chroma(
        collection_name="irpf",
        persist_directory=str(cfg.pasta),
        embedding_function=get_embeddings(cfg.provider, cfg.modelo),
        collection_metadata={"hnsw:space": "cosine"},  # distância de cosseno: 0 = idêntico, quanto menor melhor
    )


def construir_indice(csv_path: Path, cfg: IndexConfig, forcar: bool = False) -> Chroma:
    if cfg.pasta.exists() and not forcar:
        print(f"Índice '{cfg.nome}' já existe; carregando do disco (use --forcar para recriar).")
        return carregar_indice(cfg)
    if cfg.pasta.exists():
        shutil.rmtree(cfg.pasta)

    df = carregar_dataframe(csv_path)
    docs = dataframe_para_documentos(df)
    chunks = dividir_em_chunks(docs, cfg.chunk_size, cfg.chunk_overlap)
    print(f"Chunking: {len(docs)} respostas -> {len(chunks)} chunks "
          f"(tamanho {cfg.chunk_size}, sobreposição {cfg.chunk_overlap}).")

    print(f"Gerando embeddings com {cfg.provider}:{cfg.modelo} ...")
    cfg.pasta.mkdir(parents=True, exist_ok=True)
    vs = _abrir_chroma(cfg)
    ids = [f"{c.metadata['exercicio']}-{c.metadata['numero']:03d}-{c.metadata['chunk']}" for c in chunks]
    for i in range(0, len(chunks), 16):  # em lotes, para respeitar limites de API
        vs.add_documents(chunks[i:i + 16], ids=ids[i:i + 16])
    print(f"Índice salvo em {cfg.pasta} ({len(chunks)} vetores)")
    return vs


def carregar_indice(cfg: IndexConfig) -> Chroma:
    if not cfg.pasta.exists():
        raise FileNotFoundError(f"Índice '{cfg.nome}' não encontrado. Rode: python -m rag_irpf.ingest")
    return _abrir_chroma(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Constrói o índice vetorial a partir do CSV.")
    ap.add_argument("--csv", type=Path, default=config.CSV_PADRAO)
    ap.add_argument("--config", type=IndexConfig.de_texto, default=IndexConfig(),
                    help="provider:chunk:overlap[:modelo]  ex.: hf:1000:150")
    ap.add_argument("--forcar", action="store_true", help="recria o índice mesmo que já exista")
    args = ap.parse_args(argv)
    construir_indice(args.csv, args.config, args.forcar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
