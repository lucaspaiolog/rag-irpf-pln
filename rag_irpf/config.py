"""Configurações centrais do projeto.

Tudo que pode mudar entre máquinas (provedor de LLM, chaves, nomes de modelos)
é lido de variáveis de ambiente / arquivo .env. Veja .env.example.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAIZ = Path(__file__).resolve().parent.parent
CSV_PADRAO = RAIZ / "data" / "processed" / "perguntas_respostas_irpf.csv"
PASTA_INDICES = RAIZ / "indices"
PASTA_PDFS = RAIZ / "data" / "raw"

# Frase fixa usada quando a base não contém a resposta. A avaliação procura
# por ela para medir se o sistema "admite quando não sabe".
FRASE_SEM_RESPOSTA = "Não encontrei essa informação nas Perguntas e Respostas da Receita Federal"


def _env(nome: str, padrao: str) -> str:
    return os.getenv(nome, padrao).strip()


# ---- Provedores e modelos (todos sobrescrevíveis via .env) -------------------
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "hf")  # hf | openai | google | fake
HF_EMBEDDING_MODEL = _env("HF_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OPENAI_EMBEDDING_MODEL = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GOOGLE_EMBEDDING_MODEL = _env("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")

LLM_PROVIDER = _env("LLM_PROVIDER", "openai")  # openai | anthropic | google | ollama | fake
OPENAI_LLM_MODEL = _env("OPENAI_LLM_MODEL", "gpt-4o-mini")
ANTHROPIC_LLM_MODEL = _env("ANTHROPIC_LLM_MODEL", "claude-haiku-5-5")
GOOGLE_LLM_MODEL = _env("GOOGLE_LLM_MODEL", "gemini-2.5-flash")
OLLAMA_LLM_MODEL = _env("OLLAMA_LLM_MODEL", "llama3.1")
# Vazio = não envia o parâmetro (necessário para modelos que não aceitam temperature).
LLM_TEMPERATURE = _env("LLM_TEMPERATURE", "0")

# ---- Parâmetros do pipeline ----------------------------------------------------
CHUNK_SIZE = int(_env("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "150"))
TOP_K = int(_env("TOP_K", "5"))


def _slug(texto: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")


def modelo_embedding_padrao(provider: str) -> str:
    return {
        "hf": HF_EMBEDDING_MODEL,
        "openai": OPENAI_EMBEDDING_MODEL,
        "google": GOOGLE_EMBEDDING_MODEL,
        "fake": "fake-256",
    }[provider]


@dataclass(frozen=True)
class IndexConfig:
    """Uma 'configuração experimental' = provedor/modelo de embedding + chunking.

    Cada configuração gera um índice vetorial separado em disco, o que permite
    comparar configurações no evaluate.py (exigência de experimentação do trabalho).
    """

    provider: str = EMBEDDING_PROVIDER
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    model: str = ""

    @property
    def modelo(self) -> str:
        return self.model or modelo_embedding_padrao(self.provider)

    @property
    def nome(self) -> str:
        return f"{self.provider}_{_slug(self.modelo)}_c{self.chunk_size}_o{self.chunk_overlap}"

    @property
    def pasta(self) -> Path:
        return PASTA_INDICES / self.nome

    @classmethod
    def de_texto(cls, texto: str) -> "IndexConfig":
        """Lê 'provider:chunk:overlap[:modelo]'. Ex.: 'hf:1000:150' ou 'openai:500:100'."""
        partes = texto.split(":")
        if len(partes) < 3:
            raise ValueError(f"Configuração inválida '{texto}'. Use provider:chunk:overlap[:modelo]")
        return cls(provider=partes[0], chunk_size=int(partes[1]), chunk_overlap=int(partes[2]),
                   model=":".join(partes[3:]))
