"""Fábrica de LLMs e de modelos de embeddings.

Os imports são feitos dentro das funções (lazy) para que você só precise instalar
o pacote do provedor que realmente for usar.
"""
from __future__ import annotations

from . import config


def get_embeddings(provider: str | None = None, model: str | None = None):
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.modelo_embedding_padrao(provider)

    if provider == "hf":  # roda local, gratuito, multilíngue (funciona bem em português)
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model)
    if provider == "fake":  # SOMENTE para testes automatizados (vetores sem significado)
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=256)
    raise ValueError(f"EMBEDDING_PROVIDER desconhecido: {provider}")


def get_llm(provider: str | None = None):
    provider = (provider or config.LLM_PROVIDER).lower()
    kwargs = {}
    if config.LLM_TEMPERATURE != "":
        kwargs["temperature"] = float(config.LLM_TEMPERATURE)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.OPENAI_LLM_MODEL, **kwargs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.ANTHROPIC_LLM_MODEL, **kwargs)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.GOOGLE_LLM_MODEL, **kwargs)
    if provider == "ollama":  # modelo local, sem custo e sem chave de API
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.OLLAMA_LLM_MODEL, **kwargs)
    if provider == "fake":  # SOMENTE para testes automatizados
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        return FakeListChatModel(responses=["Resposta de teste."])
    raise ValueError(f"LLM_PROVIDER desconhecido: {provider}")
