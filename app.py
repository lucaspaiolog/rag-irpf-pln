"""Interface web (Streamlit).  Execute:  streamlit run app.py"""
import streamlit as st

from rag_irpf import config
from rag_irpf.config import IndexConfig
from rag_irpf.rag import RagIRPF

st.set_page_config(page_title="Assistente do Imposto de Renda", page_icon="🧾", layout="wide")


@st.cache_resource(show_spinner="Carregando índice vetorial...")
def carregar_rag(provider: str, chunk: int, overlap: int, k: int, expandir: bool) -> RagIRPF:
    return RagIRPF(IndexConfig(provider=provider, chunk_size=chunk, chunk_overlap=overlap), k=k, expandir=expandir)


# ---------------------------- barra lateral ----------------------------
with st.sidebar:
    st.header("Configurações")
    provider = st.selectbox("Embeddings", ["hf", "openai", "google"],
                            index=["hf", "openai", "google"].index(config.EMBEDDING_PROVIDER)
                            if config.EMBEDDING_PROVIDER in ("hf", "openai", "google") else 0)
    chunk = st.number_input("Tamanho do chunk", 200, 4000, config.CHUNK_SIZE, step=100)
    overlap = st.number_input("Sobreposição", 0, 1000, config.CHUNK_OVERLAP, step=25)
    k = st.slider("Trechos recuperados (k)", 1, 12, config.TOP_K)
    expandir = st.checkbox("Expandir para a resposta completa", value=True,
                           help="Após achar os chunks, envia ao LLM a resposta inteira de cada pergunta.")

try:
    rag = carregar_rag(provider, int(chunk), int(overlap), k, expandir)
except FileNotFoundError as e:
    st.error(f"{e}\n\nGere o CSV e o índice antes (veja o README): `python -m rag_irpf.ingest`")
    st.stop()

with st.sidebar:
    opcoes = rag.exercicios_disponiveis()
    exercicio_sel = st.selectbox("Ano-exercício da consulta", opcoes + ["Todas as edições"], index=0,
                                 help="Filtra a busca pelos metadados: a resposta usa só a edição escolhida.")
    st.caption(f"LLM: {config.LLM_PROVIDER} · Base: Perguntas e Respostas IRPF (Receita Federal)")
exercicio = None if exercicio_sel == "Todas as edições" else int(exercicio_sel)

# ---------------------------- tela principal ----------------------------
st.title("🧾 Assistente de dúvidas sobre o Imposto de Renda")
st.caption("Respostas geradas a partir do documento oficial **Perguntas e Respostas IRPF** da Receita Federal. "
           "Informação de apoio — não substitui orientação de contador ou da Receita.")

if "historico" not in st.session_state:
    st.session_state.historico = []

for item in st.session_state.historico:
    with st.chat_message("user"):
        st.write(item["pergunta"])
    with st.chat_message("assistant"):
        st.write(item["resposta"])

pergunta = st.chat_input("Ex.: Quem é obrigado a declarar o Imposto de Renda?")
if pergunta:
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        with st.spinner("Buscando na base e gerando resposta..."):
            try:
                r = rag.responder(pergunta, exercicio)
            except Exception as e:  # chave de API ausente, falta de rede etc.
                st.error(f"Erro ao gerar a resposta: {e}")
                st.stop()
        st.write(r.resposta)
        if r.fontes:
            with st.expander(f"📚 Fontes recuperadas ({len({(f['exercicio'], f['numero']) for f in r.fontes})} perguntas)"):
                vistos = set()
                for f in r.fontes:
                    chave = (f["exercicio"], f["numero"])
                    if chave in vistos:
                        continue
                    vistos.add(chave)
                    st.markdown(f"**P&R {f['exercicio']} · pergunta {f['numero']:03d}** — {f['pergunta']}")
                    st.caption(f"Seção: {f['secao'] or '—'} · distância: {f['score']:.3f} · arquivo: {f['fonte']}")
                    st.text(f["trecho"][:1200])
                    if f.get("fundamentacao_legal"):
                        st.caption(f"Fundamentação legal: {f['fundamentacao_legal']}")
                    st.divider()
    st.session_state.historico.append({"pergunta": pergunta, "resposta": r.resposta})
