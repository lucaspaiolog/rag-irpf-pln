# 🧾 Assistente RAG — Dúvidas sobre o Imposto de Renda (IRPF)

Trabalho prático de PLN (Prof. Vagner Macedo) — **Tema 4: Dúvidas sobre o Imposto de Renda**.

Sistema de **RAG (Retrieval-Augmented Generation)** com **LangChain** que responde perguntas em
linguagem natural usando como base o documento oficial **"Perguntas e Respostas IRPF"** da
Receita Federal (745 perguntas na edição 2026). O sistema só responde com base nos trechos
recuperados e **admite quando a base não tem a resposta**.

> Desafio técnico do tema: *atualização da base e filtro por ano-exercício.* A base contém
> várias edições (2026, 2024…) e cada trecho carrega o metadado `exercicio`; a busca é filtrada
> pela edição escolhida, evitando misturar regras de anos diferentes.

---

## Arquitetura

```
PDF oficial (Receita)            ┌──────────────────────── pipeline offline ─────────────────────────┐
  p-r-irpf-2026.pdf ──► pdf_to_csv.py ──► CSV ──► ingest.py:  limpeza → chunking → embeddings → Chroma
  pr-irpf-2024.pdf                                  (data/processed/…csv)                  (indices/)
                                                                                              │
Pergunta ──► filtro por exercício ──► busca vetorial (top-k) ──► [expansão p/ resposta completa] ──┤
                                                                         │                        │
                                                          prompt + LLM (LangChain, LCEL) ◄────────┘
                                                                         │
                                                       resposta com citações [P&R 2026, pergunta 001]
```

| Etapa exigida            | Onde está                                                              |
|--------------------------|------------------------------------------------------------------------|
| Ingestão (leitura)       | `rag_irpf/pdf_to_csv.py` (PDF→CSV) e `ingest.carregar_dataframe`       |
| Pré-processamento        | `ingest.carregar_dataframe` (limpeza, tipos, duplicados, vazios)       |
| Chunking                 | `ingest.dividir_em_chunks` (`RecursiveCharacterTextSplitter`)          |
| Embeddings               | `models.get_embeddings` (HuggingFace local, OpenAI ou Google)          |
| Armazenamento vetorial   | **Chroma** persistido em `indices/` (`ingest.construir_indice`)        |
| Retrieval                | `rag.RagIRPF.recuperar` (top-k + filtro de metadado `exercicio`)       |
| Geração                  | `rag.RagIRPF.responder` (`PROMPT → LLM → StrOutputParser`)            |
| Interface                | `app.py` (Streamlit) e `cli.py` (linha de comando)                     |
| Avaliação/experimentos   | `rag_irpf/evaluate.py` + `eval/perguntas_avaliacao.csv`                |

---

## Base de dados (real e externa)

**Fonte:** Receita Federal do Brasil — *Perguntas e Respostas IRPF*.
Página oficial com todas as edições (PDF):
<https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/perguntas-e-respostas/dirpf>

A Receita **não disponibiliza essa base em CSV**, apenas em PDF. Por isso o projeto inclui um
conversor (`pdf_to_csv.py`) que lê o PDF oficial e gera um **CSV estruturado**:

| coluna                   | conteúdo                                              |
|--------------------------|-------------------------------------------------------|
| `id`                     | `2026-001` (exercício + nº da pergunta)               |
| `exercicio`, `ano_calendario` | ex.: 2026 e 2025                                 |
| `numero`                 | nº da pergunta na edição                              |
| `secao`                  | capítulo/tópico do documento                          |
| `pergunta`, `resposta`   | texto oficial                                         |
| `fundamentacao_legal`    | base legal citada pela Receita                        |
| `perguntas_relacionadas` | "Consulte as perguntas …"                             |
| `fonte_pdf`              | arquivo de origem                                     |

> **Licença/uso:** o documento informa que a reprodução é permitida desde que citada a fonte.
> Não há dados pessoais na base. Cite sempre a Receita Federal como fonte no relatório.

---

## Como executar

### 0) Pré-requisitos
Python 3.10+ e (para o LLM) uma chave de API **ou** o Ollama instalado (modelo local, sem custo).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env  → edite e coloque sua chave
```

### 1) Baixar os PDFs oficiais
```bash
python scripts/baixar_pdfs.py
```
Se o download automático falhar, baixe manualmente na página da Receita (link acima) e salve em
`data/raw/` **mantendo o ano no nome** (ex.: `p-r-irpf-2026.pdf`, `pr-irpf-2024.pdf`).

### 2) Converter os PDFs em CSV
```bash
python -m rag_irpf.pdf_to_csv data/raw/*.pdf -o data/processed/perguntas_respostas_irpf.csv
```
**Confira a saída:** a edição 2026 deve ter **745** perguntas. O script imprime `AVISO` se
faltar número na sequência, houver resposta vazia ou pergunta sem `?`. Abra o CSV no Excel e
confira 3 ou 4 linhas contra o PDF.

### 3) Construir o índice vetorial
```bash
python -m rag_irpf.ingest
```
A primeira execução com embeddings `hf` baixa o modelo (cerca de 0,5 GB) uma única vez.

### 4) Usar o sistema
```bash
streamlit run app.py                                   # interface web (recomendada)
python cli.py "Quem é obrigado a declarar o IR?"       # linha de comando
python cli.py --exercicio 2024 "Qual o prazo de entrega?"
python cli.py                                          # modo interativo
```
Na interface você escolhe o **ano-exercício**, o `k`, o tamanho do chunk e vê as **fontes**
(número da pergunta, seção, distância, fundamentação legal) de cada resposta.

### 5) Avaliação e experimentos (comparar configurações)
```bash
# compara dois tamanhos de chunk com o mesmo embedding
python -m rag_irpf.evaluate --configs hf:1000:150,hf:500:100 --k 5

# compara dois modelos de embedding (precisa de OPENAI_API_KEY)
python -m rag_irpf.evaluate --configs hf:1000:150,openai:1000:150 --k 5 --gerar-respostas
```
Métricas reportadas por configuração: **hit@k**, **MRR**, tempo de busca e % de perguntas
fora do escopo em que o sistema **admitiu não saber**. Com `--gerar-respostas`, o CSV de saída
(`eval/resultados_*.csv`) traz a resposta do LLM e uma coluna `fiel_ao_contexto` para preencher
manualmente (1 = fiel à base, 0 = inventou) — é a avaliação de *fidelidade*.

O conjunto de avaliação (`eval/perguntas_avaliacao.csv`) tem **20 perguntas respondíveis e 5
fora do escopo**. Preencha a coluna `resposta_referencia` lendo o PDF e ajuste `palavras_chave`
se algum termo não existir na edição que você usou (formato: `;` = todos obrigatórios,
`|` = alternativas).

### 6) Testes automatizados
```bash
python -m pytest tests -q
```
Os testes usam um PDF sintético e embeddings/LLM **falsos** (offline); validam o parser e a
"fiação" do pipeline, **não** a qualidade semântica da busca (isso é papel do `evaluate.py`).

---

## Decisões de projeto

* **Chunk com cabeçalho:** cada chunk é prefixado com `[IRPF 2026 | Pergunta 001] <pergunta>`,
  para que um pedaço de resposta nunca perca a pergunta a que pertence.
* **Filtro por exercício nativo:** o Chroma filtra por metadado *antes* da busca vetorial.
* **Expansão de contexto (small-to-big):** acha-se pelos chunks, mas o LLM recebe a resposta
  **completa** das perguntas encontradas (desligável na interface e com `expandir=False`).
* **Prompt restritivo:** responde só com o contexto, cita `[P&R ano, pergunta NNN]` e usa uma
  frase fixa quando não há resposta (permite medir "admite que não sabe").
* **Provedores intercambiáveis:** troque LLM/embeddings pelo `.env`, sem mexer no código.
* **Só `langchain-core` + pacotes independentes** (`langchain-chroma`, etc.), sem depender do
  `langchain-community`, que está em descontinuação.

## Limitações conhecidas

* O parser foi validado no layout oficial (estrutura `NNN — pergunta?` … `Retorno ao sumário`),
  mas se a Receita mudar o layout de uma edição, o script avisa e pode precisar de ajuste.
* Tabelas (ex.: tabela progressiva) viram texto corrido; a leitura de números exige conferência.
* O sistema é informativo e **não substitui** contador ou orientação oficial da Receita.

## Estrutura

```
app.py  cli.py  requirements.txt  .env.example
rag_irpf/   config.py  models.py  pdf_to_csv.py  ingest.py  rag.py  evaluate.py
scripts/    baixar_pdfs.py
eval/       perguntas_avaliacao.csv
data/       raw/ (PDFs)   processed/ (CSV)
tests/      test_parser.py  test_pipeline.py  make_sample_pdf.py
```

## Fontes

* Receita Federal do Brasil. *Perguntas e Respostas IRPF* (edições 2024 e 2026).
* Júnior, J. D. et al. *BR-TaxQA-R: A Dataset for Question Answering with References for Brazilian
  Personal Income Tax Law.* arXiv:2505.15916, 2025 (referência de trabalho relacionado).
