# Stroke Prediction — Assistente Médico Virtual (AVC)

**FIAP Pós-Tech IA para Devs — Tech Challenge**

Sistema de apoio à decisão clínica para AVC, construído em três fases sobre o mesmo
dataset e o mesmo domínio: da predição de risco por machine learning até um
assistente médico virtual com LLM fine-tuned, RAG, guardrails de segurança e
fluxo de decisão automatizado.

| Fase | Entrega | Módulos |
|------|---------|---------|
| 1 | Modelos baseline de predição de AVC | `src/preprocessing.py`, `src/models.py` |
| 2 | Algoritmo Genético para otimizar hiperparâmetros + LLM para interpretar resultados | `src/genetic_algorithm/`, `src/llm/` |
| **3 (entrega atual)** | **Fine-tuning (LoRA) + assistente LangChain com RAG + fluxo LangGraph + segurança/auditoria** | `src/finetuning/`, `src/assistant/`, `src/security/`, `data/medical_corpus/` |

> **Fase 3:** ver [a seção completa abaixo](#fase-3--assistente-médico-fine-tuning--langchain--langgraph),
> o relatório técnico em [`docs/relatorio_tecnico_fase3.md`](docs/relatorio_tecnico_fase3.md)
> e os diagramas em [`docs/architecture.md`](docs/architecture.md).

## Contexto

Na Fase 1, treinamos dois modelos para prever risco de AVC:

| Modelo | Accuracy | Recall | F1 |
|--------|----------|--------|----|
| Regressão Logística | 83.37% | **44.00%** | 0.21 |
| Random Forest | 92.86% | 24.00% | 0.25 |

O **Recall** é a métrica principal (contexto médico: minimizar casos não detectados).

Na Fase 2, aplicamos Algoritmos Genéticos para otimizar os hiperparâmetros e
integramos um LLM para interpretar os resultados. Na Fase 3, esse modelo de risco
passou a ser **uma das etapas** do fluxo clínico automatizado do assistente médico.

---

## Estrutura do Repositório

```
stroke-prediction/
├── data/
│   ├── download_data.py            # Script para baixar o dataset do Kaggle
│   └── medical_corpus/             # [F3] Corpus médico (build, anonimização, curadoria)
├── src/
│   ├── preprocessing.py            # Pipeline de pré-processamento (Fase 1 refatorada)
│   ├── models.py                   # Treinamento, avaliação, save/load, predict
│   ├── genetic_algorithm/          # Motor do AG (Fase 2)
│   ├── llm/                        # Integração LLM / Gemini (Fase 2)
│   ├── finetuning/                 # [F3] Dataset e avaliação do fine-tuning
│   ├── assistant/                  # [F3] Assistente: RAG, prontuário, chain, grafo
│   └── security/                   # [F3] Guardrails e log de auditoria
├── notebooks/
│   ├── 01_baseline.ipynb           # Reprodução da Fase 1 (linha de base)
│   ├── 02_genetic_algorithm.ipynb  # Experimentos com AG
│   ├── 03_llm_integration.ipynb    # Demonstração da integração LLM
│   ├── 04_finetuning.ipynb         # [F3] Fine-tuning LoRA (Colab, GPU)
│   ├── 05_langchain_assistant.ipynb# [F3] Assistente LangChain (RAG + prontuário)
│   └── 06_langgraph_flow.ipynb     # [F3] Fluxo de decisão automatizado
├── results/
│   ├── logistic_regression.joblib  # Modelo LR treinado
│   ├── random_forest.joblib        # Modelo RF treinado
│   ├── baseline_metrics.json       # Métricas da linha de base
│   ├── ga_summary.json             # Baseline vs. otimizado pelo AG
│   └── finetuning/                 # [F3] Adapter LoRA treinado + métricas
├── tests/                          # 111 testes (pytest)
├── docs/
│   ├── architecture.md             # Diagramas e decisões técnicas (Fases 2 e 3)
│   └── relatorio_tecnico_fase3.md  # Relatório técnico da Fase 3
├── .env.example
└── requirements.txt
```

---

## Pré-requisitos

- **Python 3.10+** — obrigatório. O estado do grafo em `src/assistant/graph.py` usa
  a sintaxe de tipos `dict | None` (PEP 604), que o LangGraph resolve em tempo de
  execução; em Python 3.9 o grafo falha ao ser construído.
- Conta no Kaggle (para baixar o dataset)
- API key do Google Gemini (usada na Fase 2 e como fallback do assistente)
- Conta Google com Drive (opcional — só para rodar os notebooks no Colab)
- GPU (opcional — só para **re-treinar** o fine-tuning; o Colab fornece uma gratuita)

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repo>
cd stroke-prediction

# 2. Crie e ative um ambiente virtual (Python 3.10+)
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Se aparecer erro de política de execução, rode uma vez (libera só para o seu usuário):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Windows (Git Bash):
source venv/Scripts/activate

# Linux/Mac:
source venv/bin/activate

# Para desativar (qualquer shell): deactivate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure variáveis de ambiente
cp .env.example .env    # Windows: copy .env.example .env
# Edite .env e adicione sua GOOGLE_API_KEY
```

> O `.env` fica na **raiz do projeto** — é onde `src/llm/client.py` procura a
> `GOOGLE_API_KEY`. Ao rodar no Colab, ele precisa estar dentro da pasta do
> projeto no Drive (ver [Opção B](#opção-b--google-colab-via-drive-pessoal)).

---

## Dataset

O dataset utilizado é o [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) do Kaggle.

```bash
python data/download_data.py
```

Isso salva o CSV em `data/healthcare-dataset-stroke-data.csv`.

> O CSV não está no repositório (arquivo grande). Rode o script acima após clonar.

---

## Executando os notebooks — local ou Google Colab

**Todos os notebooks rodam nos dois ambientes sem edição manual.** A primeira
célula de cada um detecta onde está rodando e ajusta o caminho raiz do projeto
(`ROOT`) sozinha:

```python
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/stroke-prediction')
    sys.path.insert(0, str(ROOT))
    print("Rodando no Google Colab")
else:
    ROOT = Path('..').resolve()
    sys.path.insert(0, str(ROOT))
    print("Rodando Localmente")
```

A célula imprime `Rodando Localmente` ou `Rodando no Google Colab` — confira essa
saída antes de seguir, é a validação de que o ambiente foi detectado corretamente.

### Opção A — Localmente

Com o `venv` ativado e as dependências instaladas, suba o Jupyter **a partir da
raiz do projeto**:

```bash
jupyter notebook
```

Abra o notebook desejado dentro da pasta `notebooks/`. O caminho local é
resolvido como `ROOT = Path('..')`, ou seja, **o notebook precisa continuar dentro
de `notebooks/`** — se for movido para outro lugar, o `import src...` quebra.

> Sem GPU, o notebook `04_finetuning.ipynb` é impraticável localmente (ver Opção B).
> Os demais rodam em CPU normalmente.

### Opção B — Google Colab (via Drive pessoal)

No Colab o projeto **não é clonado**: ele é lido da sua conta do Google Drive.
A célula de setup monta o Drive e espera encontrar o projeto em
`MyDrive/stroke-prediction`. Passo a passo:

1. **Suba a pasta do projeto para o seu Drive pessoal.** Acesse
   [drive.google.com](https://drive.google.com) e arraste a pasta inteira do
   projeto para a raiz do **Meu Drive** (`MyDrive`).

2. **Mantenha o nome exato da pasta: `stroke-prediction`.** O caminho esperado
   pelos notebooks é `/content/drive/MyDrive/stroke-prediction`. Se a pasta for
   renomeada, ficar dentro de uma subpasta ou for para um Drive compartilhado,
   ajuste a variável `ROOT` na célula de setup.

3. **Inclua os arquivos que não vêm pelo Git.** Estes estão no `.gitignore` e
   precisam ser colocados manualmente dentro da pasta no Drive:
   - `.env` com a `GOOGLE_API_KEY` (a raiz do projeto é onde `src/llm/client.py` procura);
   - `data/healthcare-dataset-stroke-data.csv` (ou rode `data/download_data.py` no próprio Colab);
   - `results/*.joblib` — os modelos da Fase 1, se quiser pular o notebook 01.

4. **Abra o notebook no Colab.** Pelo Drive: clique com o botão direito no
   arquivo `.ipynb` → *Abrir com* → *Google Colaboratory*. Assim o notebook
   executado é o mesmo arquivo da pasta do projeto.

5. **Para o notebook 04, ative a GPU:** *Ambiente de execução* → *Alterar o tipo
   de ambiente de execução* → *T4 GPU*.

6. **Execute a primeira célula e autorize o acesso ao Drive.** O Colab abre um
   popup de permissão — é o `drive.mount()`. Depois de autorizar, a célula
   instala as dependências a partir do `requirements.txt` da própria pasta do
   Drive e imprime `Rodando no Google Colab`.

> **Por que Drive e não `git clone`:** tudo que os notebooks gravam
> (`results/finetuning/lora_adapter/`, `results/audit_log.jsonl`, índice FAISS,
> `.joblib`) fica salvo na pasta do Drive e **persiste depois que a sessão do
> Colab termina** — com um clone em `/content/` os artefatos seriam perdidos. É
> assim que o adapter treinado no notebook 04 fica disponível para os notebooks
> 05 e 06, e é de lá que você baixa a pasta de volta para o repositório local.

### Ordem de execução

| # | Notebook | Ambiente | Depende de |
|---|----------|----------|------------|
| 01 | `01_baseline.ipynb` — modelos baseline da Fase 1 | Local ou Colab | CSV do Kaggle |
| 02 | `02_genetic_algorithm.ipynb` — experimentos com AG | Local ou Colab | 01 |
| 03 | `03_llm_integration.ipynb` — explicações via Gemini | Local ou Colab | 01, 02 e `GOOGLE_API_KEY` |
| 04 | `04_finetuning.ipynb` — fine-tuning LoRA | **Colab com GPU** | `data/medical_corpus/train.jsonl` |
| 05 | `05_langchain_assistant.ipynb` — assistente LangChain | Local ou Colab | 01 e 04 (ou fallback Gemini) |
| 06 | `06_langgraph_flow.ipynb` — fluxo de decisão completo | Local ou Colab | 01 e 04 (ou fallback Gemini) |

---

## Usando os módulos em Python

```python
# Carregar modelo treinado e gerar predições
from src.models import load_model, predict
from src.preprocessing import prepare_pipeline

_, X_test, _, y_test = prepare_pipeline()
model = load_model("logistic_regression")
predictions, probabilities = predict(model, X_test)

# Rodar pipeline completo do zero
from src.preprocessing import prepare_pipeline
X_train, X_test, y_train, y_test = prepare_pipeline()
```

### Integração com LLM

O módulo `src/llm` usa a API do Google Gemini (`google-genai`) para gerar
explicações em linguagem natural a partir dos diagnósticos e dos resultados
do algoritmo genético. Configure `GOOGLE_API_KEY` no `.env` (copie
`.env.example`); Estamos utilizando o modelo `gemini-3.5-flash`.

```python
import json

from src.llm import evaluate_explanation, explain_prediction, summarize_experiment

# Explicação individual de uma predição
patient_data = X_test.iloc[0].to_dict()
prediction, probability = int(predictions[0]), float(probabilities[0])

explanation = explain_prediction(patient_data, prediction, probability)
quality = evaluate_explanation(explanation, patient_data)
print(explanation)
print(quality["score"], quality["checks"])

# Interpretação agregada de um experimento do algoritmo genético
ga_summary = json.load(open("results/ga_summary.json"))
exp = ga_summary["experiments"][0]

summary = summarize_experiment(
    baseline_metrics=ga_summary["baseline"][exp["model_type"]],
    optimized_metrics=exp["optimized_metrics"],
    best_params=exp["ga"]["best_params"],
)
```

**Avaliação de qualidade:** `evaluate_explanation(text, source_data)` roda
um checklist determinístico — sem chamar a API — com 4 regras: grounding
numérico nos dados de entrada, menção ao recall quando a métrica está
presente em `source_data`, uso de linguagem de risco/probabilidade (em vez
de afirmação categórica de diagnóstico) e tamanho razoável da resposta.
Retorna `{"score": float, "checks": {regra: bool}}`. Ver
`notebooks/03_llm_integration.ipynb` para uma demonstração completa,
incluindo as notas de prompt engineering usadas.

---

## Fase 3 — Assistente Médico (Fine-tuning + LangChain + LangGraph)

**FIAP Pós-Tech IA para Devs — Tech Challenge Fase 3**

Sobre a base da Fase 2, a Fase 3 adiciona: fine-tuning (LoRA/PEFT) de um LLM
com dados médicos internos (protocolos, FAQs, laudos), um assistente
LangChain com RAG sobre esses dados + consulta a um "prontuário" estruturado,
guardrails de segurança e logging de auditoria, e um fluxo de decisão
automatizado em LangGraph. Domínio: AVC/stroke, para manter coerência com o
dataset e os modelos já usados nas Fases 1–2.

### Requisitos do desafio → onde estão implementados

| Requisito da Fase 3 | Implementação | Demonstração |
|---------------------|---------------|--------------|
| **1.** Fine-tuning de LLM com dados médicos internos | `src/finetuning/`, adapter LoRA sobre `Qwen2.5-1.5B-Instruct` em `results/finetuning/lora_adapter/` | `notebooks/04_finetuning.ipynb` |
| **1.** Preprocessing, anonimização e curadoria dos dados | `data/medical_corpus/preprocessing.py` (scrub de PII, dedupe, filtros de tamanho, split determinístico) | `data/medical_corpus/build_corpus.py` |
| **2.** Pipeline LangChain integrando a LLM customizada | `src/assistant/chain.py` + `src/assistant/llm_backend.py` | `notebooks/05_langchain_assistant.ipynb` |
| **2.** Consulta a base de dados estruturada (prontuários) | `src/assistant/patient_db.py` (SQLite) | `notebooks/05`, `notebooks/06` |
| **2.** Contextualização com informações atualizadas do paciente | `src/assistant/prompts.py` + `src/assistant/tools.py` (risco de AVC + exames pendentes) | `notebooks/05`, `notebooks/06` |
| **3.** Limites de atuação (nunca prescrever sem validação humana) | `src/security/guardrails.py` | Cenário 3 do `notebooks/06` |
| **3.** Logging detalhado para rastreamento e auditoria | `src/security/audit_log.py` → `results/audit_log.jsonl` | Última célula do `notebooks/06` |
| **3.** Explainability (indicar a fonte da informação) | `sources` retornados pelo RAG em toda resposta (`src/assistant/retriever.py`) | `notebooks/05`, `notebooks/06` |
| **Entregável.** Fluxos do LangGraph | `src/assistant/graph.py` | `notebooks/06_langgraph_flow.ipynb` |
| **Entregável.** Dataset anonimizado / sintético | `data/medical_corpus/train.jsonl` (510) e `eval.jsonl` (89) | — |
| **Entregável.** Relatório técnico + diagrama do fluxo | [`docs/relatorio_tecnico_fase3.md`](docs/relatorio_tecnico_fase3.md), [`docs/architecture.md`](docs/architecture.md) | — |
| **4.** Código modularizado + instruções no README | `src/` por responsabilidade, 111 testes em `tests/` | este arquivo |

### Estrutura adicionada

```
data/medical_corpus/
├── preprocessing.py        # limpeza, anonimização (PII), curadoria, split train/eval
├── synthetic_generation.py # gera protocolos/FAQs/laudos sintéticos via Gemini
├── build_corpus.py         # combina MedQuAD/PubMedQA (stroke) + sintético -> train/eval.jsonl
├── synthetic_raw.jsonl     # saída bruta do gerador sintético
├── train.jsonl             # corpus curado de treino (510 exemplos)
└── eval.jsonl              # corpus curado de avaliação (89 exemplos)

src/security/
├── guardrails.py           # bloqueia prescrição direta, exige validação humana
└── audit_log.py            # log estruturado (JSONL) de cada interação

src/finetuning/
├── dataset.py              # formata exemplos para o formato de fine-tuning
└── evaluate.py             # checklist determinístico de avaliação do modelo

src/assistant/
├── patient_db.py           # CSV de stroke como "prontuário" em SQLite
├── retriever.py            # RAG (FAISS + embeddings locais) sobre o corpus
├── tools.py                # predição de risco de AVC + checagem de exames pendentes
├── prompts.py              # montagem do prompt (prontuário + fontes + pergunta)
├── llm_backend.py          # adapter LoRA local, com fallback para o Gemini
├── chain.py                # pipeline de pergunta-resposta (LangChain)
└── graph.py                # fluxo de decisão clínica (LangGraph)

results/finetuning/
├── lora_adapter/           # adapter treinado (versionado no repo, ~17MB)
└── metrics.json            # histórico de loss do treino
```

### Instalar dependências da Fase 3

Todas as dependências (LangChain, LangGraph, FAISS, `transformers`, `peft`,
`accelerate`, `datasets`) já estão em `requirements.txt`. Se o seu `venv` foi
criado na Fase 2, basta reinstalar:

```bash
pip install -r requirements.txt
```

### 1. Construir o corpus médico (protocolos + FAQs + dados públicos)

O corpus curado já está versionado em `data/medical_corpus/`. Só é necessário
rodar de novo para regenerá-lo do zero:

```bash
# (Opcional) gerar novos exemplos sintéticos via Gemini — usa GOOGLE_API_KEY do .env
python -m data.medical_corpus.synthetic_generation

# Busca MedQuAD/PubMedQA (filtrado por "stroke") + combina com o sintético,
# aplica curadoria/anonimização e grava train.jsonl / eval.jsonl
python -m data.medical_corpus.build_corpus
```

### 2. Testar os módulos de segurança

```bash
python -c "from src.security.guardrails import enforce_human_validation; print(enforce_human_validation('Tome 500mg de AAS agora.'))"
```

```bash
python -c "from src.security.audit_log import log_interaction, read_audit_log; log_interaction(9046, 'pergunta teste', 'resposta teste'); print(read_audit_log())"
```

### 3. Construir o "prontuário" (SQLite) e testar consultas

```bash
python -c "from src.assistant.patient_db import build_patient_db, get_patient_record; build_patient_db(); print(get_patient_record(9046))"
```

### 4. Testar a busca semântica (RAG/FAISS)

Baixa o modelo de embeddings (`all-MiniLM-L6-v2`, ~90MB) na primeira execução:

```bash
python -c "from src.assistant.retriever import build_vectorstore, retrieve; vs = build_vectorstore(); [print(r['source'], '-', r['text'][:80]) for r in retrieve(vs, 'criterios para trombolise no AVC', k=2)]"
```

### 5. Testar formatação/avaliação do dataset de fine-tuning

```bash
python -c "from src.finetuning.dataset import load_split, format_example; ex = load_split('data/medical_corpus/eval.jsonl')[0]; print(format_example(ex))"
```

### 6. Rodar o assistente (chain: RAG + prontuário + guardrails)

Requer o vectorstore (RAG) montado — passo 4 — e os modelos da Fase 1
treinados (`notebooks/01_baseline.ipynb`), já que `run_flow`/`answer_question`
consultam o prontuário SQLite e o modelo de risco de AVC.

`get_generate_fn()` usa o adapter fine-tuned de `results/finetuning/lora_adapter/`
(já versionado no repositório) — na primeira execução ele baixa os pesos do modelo
base `Qwen2.5-1.5B-Instruct` (~3 GB) do Hugging Face. Se a pasta do adapter for
removida, o backend cai automaticamente para o Gemini (`GOOGLE_API_KEY` no `.env`):

```bash
python -c "from src.assistant.chain import answer_question; from src.assistant.retriever import build_vectorstore; from src.assistant.llm_backend import get_generate_fn; vs = build_vectorstore(); gen = get_generate_fn(); result = answer_question(9046, 'Quais os proximos passos para este paciente?', vs, gen); print(result['response']); print(result['sources'])"
```

Ou via notebook: `notebooks/05_langchain_assistant.ipynb` (ver
[Executando os notebooks](#executando-os-notebooks--local-ou-google-colab)).

### 7. Rodar o fluxo completo de decisão clinica (LangGraph)

Executa `receive_patient_data -> ... -> run_stroke_prediction ->
suggest_conduct -> apply_guardrails -> [emit_alert] -> audit_log`
(`src/assistant/graph.py`), gravando o log de auditoria em
`results/audit_log.jsonl` via `src/security/audit_log.py`:

```bash
python -c "from src.assistant.graph import run_flow; from src.assistant.retriever import build_vectorstore; from src.assistant.llm_backend import get_generate_fn; vs = build_vectorstore(); gen = get_generate_fn(); state = run_flow(9046, 'Quais os proximos passos para este paciente?', vs, gen); print(state['response']); print('alerta:', state.get('alert'), state.get('alert_reason'))"
```

Ou via notebook: `notebooks/06_langgraph_flow.ipynb`.

### 8. Fine-tuning (LoRA/PEFT) — re-treinar no Google Colab

**O adapter já treinado está versionado no repositório**
(`results/finetuning/lora_adapter/`), então os passos 6 e 7 já usam o modelo
fine-tuned sem nenhum treino adicional. Rode este passo apenas para re-treinar.

O treino exige GPU — siga a [Opção B (Colab)](#opção-b--google-colab-via-drive-pessoal)
e selecione o ambiente T4 antes de executar `notebooks/04_finetuning.ipynb`.

Configuração usada na entrega (célula de treino do notebook):

| Parâmetro | Valor |
|-----------|-------|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` (não-gated) |
| Método | LoRA — `r=16`, `alpha=32`, `dropout=0.05`, alvos `q/k/v/o_proj` |
| Parâmetros treináveis | 4.358.144 de 1.548.072.448 (**0,28%**) |
| Épocas / batch | 3 épocas, batch 4 × grad. accumulation 4 |
| Learning rate | 2e-4, `fp16` |
| Tempo de treino | ~3,4 min em GPU T4 (96 steps) |

Ao final, o notebook salva o adapter em `results/finetuning/lora_adapter/` e as
métricas em `results/finetuning/metrics.json` — como o projeto está montado no
Drive, os arquivos já ficam na pasta do projeto. Baixe a pasta do Drive de volta
para o repositório local no mesmo caminho: `src/assistant/llm_backend.py` detecta
o adapter automaticamente (`local_adapter_available()`) e passa a usá-lo em vez
do fallback Gemini.

### Rodar os testes automatizados

```bash
pytest tests/ -q
```

São 111 testes cobrindo toda a lógica determinística: curadoria/anonimização do
corpus, guardrails, log de auditoria, consultas ao prontuário, formatação do
prompt e o roteamento condicional do grafo LangGraph (com LLM e índice FAISS
mockados — nenhum teste consome GPU ou cota de API).

---

## Integrantes

| Papel | Responsabilidade |
|-------|-----------------|
| Integrante 1 | Algoritmo Genético (TASK-1.1 a 1.6) |
| Integrante 2 | Integração LLM (TASK-2.1 a 2.8) |
| Integrante 3 | Integração, testes, docs e relatório final (TASK-3.1 a 3.5) |

---

## Dependências Principais

| Biblioteca | Uso | Fase |
|-----------|-----|------|
| scikit-learn | Modelos de ML | 1 |
| imbalanced-learn | SMOTE | 1 |
| pandas / numpy | Manipulação de dados | 1 |
| joblib | Persistência de modelos | 1 |
| deap | Algoritmo Genético | 2 |
| google-genai | LLM (Gemini) — Fase 2 e fallback do assistente | 2 |
| langchain / langchain-community | Pipeline do assistente e RAG | 3 |
| langgraph | Fluxo de decisão clínica com aresta condicional | 3 |
| faiss-cpu | Índice vetorial do RAG | 3 |
| sentence-transformers | Embeddings locais (`all-MiniLM-L6-v2`) | 3 |
| transformers / peft / accelerate | Fine-tuning LoRA e inferência do adapter | 3 |
| datasets | Formatação do corpus para o treino | 3 |
| pytest | Testes automatizados | 1–3 |
