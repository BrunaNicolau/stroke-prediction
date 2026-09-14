# Stroke Prediction — Phase 2: Genetic Algorithm + LLM

**FIAP Pós-Tech IA para Devs — Tech Challenge Fase 2**

Otimização de modelos de previsão de AVC usando Algoritmos Genéticos, com integração de LLM para interpretação dos diagnósticos em linguagem natural.

## Contexto

Na Fase 1, treinamos dois modelos para prever risco de AVC:

| Modelo | Accuracy | Recall | F1 |
|--------|----------|--------|----|
| Regressão Logística | 83.37% | **44.00%** | 0.21 |
| Random Forest | 92.86% | 24.00% | 0.25 |

O **Recall** é a métrica principal (contexto médico: minimizar casos não detectados).

Nesta fase, aplicamos Algoritmos Genéticos para otimizar os hiperparâmetros e integramos um LLM para interpretar os resultados.

---

## Estrutura do Repositório

```
stroke-prediction-phase2/
├── data/
│   └── download_data.py            # Script para baixar o dataset do Kaggle
├── src/
│   ├── preprocessing.py            # Pipeline de pré-processamento (Fase 1 refatorada)
│   ├── models.py                   # Treinamento, avaliação, save/load, predict
│   ├── genetic_algorithm/          # Motor do AG
│   └── llm/                        # Integração LLM
├── notebooks/
│   ├── 01_baseline.ipynb           # Reprodução da Fase 1 (linha de base)
│   ├── 02_genetic_algorithm.ipynb  # Experimentos com AG
│   └── 03_llm_integration.ipynb    # Demonstração da integração LLM
├── results/
│   ├── logistic_regression.joblib  # Modelo LR treinado
│   ├── random_forest.joblib        # Modelo RF treinado
│   └── baseline_metrics.json       # Métricas da linha de base
├── tests/
├── docs/
├── .env.example
└── requirements.txt
```

---

## Pré-requisitos

- Python 3.10+
- Conta no Kaggle (para baixar o dataset)
- API key do Google Gemini

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repo>
cd stroke-prediction-phase2

# 2. Crie e ative um ambiente virtual
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
copy .env.example .env
# Edite .env e adicione sua GOOGLE_API_KEY
```

---

## Dataset

O dataset utilizado é o [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) do Kaggle.

```bash
python data/download_data.py
```

Isso salva o CSV em `data/healthcare-dataset-stroke-data.csv`.

> O CSV não está no repositório (arquivo grande). Rode o script acima após clonar.

---

## Como Executar

### 1. Reproduzir a linha de base (Fase 1)

```bash
jupyter notebook notebooks/01_baseline.ipynb
```

Isso treina os dois modelos baseline e salva em `results/`.

### 2. Experimentos com Algoritmo Genético

```bash
jupyter notebook notebooks/02_genetic_algorithm.ipynb
```

### 3. Demonstração LLM

```bash
jupyter notebook notebooks/03_llm_integration.ipynb
```

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

---

## Fase 3 — Assistente Médico (Fine-tuning + LangChain + LangGraph)

**FIAP Pós-Tech IA para Devs — Tech Challenge Fase 3**

Sobre a base da Fase 2, a Fase 3 adiciona: fine-tuning (LoRA/PEFT) de um LLM
com dados médicos internos (protocolos, FAQs, laudos), um assistente
LangChain com RAG sobre esses dados + consulta a um "prontuário" estruturado,
guardrails de segurança e logging de auditoria, e um fluxo de decisão
automatizado em LangGraph. Domínio: AVC/stroke, para manter coerência com o
dataset e os modelos já usados nas Fases 1–2.

### Estrutura adicionada

```
data/medical_corpus/
├── preprocessing.py        # limpeza, anonimização (PII), curadoria, split train/eval
├── synthetic_generation.py # gera protocolos/FAQs/laudos sintéticos via Gemini
├── build_corpus.py         # combina MedQuAD/PubMedQA (stroke) + sintético -> train/eval.jsonl
├── synthetic_raw.jsonl     # saída bruta do gerador sintético
├── train.jsonl             # corpus curado de treino
└── eval.jsonl              # corpus curado de avaliação

src/security/
├── guardrails.py           # bloqueia prescrição direta, exige validação humana
└── audit_log.py            # log estruturado (JSONL) de cada interação

src/finetuning/
├── dataset.py               # formata exemplos para o formato de fine-tuning
└── evaluate.py               # checklist determinístico de avaliação do modelo

src/assistant/
├── patient_db.py            # CSV de stroke como "prontuário" em SQLite
└── retriever.py              # RAG (FAISS + embeddings locais) sobre o corpus
```

### Instalar dependências da Fase 3

Já estão em `requirements.txt`; se seu `venv` já existia da Fase 2, instale o
que falta:

```powershell
pip install langchain langchain-community langgraph faiss-cpu sentence-transformers "datasets<3.0"
```

### 1. Construir o corpus médico (protocolos + FAQs + dados públicos)

```powershell
# (Opcional) gerar novos exemplos sintéticos via Gemini — usa GOOGLE_API_KEY do .env
python -m data.medical_corpus.synthetic_generation

# Busca MedQuAD/PubMedQA (filtrado por "stroke") + combina com o sintético,
# aplica curadoria/anonimização e grava train.jsonl / eval.jsonl
python -m data.medical_corpus.build_corpus
```

### 2. Testar os módulos de segurança

```powershell
python -c "from src.security.guardrails import enforce_human_validation; print(enforce_human_validation('Tome 500mg de AAS agora.'))"
python -c "from src.security.audit_log import log_interaction, read_audit_log; log_interaction(9046, 'pergunta teste', 'resposta teste'); print(read_audit_log())"
```

### 3. Construir o "prontuário" (SQLite) e testar consultas

```powershell
python -c "from src.assistant.patient_db import build_patient_db, get_patient_record; build_patient_db(); print(get_patient_record(9046))"
```

### 4. Testar a busca semântica (RAG/FAISS)

Baixa o modelo de embeddings (`all-MiniLM-L6-v2`, ~90MB) na primeira execução:

```powershell
python -c "from src.assistant.retriever import build_vectorstore, retrieve; vs = build_vectorstore(); [print(r['source'], '-', r['text'][:80]) for r in retrieve(vs, 'criterios para trombolise no AVC', k=2)]"
```

### 5. Testar formatação/avaliação do dataset de fine-tuning

```powershell
python -c "from src.finetuning.dataset import load_split, format_example; ex = load_split('data/medical_corpus/eval.jsonl')[0]; print(format_example(ex))"
```

### 6. Rodar o assistente (chain: RAG + prontuário + guardrails)

Requer o vectorstore (RAG) montado — passo 4 — e os modelos da Fase 1
treinados (`notebooks/01_baseline.ipynb`), já que `run_flow`/`answer_question`
consultam o prontuário SQLite e o modelo de risco de AVC. Sem
`results/finetuning/lora_adapter/` localmente, `get_generate_fn` cai
automaticamente para o Gemini (`GOOGLE_API_KEY` no `.env`):

```powershell
python -c "from src.assistant.chain import answer_question; from src.assistant.retriever import build_vectorstore; from src.assistant.llm_backend import get_generate_fn; vs = build_vectorstore(); gen = get_generate_fn(); result = answer_question(9046, 'Quais os proximos passos para este paciente?', vs, gen); print(result['response']); print(result['sources'])"
```

Ou via notebook:

```powershell
jupyter notebook notebooks/05_langchain_assistant.ipynb
```

### 7. Rodar o fluxo completo de decisão clinica (LangGraph)

Executa `receive_patient_data -> ... -> run_stroke_prediction ->
suggest_conduct -> apply_guardrails -> [emit_alert] -> audit_log`
(`src/assistant/graph.py`), gravando o log de auditoria em
`src/security/audit_log.py`:

```powershell
python -c "from src.assistant.graph import run_flow; from src.assistant.retriever import build_vectorstore; from src.assistant.llm_backend import get_generate_fn; vs = build_vectorstore(); gen = get_generate_fn(); state = run_flow(9046, 'Quais os proximos passos para este paciente?', vs, gen); print(state['response']); print('alerta:', state.get('alert'), state.get('alert_reason'))"
```

Ou via notebook:

```powershell
jupyter notebook notebooks/06_langgraph_flow.ipynb
```

### 8. Fine-tuning (LoRA/PEFT) — roda no Google Colab

Pesado para CPU local (`transformers`, `peft`, `accelerate`, `datasets` são
instalados apenas lá — ver comentário em `requirements.txt`):

```powershell
jupyter notebook notebooks/04_finetuning.ipynb
```

Depois de treinado, baixe o adapter do Colab para
`results/finetuning/lora_adapter/` — `src/assistant/llm_backend.py` passa a
usá-lo automaticamente em vez do fallback Gemini.

### Rodar os testes automatizados

```powershell
pytest tests/ -q
```

---

## Integrantes

| Papel | Responsabilidade |
|-------|-----------------|
| Integrante 1 | Algoritmo Genético (TASK-1.1 a 1.6) |
| Integrante 2 | Integração LLM (TASK-2.1 a 2.8) |
| Integrante 3 | Integração, testes, docs e relatório final (TASK-3.1 a 3.5) |

---

## Dependências Principais

| Biblioteca | Uso |
|-----------|-----|
| scikit-learn | Modelos de ML |
| imbalanced-learn | SMOTE |
| deap | Algoritmo Genético |
| google-genai | LLM (Gemini) |
| pandas / numpy | Manipulação de dados |
| joblib | Persistência de modelos |
| pytest | Testes automatizados |
