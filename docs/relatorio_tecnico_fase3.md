# Relatório Técnico — Fase 3: Assistente Médico Virtual

**FIAP Pós-Tech IA para Devs — Tech Challenge Fase 3**
**Domínio:** AVC / Acidente Vascular Cerebral (continuidade das Fases 1–2)

> Fonte deste relatório: `docs/relatorio_tecnico_fase3.md`. Os relatórios
> das Fases 1–2 (`docs/relatorio_tecnico.docx`/`.pdf`) permanecem como
> entregável daquela fase; este documento cobre apenas os itens novos da
> Fase 3. Conversão para DOCX/PDF fica a critério do grupo antes da entrega
> final.

---

## 1. Contexto e Objetivo

Nas Fases 1 e 2, o projeto treinou e otimizou modelos de machine learning
para prever risco de AVC, e integrou um LLM (Gemini) para interpretar esses
resultados em linguagem natural. A Fase 3 avança para um assistente médico
virtual: um LLM afinado (fine-tuned) com dados médicos internos, orquestrado
via LangChain para consultar um "prontuário" estruturado e responder dúvidas
clínicas com contexto atualizado do paciente, e um fluxo de decisão
automatizado via LangGraph que integra a predição de risco já existente,
guardrails de segurança e logging de auditoria.

Optamos por manter o domínio em AVC/stroke (em vez de um hospital genérico
multi-especialidade) para reaproveitar o dataset e o modelo de predição já
validados nas fases anteriores, e para poder gerar dados sintéticos
coerentes com um tema já dominado pelo grupo.

## 2. Processo de Fine-tuning

### 2.1 Dados

Não há protocolos, laudos ou receitas reais do hospital disponíveis, então o
corpus de treino combina duas fontes (`data/medical_corpus/`):

1. **Dados públicos** — subconjunto de **MedQuAD** e **PubMedQA** (sugeridos
   pelo próprio desafio), filtrado por relevância ao tema stroke via busca
   full-text na API pública `datasets-server.huggingface.co` (não a
   biblioteca `datasets` completa, para evitar baixar os datasets inteiros
   — MedQuAD tem ~47 mil linhas, das quais ~650 são sobre stroke).
2. **Dados sintéticos** — protocolos internos, FAQs de médicos e modelos de
   laudo/receita gerados via Gemini (`data/medical_corpus/synthetic_generation.py`),
   claramente rotulados como material fictício de treinamento.

Ambas as fontes passam por curadoria/anonimização
(`data/medical_corpus/preprocessing.py`):

- **Scrub de PII** — regex para CPF, e-mail, telefone e datas.
- **Curadoria** — descarta exemplos com instrução vazia ou resposta muito
  curta (< 20 caracteres) ou muito longa (> 4000 caracteres).
- **Deduplicação** — remove exemplos exatamente duplicados.
- **Split determinístico** — 85% treino / 15% avaliação (seed fixa).

Resultado de uma execução de referência: 626 exemplos do MedQuAD + 33 do
PubMedQA + 50 sintéticos = 709 brutos → **599 curados** → **510 treino / 89
avaliação**.

### 2.2 Modelo Base e Método

- **Modelo base:** `Qwen/Qwen2.5-1.5B-Instruct` — um LLM pequeno e
  **não-gated** no Hugging Face, evitando a burocracia de acesso aos pesos
  oficiais do Llama. O desafio aceita "LLaMA, Falcon **ou outro**", e um
  modelo de 1.5B parâmetros é treinável em poucas horas numa GPU T4
  gratuita do Google Colab.
- **Método:** LoRA (Low-Rank Adaptation) via `peft`, em vez de fine-tuning
  completo — muito mais leve computacionalmente e o adapter resultante
  (poucos MB) pode ser versionado no repositório.
- **Ambiente de treino:** Google Colab (GPU gratuita). O notebook
  `notebooks/04_finetuning.ipynb` foi desenhado para rodar lá — treinar um
  LLM mesmo pequeno em CPU local é impraticável em tempo hábil.

### 2.3 Pipeline

```
data/medical_corpus/{train,eval}.jsonl
  -> src/finetuning/dataset.py (build_hf_dataset)
  -> transformers.AutoModelForCausalLM + peft.LoraConfig
  -> transformers.Trainer.train()
  -> adapter salvo em results/finetuning/lora_adapter/
  -> métricas em results/finetuning/metrics.json
```

### 2.4 Status desta entrega

O pipeline de fine-tuning está implementado e testado quanto à formatação de
dados (`src/finetuning/dataset.py`, `src/finetuning/evaluate.py`, cobertos
por testes automatizados). A execução real do treino no Colab é a próxima
etapa antes da entrega final — o adapter treinado deve ser baixado para
`results/finetuning/lora_adapter/`, onde `src/assistant/llm_backend.py` o
detecta automaticamente e passa a usá-lo no lugar do fallback Gemini.

## 3. Descrição do Assistente Médico

O assistente (`src/assistant/`) é um **pipeline determinístico** (não um
agente LLM com tool-calling livre): cada etapa roda em ordem fixa, o que o
torna confiável e testável — não há risco do modelo "decidir" pular a
consulta ao prontuário ou o guardrail de segurança.

| Componente | Arquivo | Papel |
|---|---|---|
| Prontuário | `patient_db.py` | Carrega `data/healthcare-dataset-stroke-data.csv` (já usado nas Fases 1–2) como tabela SQLite `prontuarios` |
| RAG | `retriever.py` | Índice FAISS sobre o corpus médico, embeddings locais (`sentence-transformers/all-MiniLM-L6-v2`) — 100% offline |
| Predição de risco | `tools.py` | Reaproveita o modelo treinado em `src/models.py` para estimar risco de AVC do paciente |
| Exames pendentes | `tools.py` | Checagem determinística (mock) de campos ausentes no registro (ex.: IMC nulo) |
| Prompt | `prompts.py` | Monta o prompt final: prontuário + fontes recuperadas + exames pendentes + pergunta |
| LLM backend | `llm_backend.py` | Usa o adapter LoRA local se disponível; senão cai para o Gemini (`src/llm/client.py`, Fase 2) |
| Guardrail | `src/security/guardrails.py` | Bloqueia/anota respostas que soam como prescrição direta |
| Orquestração | `chain.py` | Pipeline de pergunta-resposta pontual |
| Fluxo completo | `graph.py` | Mesmo pipeline + predição de risco + alerta + auditoria, como grafo LangGraph |

Cada resposta do assistente vem acompanhada das **fontes** (`sources`) que a
embasaram — trecho do protocolo/FAQ e sua origem (MedQuAD, PubMedQA ou
sintético) — atendendo ao requisito de explainability do desafio.

## 4. Segurança e Validação

- **Nunca prescrever diretamente:** `src/security/guardrails.py` detecta
  padrões de prescrição direta (verbos imperativos como "tome"/"administre",
  dosagens como "500mg") via regex determinístico. Quando detectado, a
  resposta recebe um disclaimer obrigatório e a flag
  `requires_human_validation=True`.
- **Logging de auditoria:** `src/security/audit_log.py` grava, em
  `results/audit_log.jsonl`, cada interação — timestamp, `patient_id`
  pseudonimizado (SHA-256), pergunta, resposta, fontes citadas e flags de
  guardrail — usando o módulo `logging` padrão do Python.
- **Explainability:** ver seção 3 — toda resposta cita a fonte usada.

## 5. Fluxo LangChain / LangGraph

```mermaid
flowchart TD
    A([receive_patient_data]) --> B[query_patient_record]
    B --> C[check_pending_exams]
    C --> D["run_stroke_prediction\n(reaproveita src/models.py)"]
    D --> E["suggest_conduct\n(LLM + RAG)"]
    E --> F["apply_guardrails\n(src/security/guardrails.py)"]
    F --> G{"risco alto OU\nguardrail acionado?"}
    G -->|sim| H[emit_alert]
    G -->|não| I["audit_log\n(src/security/audit_log.py)"]
    H --> I
    I --> J([Fim])
```

A aresta condicional (`add_conditional_edges` do LangGraph) só visita
`emit_alert` quando a probabilidade de AVC prevista está acima do limiar
(`HIGH_RISK_THRESHOLD = 0.5`) ou quando o guardrail de prescrição direta foi
acionado — nos dois casos, o log de auditoria roda ao final, garantindo
rastreabilidade completa independentemente do resultado. Ver
`docs/architecture.md` para os diagramas completos (fine-tuning e pipeline
do assistente) e a tabela de decisões técnicas.

## 6. Avaliação do Modelo e Análise dos Resultados

### 6.1 Avaliação determinística (sem GPU/API)

`src/finetuning/evaluate.py` implementa um checklist reprodutível (mesmo
espírito do checklist da Fase 2 em `src/llm/evaluation.py`, evitando
"LLM-as-judge"):

| Regra | Verificação |
|---|---|
| `mentions_key_terms` | ao menos um termo significativo (≥5 letras) da resposta esperada aparece na resposta gerada |
| `avoids_direct_prescription` | reaproveita `src.security.guardrails.contains_direct_prescription` |
| `reasonable_length` | 20–4000 caracteres |

Score final: média das 3 regras, em `[0, 1]`. Esse checklist será aplicado
sobre `data/medical_corpus/eval.jsonl` após o treino real no Colab (célula
final de `notebooks/04_finetuning.ipynb`), comparando a saída do modelo
antes e depois do LoRA.

### 6.2 Testes automatizados

Toda a lógica determinística nova está coberta por testes (`pytest tests/
-q`): curadoria/anonimização do corpus, guardrails, audit log, consultas ao
prontuário, formatação do prompt do assistente e o roteamento condicional
do grafo LangGraph (cenários de alto risco, baixo risco e guardrail
acionado, com o LLM e o índice FAISS mockados). Resultado nesta entrega:
**111 testes passando** (54 da Fase 2 + 57 novos da Fase 3).

### 6.3 Validação manual end-to-end

Antes de escrever os testes, cada módulo foi exercitado manualmente:
construção do corpus (599 exemplos curados), guardrail com texto seguro e
inseguro, log de auditoria, prontuário SQLite, busca semântica FAISS
(retornando corretamente os protocolos sintéticos mais relevantes para uma
pergunta sobre trombólise) e o fluxo LangGraph completo para três cenários
— paciente de alto risco (disparou alerta corretamente, probabilidade
0.77), paciente de baixo risco com exame pendente (não disparou alerta) e
uma resposta com prescrição direta simulada (disparou alerta mesmo com
risco baixo, confirmando que o guardrail sozinho já aciona o alerta).

## 7. Limitações Conhecidas

- Dados de treino são sintéticos/públicos, não protocolos reais do hospital — ver `docs/architecture.md` para a lista completa de limitações técnicas (guardrail por regex, cota da API Gemini, RAG sem reranking, fine-tuning não executável localmente).
- O treino real do adapter LoRA ainda precisa ser executado no Colab antes da entrega final (pipeline pronto e testado, mas não executado nesta versão do relatório).
- Os notebooks 05 e 06 (demonstração do assistente e do fluxo completo) têm o código pronto e validado manualmente, mas não foram executados com saída real gravada nesta versão por ter esgotado a cota diária gratuita da API do Gemini durante o desenvolvimento — podem ser reexecutados quando a cota renovar ou assim que o adapter local estiver disponível (elimina a dependência da API).

## 8. Próximos Passos

1. Rodar `notebooks/04_finetuning.ipynb` no Colab e baixar o adapter para `results/finetuning/lora_adapter/`.
2. Reexecutar `notebooks/05_langchain_assistant.ipynb` e `notebooks/06_langgraph_flow.ipynb` (com o adapter local ou após renovação de cota) e gravar as saídas reais.
3. Gravar o vídeo de demonstração (até 15 min) cobrindo: fine-tuning, execução do fluxo automatizado, resposta a perguntas clínicas contextualizadas, e logs/validação das respostas.
4. Converter este relatório (junto com `docs/architecture.md`) para DOCX/PDF, como feito nas Fases 1–2.
