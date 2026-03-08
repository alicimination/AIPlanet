# Reliable Multimodal Math Mentor

Production-style Streamlit application for **JEE-style math solving** with:
- Multimodal input (**Text / Image / Audio**)
- Multi-agent reasoning pipeline
- Retrieval-Augmented Generation (RAG)
- Human-in-the-loop controls
- Persistent memory + similarity reuse
- **Free local model stack** (no paid API requirement)

## 100% Local/Free Runtime Design

This project is configured so all model artifacts are stored in the repo folder:
- `./models/huggingface`
- `./models/sentence_transformers`
- `./models/whisper`
- `./models/paddle`

`utils/local_paths.py` sets environment variables (`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `TRANSFORMERS_CACHE`, `PADDLE_HOME`) so downloads avoid global user paths.
It also sets `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip startup connectivity checks to model hosters in restricted environments.

## Architecture

```text
Input (Text/Image/Audio)
  -> OCR/ASR extraction + user confirmation
  -> Parser Agent (structured JSON)
  -> Intent Router Agent (topic/strategy)
  -> Solver Agent (RAG + SymPy tool)
  -> Verifier Agent (checks + uncertainty)
  -> Explainer Agent (student-friendly steps)
  -> HITL gate + feedback
  -> SQLite memory + similarity reuse
```

## Project Structure

```text
math-mentor/
├── app.py
├── agents/
├── rag/
├── multimodal/
├── memory/
├── tools/
├── hitl/
├── knowledge_base/
├── utils/
│   ├── local_paths.py
│   ├── prompts.py
│   └── logging.py
├── scripts/
│   └── setup_local.sh
├── requirements.txt
└── README.md
```

## Key Features

### 1) Multimodal Input
- **Text**: direct problem input.
- **Image**: OCR via PaddleOCR (Tesseract fallback), editable extraction box.
- **Audio**: Whisper transcription + math phrase normalization + user confirmation.

### 2) Parser Agent Output
Produces structured JSON with topic, variables, constraints, ambiguity flag.

### 3) RAG
- KB markdown docs → chunking → sentence-transformer embeddings → FAISS storage.
- Retrieval top-k = 4.
- UI shows retrieved sources and chunk content.
- If no retrieval, app explicitly states no source found (no fabricated citations).

### 4) Multi-agent Pipeline
1. Parser Agent
2. Intent Router Agent
3. Solver Agent
4. Verifier Agent
5. Explainer Agent

### 5) HITL Triggers
- Low OCR confidence
- Low ASR confidence
- Parser ambiguity
- Verifier uncertainty
- User can recheck via incorrect feedback path

### 6) Memory + Self-learning
SQLite stores:
- original_input
- parsed_problem
- retrieved_context
- solution
- verification_result
- user_feedback
- timestamp

Similarity search retrieves similar solved problems to reuse patterns.
OCR corrections are also stored.

## Setup (venv-only, no global installs)

```bash
bash scripts/setup_local.sh
source .venv/bin/activate
```

> `sqlite3` is Python stdlib; no separate install required.

### Optional OS packages
- Tesseract binary (if using pytesseract fallback)
- FFmpeg (recommended for Whisper audio handling)

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

## Deployment

Compatible with:
- Streamlit Cloud
- HuggingFace Spaces (Streamlit SDK)

## Notes
- First run may download local models into `./models`.
- Solver has symbolic automation and fallback explanations; verifier + HITL protects reliability.
