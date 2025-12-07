## AI App Name
Onasi AI-Based Clinical Decision Support System (CDSS)

## Summary
Evidence-based CDSS providing drug safety checks, clinical management pathways, and basic radiology reporting. The system combines Retrieval-Augmented Generation (RAG) with large language models and a lightweight FastAPI service. Content is fact-based and sourced from reputable medical references.

## Core features
- Drug interaction and safety assessment (drug–drug, drug–allergy, drug–disease; pregnancy/lactation warnings; severity)
- Clinical pathways generator (low/medium/high risk options with monitoring and trade-offs)
- X‑ray report assistant (image type triage; structured report: View, Findings, Impression, Recommendations)
- Auditable outputs with references where available

## Architectural & Layers
- Knowledge & Retrieval: PDF ingestion (PyPDF) → chunking → FAISS vector store; OpenAI embeddings for semantic search
- Reasoning (Text): OpenRouter-hosted LLM for query answering and pathway generation
- Vision (Images): GPT‑4o via OpenRouter for multimodal chest/skeletal X‑rays
- API & Orchestration: FastAPI endpoints (`/clinical-assess`, `/clinical-pathways`, `/generate_report`) with prompt templates and contract enforcement

## Technology/Tool Stack
- Backend: Python, FastAPI, Uvicorn, Pydantic
- RAG: LangChain, FAISS, `text-embedding-ada-002` (OpenAI)
- LLM (text): `meta-llama/llama-3.3-70b-instruct:free` via OpenRouter
- LLM (vision): `openai/gpt-4o` via OpenRouter
- Knowledge assets (examples): Drugs.com, UpToDate, FDA drug labels
- Deployment: Render Web Services (Starter)

## API and hosting costs (indicative)
- OpenAI embeddings (`text-embedding-ada-002`): approximately $0.0001 per 1K tokens — see OpenAI API pricing
- GPT‑4o (used via OpenRouter for imaging): approximately $2.50 per million input tokens and $10.00 per million output tokens — see OpenAI API pricing
- OpenRouter `meta-llama/llama‑3.3‑70b‑instruct:free`: $0/token on the free route (queue/throughput limits; paid routes vary by provider)
- Render Web Service Starter plan: ~$7/month — see Render pricing

Notes
- Fact-based only; no regional localization. 
- Knowledge assets include public medical websites (e.g., Drugs.com, UpToDate) and FDA labels.

## Client requirements (indicative)
- Server OS: Linux (Ubuntu 20.04+/Debian) or Windows Server 2019+
- Resources (per instance): 2 vCPU, 2–4 GB RAM, 10–20 GB SSD (FAISS index + logs)
- Runtime: Python 3.10+, FastAPI via Uvicorn; deploy as Docker container or system service
- Database: Not required by default (FAISS stored on disk). Optional: PostgreSQL 14+ or SQL Server 2019+ for auditing/analytics if you choose to persist results
- Networking: Public HTTPS endpoint with reverse proxy (e.g., Nginx) or managed platform (Render)
- Secrets: Environment variables or secret manager for API keys (OpenRouter, OpenAI)

## Security
- Transport: Enforce HTTPS/TLS for all endpoints; terminate TLS at load balancer or reverse proxy
- Secrets handling: Store API keys only in env/secret manager; never in code or logs
- Data handling: By default, no patient data persistence; if enabling storage, use encryption at rest, access controls, and retention policies
- Access control: Optionally protect endpoints with API key or token; restrict ingress via firewall/IP allowlist when feasible
- Observability: Centralized logging (exclude PHI/PII), rate limiting, and basic request auditing
- Supply chain: Pin dependencies, patch regularly; restrict outbound egress to required AI endpoints