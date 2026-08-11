# 🚀 Complete Generative AI Engineering Roadmap
### From Zero to Production-Ready AI Systems in 25 days

---

## 👤 About This Roadmap

This is not a tutorial collection.
This is a complete engineering curriculum that takes you from understanding
how tokens work all the way to deploying production-grade multi-agent AI systems
that handle real users, real failures, and real business requirements.

**Built by:** BBA Student + AI Engineering Self-Learner  
**Duration:** 25 Days  
**Projects:** 6 Production-Grade Systems  
**Background:** No CS degree. Business background. Built everything anyway.  
**Goal:** Become a Production AI Engineer

---

## 📋 Table of Contents

- [What You Will Build](#-what-you-will-build)
- [Why This Roadmap Exists](#-why-this-roadmap-exists)
- [Prerequisites & Setup](#-prerequisites--setup)
- [Week 1 — Foundations](#-week-1--foundations-days-1-5)
- [Week 2 — Tool Use & RAG](#-week-2--tool-use--rag-days-6-12)
- [Week 3 — Agents & Reliability](#-week-3--agents--reliability-days-13-19)
- [Week 4 — Production & Hardening](#-week-4--production--hardening-days-20-25)
- [Gap Coverage](#-gap-coverage--what-most-curricula-miss)
- [Complete Tech Stack](#-complete-tech-stack)
- [All 6 Projects](#-all-6-projects)
- [Free Learning Resources](#-free-learning-resources)
- [Internship Readiness](#-internship-readiness)
- [Progress Tracker](#-progress-tracker)
- [What Comes After](#-what-comes-after)

---

## 🏗️ What You Will Build

| # | Project | Core Tech | What It Does |
|---|---------|-----------|--------------|
| 1 | Invoice & Document Extractor | Pydantic, asyncio, OpenAI | Ingests messy documents, outputs 100% valid typed JSON |
| 2 | CLI Data Agent | Tool Calling, SQLite, httpx | Terminal agent that queries DBs and APIs dynamically |
| 3 | Hybrid RAG Engine | Qdrant, BM25, CrossEncoder | PDF search combining keyword + semantic + reranking |
| 4 | Autonomous ReAct Agent | State Machine, Redis, HITL | Multi-step agent with memory, self-correction, human approval |
| 5 | Automated Eval Suite | DeepEval, RAGAS, CI/CD | 50+ golden tests running on every git commit |
| 6 | Production Microservice | FastAPI, Docker, LiteLLM | Full streaming backend with caching, routing, tracing |

---

## 💡 Why This Roadmap Exists

Most AI tutorials teach you to call an API and call it a day.
This roadmap teaches you what happens when that API fails at 3am with 1000 users online.
---
What most people learn: What this roadmap teaches:
────────────────────── ──────────────────────────
Call OpenAI API → Build reliable async pipelines
Copy LangChain tutorial → Understand what LangChain hides
Make a demo chatbot → Deploy a production microservice
Hope it works → Write eval suites that prove it works
---

**The honest gaps this roadmap also covers that others skip:**
- Multi-agent orchestration (supervisor + specialist pattern)
- Workflow durability (survive crashes mid-task)
- PII detection and security layers
- Token bucket rate limiting and backpressure
- Cost budgets and per-user spending limits
- Comprehensive error taxonomy and recovery strategies

---
### System requirments

Python 3.11+
Docker Desktop
VS Code + Pylance extension
Git
8GB RAM minimum (16GB recommended)


### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/genai-roadmap
cd genai-roadmap

# Create virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install all dependencies
pip install openai google-generativeai pydantic fastapi uvicorn \
            qdrant-client chromadb sentence-transformers redis \
            langgraph ragas deepeval guardrails-ai litellm \
            httpx rank-bm25 tiktoken aiosqlite rich pypdf \
            pdfplumber python-dotenv pytest opentelemetry-api \
            opentelemetry-sdk temporalio cohere

# Setup environment variables
cp .env.example .env