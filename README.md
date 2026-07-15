## Multi-Agent Customer Intelligence Platform 

## Overview

This project implements a Multi-Agent Customer Intelligence Platform that processes customer queries using four specialized AI agents sharing a single LLM backend.

The platform is designed to support 100+ concurrent customer sessions while operating within an 8 GB GPU KV Cache budget by leveraging LMCache Prefix Caching and KV Cache Offloading.

## Agent Pipeline 
```
User Request
      │
      ▼
Intent Classifier
      │
      ▼
Knowledge Retriever
      │
      ▼
Response Generator
      │
      ▼
Quality Checker
      │
      ▼
Final Response 
```

## Features
```
Multi-Agent Architecture
Shared LLM Backend
LMCache Prefix Caching
Session Management
Quality Gate with Retry
Knowledge Retrieval using Vector Store
FastAPI REST API
Logging & Error Handling
Unit & Integration Testing
Load Testing for 100+ Sessions
```

## Project Structure
```
customer-intel-platform
│
├── agents/                  # AI Agents
│   ├── intent_classifier.py
│   ├── knowledge_retriever.py
│   ├── response_generator.py
│   ├── quality_checker.py
│   └── base.py
│
├── api/                     # REST APIs
│   ├── main.py
│   └── schemas.py
│
├── config/                  # Configuration
│   ├── settings.py
│   └── agent_prompts.py
│
├── core/                    # Pipeline & Cache
│   ├── pipeline.py
│   ├── cache_manager.py
│   ├── session_manager.py
│   ├── llm_backend.py
│   └── logging_setup.py
│
├── retrieval/               # Knowledge Base
│   ├── vector_store.py
│   ├── summarizer.py
│   └── sample_kb.json
│
├── scripts/                 # Utility Scripts
│   ├── run_server.py
│   ├── ingest_kb.py
│   ├── load_test.py
│   └── benchmark_cache.py
│
├── deploy/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

## High Level Architecture

<img width="1094" height="998" alt="image" src="https://github.com/user-attachments/assets/e19f9415-10f6-4724-a6c8-739e7513a9ee" />


LMCache Architecture

<img width="1094" height="902" alt="image" src="https://github.com/user-attachments/assets/e7aa3e92-81ad-436d-9ba5-20a47529de38" />


## Installations

1. Clone Repository
   
```bash
git clone https://github.com/TirumalaSrividya/multi-agent-customer-intelligence.git
cd multi-agent-customer-intelligence
```

2. Install Dependencies

``` 
pip install -r requirements.txt

``` 

## Run Project

Start FastAPI Server

``` 
python scripts/run_server.py 
``` 

Swagger UI Available at 
```
http://localhost:8080/docs
```


## Knowledge Base Ingestion

```
python scripts/ingest_kb.py
 ```


## Run Tests

Run all tests 
```
pytest
```

