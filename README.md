## Multi-Agent Customer Intelligence Platform 

## Overview

This project implements a Multi-Agent Customer Intelligence Platform that processes customer queries using four specialized AI agents sharing a single LLM backend.

The platform is designed to support 100+ concurrent customer sessions while operating within an 8 GB GPU KV Cache budget by leveraging LMCache Prefix Caching and KV Cache Offloading.

## Agent Pipeline 

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


## Features

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


## Project Structure

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


## High Level Architecture

                    +----------------------+
                    |     User Request     |
                    +----------+-----------+
                               |
                               ▼
                 +---------------------------+
                 | Intent Classifier Agent   |
                 +------------+--------------+
                              |
                              ▼
                 +---------------------------+
                 | Knowledge Retriever Agent |
                 +------------+--------------+
                              |
                              ▼
                 +---------------------------+
                 | Response Generator Agent  |
                 +------------+--------------+
                              |
                              ▼
                 +---------------------------+
                 | Quality Checker Agent     |
                 +------------+--------------+
                              |
               Quality Score >= Threshold?
                     /                 \
                  Yes                   No
                   |                     |
                   ▼                     ▼
          Return Response              Retry


LMCache Architecture

             Shared LLM Backend
                     │
     ┌───────────────┼───────────────┐
     │               │               │
 Intent Agent   Retriever      Generator
     │               │               │
     └───────────────┼───────────────┘
                     │
             Shared KV Cache
          (LMCache Prefix Cache)
                     │
             CPU / Disk Offload
                     │
               100+ Sessions


## Installations

1. Clone Repository

git clone <repo_url>
cd Agentic_Customer_Pipeline

2. Install Dependencies

``` 
pip install -r requirements.txt

``` 

## Run Project

Start FastAPI Server

``` 
python scripts/run_server.py 
``` 

Swagger UI Available at http://localhost:8080/docs


Example Request

{
  "session_id": "user1",
  "message": "How do I reset my password?"
}


## Knowledge Base Ingestion

```
python scripts/ingest_kb.py
 ```


## Run Tests

Run all tests 
```
pytest
```

