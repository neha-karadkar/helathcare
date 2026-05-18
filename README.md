# Healthcare Document Summarization Agent

A professional agent for summarizing healthcare documents using Azure AI Search and Azure OpenAI GPT-4.1. Retrieves relevant content from Azure AI Search, applies compliance guardrails, and generates concise, accurate summaries suitable for healthcare professionals.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

**Direct execution:**
```
python code/agent.py
```

**As a FastAPI server:**
```
uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME` — Agent name (pre-configured)
- `AGENT_ID` — Agent unique identifier (pre-configured)
- `PROJECT_NAME` — Project name (pre-configured)
- `PROJECT_ID` — Project unique identifier (pre-configured)

**General Configuration**
- `ENVIRONMENT` — Deployment environment (e.g., development, production)
- `SERVICE_NAME` — Service name for observability/logging
- `SERVICE_VERSION` — Service version

**Azure Key Vault**
- `USE_KEY_VAULT` — Enable Azure Key Vault integration (`true`/`false`)
- `KEY_VAULT_URI` — Azure Key Vault URI
- `AZURE_USE_DEFAULT_CREDENTIAL` — Use DefaultAzureCredential (`true`/`false`)

**Azure Authentication (for Key Vault, if not using managed identity)**
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER` — LLM provider (`openai`, `azure`, etc.)
- `LLM_MODEL` — LLM model name (e.g., `gpt-4.1`)
- `LLM_TEMPERATURE` — Model temperature (float)
- `LLM_MAX_TOKENS` — Max tokens in model response

**API Keys / Secrets**
- `OPENAI_API_KEY` — OpenAI API key (if using OpenAI)
- `AZURE_OPENAI_API_KEY` — Azure OpenAI API key (if using Azure)
- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI endpoint URL
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` — Azure OpenAI embedding deployment name
- `ANTHROPIC_API_KEY` — Anthropic API key (if using Anthropic)
- `GOOGLE_API_KEY` — Google API key (if using Google)
- `AZURE_CONTENT_SAFETY_KEY` — Azure Content Safety API key
- `OBS_AZURE_SQL_USERNAME` — Azure SQL username for observability
- `OBS_AZURE_SQL_PASSWORD` — Azure SQL password for observability

**Service Endpoints**
- `AZURE_CONTENT_SAFETY_ENDPOINT` — Azure Content Safety endpoint URL
- `AZURE_SEARCH_ENDPOINT` — Azure AI Search endpoint URL
- `AZURE_SEARCH_API_KEY` — Azure AI Search API key
- `AZURE_SEARCH_INDEX_NAME` — Azure AI Search index name

**Observability (Azure SQL)**
- `OBS_DATABASE_TYPE` — Observability DB type (e.g., `azure_sql`)
- `OBS_AZURE_SQL_SERVER` — Azure SQL server
- `OBS_AZURE_SQL_DATABASE` — Azure SQL database name
- `OBS_AZURE_SQL_PORT` — Azure SQL port
- `OBS_AZURE_SQL_SCHEMA` — Azure SQL schema
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE` — Trust SQL Server certificate (`yes` recommended)

**Agent-Specific**
- `VALIDATION_CONFIG_PATH` — Path to validation config file (optional)
- `LLM_MODELS` — JSON list of LLM model configs for token pricing (optional)
- `CONTENT_SAFETY_ENABLED` — Enable Azure Content Safety runtime checks (`true`/`false`)
- `CONTENT_SAFETY_SEVERITY_THRESHOLD` — Content Safety severity threshold (integer)

---

## API Endpoints

### **GET** `/health`
- **Description:** Health check endpoint.
- **Response:**
  ```
  {
    "status": "ok"
  }
  ```

### **POST** `/query`
- **Description:** Generate a summary of the uploaded healthcare document. No request body required; agent uses internal configuration.
- **Response:**
  ```
  {
    "success": true|false,
    "summary": "string|null",
    "error": null|string,
    "tool_calls_made": []
  }
  ```

### **Global Exception Handler**
- **Response:**
  ```
  {
    "success": false,
    "error": "Internal server error: ...",
    "tips": "Check your request format and try again. If the error persists, contact support."
  }
  ```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t healthcare-document-summarization-agent -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name healthcare-document-summarization-agent healthcare-document-summarization-agent
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs healthcare-document-summarization-agent
```

### 7. Stop the container:
```
docker stop healthcare-document-summarization-agent
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Healthcare Document Summarization Agent** — Accurate, compliant, and concise healthcare document summarization with Azure AI Search and GPT-4.1.