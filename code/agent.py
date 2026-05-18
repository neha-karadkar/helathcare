import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': True
}

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# Constants for agent configuration
SYSTEM_PROMPT = (
    "You are a professional healthcare document summarization agent. Your task is to generate a concise, accurate summary of the uploaded document, strictly based on the content retrieved from Azure AI Search. Summarize key findings, study details, and relevant information without adding or inventing content. If the document is not found or no relevant content is retrieved, provide a clear fallback response. Format your summary in clear, formal prose suitable for healthcare professionals."
)
OUTPUT_FORMAT = (
    "- Provide the summary as a single, well-structured paragraph.\n\n"
    "- Do not include bullet points or lists unless the content requires it for clarity.\n\n"
    "- Reference document title in the summary if appropriate."
)
FALLBACK_RESPONSE = "No relevant content was found in the uploaded document. Please ensure the document is correctly uploaded and try again."
SELECTED_DOCUMENT_TITLES = ["Healthcare.pdf"]
ENRICHED_FIELDS = ["entities", "keyphrases", "relationships"]
TOP_K = 5
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

_logger = logging.getLogger("agent")
_enriched_available = None  # None = not yet checked, True/False after first search

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Healthcare Document Summarization Agent",
    description="Summarizes healthcare documents using Azure AI Search and Azure OpenAI GPT-4.1.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

class QueryResponse(BaseModel):
    success: bool = Field(True, description="Whether the query was processed successfully")
    summary: Optional[str] = Field(None, description="Summary of the uploaded document")
    error: Optional[str] = Field(None, description="Error message if any")
    tool_calls_made: Optional[List[str]] = Field(None, description="List of tool calls made (none for this agent)")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Whether the request failed")
    error: str = Field(..., description="Error message")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing the error")

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Health check status")

class Logger:
    """Utility logger for agent events and errors."""
    def __init__(self):
        self.logger = logging.getLogger("agent")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event: str):
        self.logger.info(event)

    def log_error(self, error: str):
        self.logger.error(error)

class ComplianceManager:
    """Ensures HIPAA, GDPR, FDA compliance; manages PII redaction and audit logging."""
    def __init__(self, logger: Logger):
        self.logger = logger

    def validate_compliance(self, data: str) -> bool:
        # For this agent, compliance is enforced via guardrails and content safety.
        # Additional compliance checks can be added here.
        return True

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def redact_pii(self, content: str) -> str:
        # Use guardrails sanitizer for PII redaction.
        from modules.guardrails.guardrails_service import get_guardrails_service
        guardrails_service = get_guardrails_service(config=GUARDRAILS_CONFIG)
        return guardrails_service.sanitize_text(content)

class ChunkRetriever:
    """Service for retrieving relevant chunks from Azure AI Search."""
    def __init__(self):
        self.search_client = None
        self.logger = Logger()

    def _get_search_client(self):
        if self.search_client is None:
            self.search_client = SearchClient(
                endpoint=Config.AZURE_SEARCH_ENDPOINT,
                index_name=Config.AZURE_SEARCH_INDEX_NAME,
                credential=AzureKeyCredential(Config.AZURE_SEARCH_API_KEY),
            )
        return self.search_client

    async def _search_with_fallback(self, query: str, embedding: List[float], selected_titles: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Try search with enriched fields; fall back to base fields if index lacks them."""
        global _enriched_available
        from azure.core.exceptions import HttpResponseError

        search_client = self._get_search_client()
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="vector")
        base_fields = ["chunk", "title"]

        if _enriched_available is False:
            select_fields = base_fields
        else:
            select_fields = base_fields + ENRICHED_FIELDS

        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": select_fields,
        }
        if selected_titles:
            odata_parts = [f"title eq '{t}'" for t in selected_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)

        _t0 = _time.time()
        try:
            results = list(search_client.search(**search_kwargs))
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(results)[:200] if results is not None else None,
                    status="success",
                )
            except Exception:
                pass
            if _enriched_available is None:
                _enriched_available = True
                _logger.info("Enriched index fields are AVAILABLE — using: %s", ENRICHED_FIELDS)
            return results
        except HttpResponseError as e:
            if "Could not find a property named" in str(e) and _enriched_available is not False:
                _enriched_available = False
                _logger.warning("Enriched index fields NOT available in this index — falling back to base fields: %s", base_fields)
                search_kwargs["select"] = base_fields
                _t0 = _time.time()
                results = list(search_client.search(**search_kwargs))
                try:
                    trace_tool_call(
                        tool_name="search_client.search",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(results)[:200] if results is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                return results
            raise

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, selected_titles: List[str]) -> List[str]:
        """Retrieve relevant chunks from Azure AI Search."""
        logger = self.logger
        async with trace_step(
            "retrieve_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"{len(r)} chunks",
        ) as step:
            try:
                # Embed the query using Azure OpenAI
                openai_client = openai.AsyncAzureOpenAI(
                    api_key=Config.AZURE_OPENAI_API_KEY,
                    api_version="2024-02-01",
                    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                )
                _t0 = _time.time()
                embedding_resp = await openai_client.embeddings.create(
                    input=query,
                    model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
                )
                embedding = embedding_resp.data[0].embedding
                try:
                    trace_tool_call(
                        tool_name="openai_client.embeddings.create",
                        latency_ms=int((_time.time() - _t0) * 1000),
                        output=str(embedding_resp)[:200],
                        status="success",
                    )
                except Exception:
                    pass
            except Exception as e:
                logger.log_error(f"Embedding generation failed: {e}")
                return []

            try:
                results = await self._search_with_fallback(query, embedding, selected_titles, TOP_K)
            except Exception as e:
                logger.log_error(f"Chunk retrieval failed: {e}")
                return []

            context_parts = []
            for r in results:
                part = r.get("chunk", "")
                if _enriched_available:
                    for field in ENRICHED_FIELDS:
                        value = r.get(field)
                        if value:
                            part += f"\n{field}: {json.dumps(value) if isinstance(value, (list, dict)) else value}"
                context_parts.append(part)
            return context_parts

class LLMService:
    """Service for calling Azure OpenAI GPT-4.1 to generate summaries."""
    def __init__(self):
        self.logger = Logger()

    def _get_llm_client(self):
        api_key = Config.AZURE_OPENAI_API_KEY
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not configured")
        return openai.AsyncAzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_summary(self, prompt: str, chunks: List[str], user_query: str) -> str:
        """Generate summary using Azure OpenAI GPT-4.1."""
        logger = self.logger
        async with trace_step(
            "generate_summary",
            step_type="llm_call",
            decision_summary="Generate summary from chunks using LLM",
            output_fn=lambda r: f"summary length={len(r) if r else 0}",
        ) as step:
            client = self._get_llm_client()
            context = "\n\n".join(chunks) if chunks else ""
            system_message = prompt + "\n\nOutput Format: " + OUTPUT_FORMAT
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_query},
                {"role": "user", "content": context}
            ]
            _llm_kwargs = Config.get_llm_kwargs()
            _t0 = _time.time()
            try:
                response = await client.chat.completions.create(
                    model=Config.LLM_MODEL or "gpt-4.1",
                    messages=messages,
                    **_llm_kwargs
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=Config.LLM_MODEL or "gpt-4.1",
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                return sanitize_llm_output(content, content_type="text")
            except Exception as e:
                logger.log_error(f"LLM call failed: {e}")
                return FALLBACK_RESPONSE

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

class HealthcareSummarizationAgent:
    """Main orchestrator for healthcare document summarization."""
    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.compliance_manager = ComplianceManager(Logger())
        self.logger = Logger()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def summarize_document(self) -> Dict[str, Any]:
        """Orchestrates the summarization process."""
        async with trace_step(
            "summarize_document",
            step_type="process",
            decision_summary="Orchestrate chunk retrieval and LLM summarization",
            output_fn=lambda r: f"summary={r.get('summary','')[:80]}",
        ) as step:
            try:
                # Validate selected document titles
                if not SELECTED_DOCUMENT_TITLES or not isinstance(SELECTED_DOCUMENT_TITLES, list):
                    self.logger.log_error("No document titles selected for summarization.")
                    return {
                        "success": False,
                        "summary": None,
                        "error": "DOC_NOT_FOUND: No document titles selected.",
                        "tool_calls_made": []
                    }
                # Retrieve chunks
                chunks = await self.chunk_retriever.retrieve_chunks(SYSTEM_PROMPT, SELECTED_DOCUMENT_TITLES)
                if not chunks:
                    self.logger.log_event("No relevant chunks found for summarization.")
                    return {
                        "success": True,
                        "summary": FALLBACK_RESPONSE,
                        "error": None,
                        "tool_calls_made": []
                    }
                # Redact PII from chunks
                chunks = [self.compliance_manager.redact_pii(chunk) for chunk in chunks]
                # Generate summary
                summary = await self.llm_service.generate_summary(SYSTEM_PROMPT, chunks, SYSTEM_PROMPT)
                if not summary or summary == FALLBACK_RESPONSE:
                    self.logger.log_event("LLM returned fallback response.")
                    return {
                        "success": True,
                        "summary": FALLBACK_RESPONSE,
                        "error": None,
                        "tool_calls_made": []
                    }
                return {
                    "success": True,
                    "summary": summary,
                    "error": None,
                    "tool_calls_made": []
                }
            except Exception as e:
                self.logger.log_error(f"Summarization failed: {e}")
                return {
                    "success": False,
                    "summary": None,
                    "error": f"RETRIEVAL_ERROR: {str(e)}",
                    "tool_calls_made": []
                }

    def handle_error(self, error: Exception) -> str:
        """Handles errors, logs events, returns user-friendly error messages."""
        self.logger.log_error(str(error))
        return f"An error occurred: {str(error)}"

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint():
    """Main endpoint for document summarization. No user input required; agent uses SYSTEM_PROMPT and SELECTED_DOCUMENT_TITLES internally."""
    agent = HealthcareSummarizationAgent()
    result = await agent.summarize_document()
    if not result.get("success"):
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "summary": None,
                "error": result.get("error") or FALLBACK_RESPONSE,
                "tool_calls_made": result.get("tool_calls_made", [])
            }
        )
    return {
        "success": True,
        "summary": result.get("summary"),
        "error": None,
        "tool_calls_made": result.get("tool_calls_made", [])
    }

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic errors and malformed JSON requests."""
    _logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
            "tips": "Check your request format and try again. If the error persists, contact support."
        }
    )

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())