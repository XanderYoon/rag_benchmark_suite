from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SUPPORTED_RETRIEVAL_METHODS = ("faiss", "graphrag", "lightrag")
SUPPORTED_RETRIEVAL_PROVIDERS = ("openai", "ollama")
SUPPORTED_BENCHMARK_TOOLS = ("ragas",)


class ToolDefinition(BaseModel):
    """Describe one MCP tool and its JSON-schema input contract."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any]


class KnowledgeBaseSummary(BaseModel):
    """Return validated knowledge base metadata for retrieval calls."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base_path: str
    method_id: str
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    warnings: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """Represent one retrieved chunk returned to MCP clients."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    paper_id: str
    rank: int
    score: float
    text: str
    citation_label: str


class RetrieveEvidenceRequest(BaseModel):
    """Validate inputs for one retrieval-only MCP tool call."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, description="Natural-language retrieval query.")
    knowledge_base_path: str = Field(
        min_length=1,
        description="Directory containing retrieval artifacts such as index manifest and metadata.",
    )
    retrieval_method: Literal["faiss", "graphrag", "lightrag"] | None = Field(
        default=None,
        description="Optional retrieval method override. When omitted, the method is inferred from the KB manifest.",
    )
    retrieval_model: str | None = Field(
        default=None,
        description="Embedding model used for query-time retrieval. Defaults to the KB embedding model when present.",
    )
    retrieval_provider: Literal["openai", "ollama"] | None = Field(
        default=None,
        description="Optional retrieval provider override. Defaults to the KB provider or a model-derived fallback.",
    )
    top_k: int = Field(default=5, ge=1, le=25, description="Maximum number of evidence chunks to return.")


class RetrieveEvidenceResult(BaseModel):
    """Return grounded retrieval payloads with stable keys."""

    model_config = ConfigDict(extra="forbid")

    query: str
    knowledge_base: KnowledgeBaseSummary
    retrieval_method: str
    retrieval_model: str
    retrieval_provider: str
    evidence: list[EvidenceItem]
    warnings: list[str] = Field(default_factory=list)


class BenchmarkTimingSummary(BaseModel):
    """Expose the most useful timing fields from one benchmark run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    actual_total_seconds: float | None = None
    expected_seconds: float | None = None
    telemetry_file: str | None = None
    telemetry_error: str | None = None


class BenchmarkPrimarySummary(BaseModel):
    """Return compact legacy-style metrics for quick inspection."""

    model_config = ConfigDict(extra="forbid")

    baseline: dict[str, Any] = Field(default_factory=dict)
    tool_results: dict[str, Any] = Field(default_factory=dict)
    case_count: int = 0


class RunRetrievalBenchmarkRequest(BaseModel):
    """Validate one benchmark execution request for MCP callers."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    embedded_chunks_path: str = Field(
        min_length=1,
        description="Directory containing prebuilt retrieval artifacts such as a FAISS or graph index.",
    )
    retrieval_model: str = Field(min_length=1, description="Embedding model used by the retrieval runner.")
    evaluation_model: str | None = Field(
        default=None,
        description="Optional evaluation model. Defaults to the benchmark service default.",
    )
    max_cases: int = Field(default=24, ge=1, le=500, description="Maximum number of benchmark probes to execute.")
    top_k: int = Field(default=5, ge=1, le=25, description="Number of retrieved chunks per benchmark case.")
    tools: list[Literal["ragas"]] = Field(
        default_factory=lambda: ["ragas"],
        description="Evaluator tools to run after retrieval.",
    )
    include_auto_probes: bool = Field(default=True, description="Include automatically generated probe cases.")
    include_verified_probes: bool = Field(default=False, description="Include human-verified probe cases.")
    verified_questions_path: str = Field(
        default="data/verified_questions.json",
        description="Path to verified benchmark questions when verified probes are enabled.",
    )
    retrieval_methods: list[Literal["faiss", "graphrag", "lightrag"]] = Field(
        default_factory=lambda: ["faiss"],
        description="Retrieval methods to benchmark. Current runtime executes the first normalized method.",
    )
    telemetry_output_dir: str | None = Field(
        default=None,
        description="Optional directory for benchmark telemetry output.",
    )

    @field_validator("tools", mode="after")
    @classmethod
    def _deduplicate_tools(cls, raw_tools: list[str]) -> list[str]:
        normalized: list[str] = []
        for tool_name in raw_tools:
            if tool_name not in normalized:
                normalized.append(tool_name)
        if not normalized:
            raise ValueError("At least one benchmark evaluation tool must be provided.")
        return normalized

    @field_validator("retrieval_methods", mode="after")
    @classmethod
    def _deduplicate_methods(cls, raw_methods: list[str]) -> list[str]:
        normalized: list[str] = []
        for method_id in raw_methods:
            if method_id not in normalized:
                normalized.append(method_id)
        if not normalized:
            raise ValueError("At least one retrieval method must be provided.")
        return normalized

    @model_validator(mode="after")
    def _validate_probe_sources(self) -> "RunRetrievalBenchmarkRequest":
        if not self.include_auto_probes and not self.include_verified_probes:
            raise ValueError("At least one probe source must be enabled.")
        return self


class RunRetrievalBenchmarkResult(BaseModel):
    """Return one benchmark run payload in a stable MCP-friendly shape."""

    model_config = ConfigDict(extra="forbid")

    retrieval_methods: list[str]
    probe_source_breakdown: dict[str, int] = Field(default_factory=dict)
    primary_summary: BenchmarkPrimarySummary
    timing: BenchmarkTimingSummary
    source_results: dict[str, Any] = Field(default_factory=dict)
    jobs: list[dict[str, Any]] = Field(default_factory=list)


class JsonRpcRequest(BaseModel):
    """Validate one JSON-RPC 2.0 request envelope."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"]
    method: str
    id: str | int | None = None
    params: dict[str, Any] | None = None


class JsonRpcError(BaseModel):
    """Represent a JSON-RPC error object."""

    model_config = ConfigDict(extra="forbid")

    code: int
    message: str
    data: Any | None = None


class JsonRpcResponse(BaseModel):
    """Represent a JSON-RPC success or error response."""

    model_config = ConfigDict(extra="forbid")

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


class ToolCallParams(BaseModel):
    """Validate MCP tool-call params."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
