from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


DetectedLanguage = Literal["English", "Hindi", "Marathi", "Hinglish/Mixed", "Other"]

VerdictLabel = Literal[
    "Supported by Evidence",
    "Contradicted by Evidence",
    "Misleading / Out of Context",
    "Insufficient Evidence",
]

SourceTier = Literal[
    "Tier 1 (Official / Fact-Checker)",
    "Tier 2 (Established News)",
    "Tier 3 (General Web)",
]


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    original_claim: str = Field(min_length=1)
    canonical_claim_en: str = Field(min_length=1)
    check_worthy: bool = True
    warning_signs: List[str] = Field(default_factory=list)
    search_queries: List[str] = Field(min_length=1, max_length=3)


class ClaimExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detected_language: DetectedLanguage
    message_summary_en: str = Field(min_length=1)
    overall_warning_signs: List[str] = Field(default_factory=list)
    claims: List[ExtractedClaim] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    domain: str
    snippet: str
    source_tier: SourceTier
    retriever_source: str = "DuckDuckGo"


class VerifiedClaimResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    original_claim: str
    canonical_claim_en: str
    verdict: VerdictLabel
    confidence_score: float = Field(ge=0.0, le=1.0)
    explanation_en: str = Field(min_length=1)
    explanation_native: Optional[str] = None
    warning_signs: List[str] = Field(default_factory=list)
    cited_urls: List[str] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    verification_steps: List[str] = Field(default_factory=list)
    needs_human_review: bool = False
    escalation_reason: Optional[str] = None


class FactCheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detected_language: DetectedLanguage
    retrieval_mode: Literal["live_web", "offline_no_live_search"]
    overall_verdict: VerdictLabel
    message_summary_en: str
    message_summary_native: Optional[str] = None
    overall_warning_signs: List[str] = Field(default_factory=list)
    claims: List[VerifiedClaimResult] = Field(default_factory=list)
    general_advice: List[str] = Field(default_factory=list)
    injection_alerts: List[str] = Field(default_factory=list)
    agent_trace: List[str] = Field(default_factory=list)
    latency_seconds: dict = Field(default_factory=dict)
    disclaimer: str = (
        "This report synthesizes retrieved web and fact-check evidence via AI. "
        "It is a verification aid, not an infallible determination of truth. "
        "Always inspect the linked primary sources before forwarding."
    )