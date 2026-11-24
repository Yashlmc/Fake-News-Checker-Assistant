import os
from urllib.parse import urlparse

import requests

from factcheck_schemas import EvidenceItem


# ==================================================
# 1. DOMAIN CREDIBILITY TIERING RUBRIC
# ==================================================

TIER_1_DOMAINS = {
    # Indian Fact-Checkers (IFCN signatories) & Official Bodies
    "pib.gov.in",
    "factcheck.pib.gov.in",
    "altnews.in",
    "boomlive.in",
    "factly.in",
    "vishvasnews.com",
    "newschecker.in",
    "thequint.com",
    "rbi.org.in",
    "mohfw.gov.in",
    "isro.gov.in",
    "eci.gov.in",
    "uidai.gov.in",
    "npci.org.in",
    "india.gov.in",
    "meity.gov.in",
    # Global Official & Fact-Checkers
    "who.int",
    "cdc.gov",
    "unesco.org",
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "fullfact.org",
    "reuters.com",
    "apnews.com",
    "afp.com",
}

TIER_2_DOMAINS = {
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "timesofindia.indiatimes.com",
    "ndtv.com",
    "livemint.com",
    "economictimes.indiatimes.com",
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "nytimes.com",
    "scroll.in",
    "lokmat.com",
    "maharashtratimes.com",
    "esakal.com",
    "amarujala.com",
    "bhaskar.com",
    "jagran.com",
}


def extract_domain(url):
    """Extract normalized root domain (without www.) from a URL."""
    try:
        host = urlparse(str(url)).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def classify_source_tier(url):
    """Classify a retrieved URL into Tier 1, Tier 2, or Tier 3 credibility."""
    domain = extract_domain(url)
    if not domain:
        return "Tier 3 (General Web)"

    for d in TIER_1_DOMAINS:
        if domain == d or domain.endswith("." + d):
            return "Tier 1 (Official / Fact-Checker)"

    if domain.endswith(".gov.in") or domain.endswith(".nic.in") or domain.endswith(".gov"):
        return "Tier 1 (Official / Fact-Checker)"

    for d in TIER_2_DOMAINS:
        if domain == d or domain.endswith("." + d):
            return "Tier 2 (Established News)"

    return "Tier 3 (General Web)"


# ==================================================
# 2. LIVE DUCKDUCKGO RETRIEVER
# ==================================================

def _search_duckduckgo(query, max_results=5):
    """Run live web search using ddgs (or duckduckgo_search fallback)."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    items = []
    with DDGS() as ddgs:
        results = ddgs.text(query, region="in-en", max_results=max_results)
        for r in results or []:
            url = r.get("href") or r.get("url") or ""
            title = (r.get("title") or "").strip()
            snippet = (r.get("body") or r.get("snippet") or "").strip()
            if not url or not snippet:
                continue
            items.append(
                EvidenceItem(
                    title=title or "Untitled Result",
                    url=url,
                    domain=extract_domain(url),
                    snippet=snippet[:450],
                    source_tier=classify_source_tier(url),
                    retriever_source="DuckDuckGo",
                )
            )
    return items


# ==================================================
# 3. OPTIONAL GOOGLE FACT CHECK TOOLS API
# ==================================================

def _search_google_factcheck(query, max_results=3):
    """
    Query Google Fact Check Tools API if GOOGLE_FACTCHECK_API_KEY is set in .env.
    Gracefully returns [] when no key is configured.
    """
    api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if not api_key:
        return []

    try:
        resp = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query, "key": api_key, "pageSize": max_results},
            timeout=8,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    items = []
    for claim_obj in payload.get("claims", []):
        claim_text = claim_obj.get("text", "")
        for rev in claim_obj.get("claimReview", [])[:1]:
            url = rev.get("url", "")
            rating = rev.get("textualRating", "Reviewed")
            publisher = rev.get("publisher", {}).get("name", "Fact-Checker")
            title = rev.get("title") or f"{publisher}: {rating}"
            if not url:
                continue
            items.append(
                EvidenceItem(
                    title=f"[{rating}] {title}",
                    url=url,
                    domain=extract_domain(url),
                    snippet=f"Fact-check rating '{rating}' by {publisher} for claim: {claim_text}"[:450],
                    source_tier="Tier 1 (Official / Fact-Checker)",
                    retriever_source="Google Fact Check API",
                )
            )
    return items


# ==================================================
# 4. UNIFIED LIVE EVIDENCE RETRIEVER
# ==================================================

TIER_PRIORITY = {
    "Tier 1 (Official / Fact-Checker)": 0,
    "Tier 2 (Established News)": 1,
    "Tier 3 (General Web)": 2,
}


def retrieve_live_evidence(search_queries, max_items=6, enable_live_search=True):
    """
    Execute multi-query live retrieval across Google Fact Check API (if configured)
    and DuckDuckGo, deduplicate by URL, and rank Tier-1 official/fact-check sources first.
    """
    if not enable_live_search:
        return [], "offline_no_live_search"

    by_url = {}
    any_success = False

    for q in search_queries[:2]:
        q = str(q).strip()
        if not q:
            continue

        # 1. Google Fact Check API (if key set)
        for item in _search_google_factcheck(q, max_results=3):
            any_success = True
            by_url.setdefault(item.url, item)

        # 2. DuckDuckGo general + fact-check targeted search
        try:
            for item in _search_duckduckgo(q, max_results=4):
                any_success = True
                by_url.setdefault(item.url, item)
        except Exception:
            pass

    if not by_url and not any_success:
        return [], "offline_no_live_search"

    ranked = sorted(
        by_url.values(),
        key=lambda x: (TIER_PRIORITY.get(x.source_tier, 2), x.retriever_source != "Google Fact Check API"),
    )
    return ranked[:max_items], "live_web"