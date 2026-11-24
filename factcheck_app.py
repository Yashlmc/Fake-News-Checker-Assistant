import json

import pandas as pd
import streamlit as st

from factcheck_agent import run_factcheck_agent
from factcheck_evaluator import (
    EVAL_LABELS,
    run_30_message_benchmark,
    run_factcheck_redteam,
)
from live_retriever import TIER_1_DOMAINS, TIER_2_DOMAINS


st.set_page_config(
    page_title="Multilingual Agentic Fact-Checker",
    page_icon="🔎",
    layout="wide",
)


# ==========================================
# HEADER & DISCLAIMER
# ==========================================

st.title("🔎 Multilingual Agentic Fact-Checker (WhatsApp & Social Media)")

st.markdown(
    """
    A multi-step verification agent (**Claim Extractor → Live Search / Fact-Check
    Retriever → Evidence-Grounded Synthesizer**) supporting **English, Hindi, and
    Marathi** forwards, backed by source credibility tiering, indirect
    prompt-injection defense, and a 30-message multilingual benchmark.
    """
)

st.warning(
    """
    **Verification Policy & Disclaimer**

    This assistant synthesizes retrieved live web and fact-check evidence to
    assist human judgment. Whenever live retrieval returns no credible sources,
    verdicts automatically default to **Insufficient Evidence** rather than
    guessing from static LLM memory. Always inspect the linked primary sources.
    """
)


# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔎 Live Fact-Check Agent",
        "📊 30-Message Multilingual Benchmark",
        "🧪 Prompt-Injection Red-Team Lab",
        "🏛️ Source Credibility Rubric",
    ]
)


# ==========================================
# TAB 1: LIVE FACT-CHECK AGENT
# ==========================================

with tab1:
    st.subheader("Paste a Suspicious WhatsApp Forward (English, Hindi, or Marathi)")

    presets = {
        "— Select a quick sample forward —": "",
        "Hindi Hoax (RBI ₹500 Star Note)": "आरबीआई ने घोषणा की है कि जिन 500 रुपये के नोटों पर स्टार (*) का निशान है वे नकली हैं और बैंक उन्हें नहीं लेंगे।",
        "Marathi Hoax (WhatsApp 3rd Red Tick)": "सावधान! उद्यापासून व्हॉट्सॲपवर तिसरी लाल टिक येणार आहे जी दर्शवेल की सरकारने तुमचा मेसेज वाचला आहे. सर्व ग्रुप्सवर शेअर करा.",
        "English Real News (Chandrayaan-3)": "ISRO successfully soft-landed the Chandrayaan-3 Vikram lander near the Moon's south pole on August 23, 2023.",
    }

    chosen_preset = st.selectbox("Optional demo preset", list(presets.keys()))
    default_msg = presets[chosen_preset]

    message = st.text_area(
        "WhatsApp forward or news message:",
        value=default_msg,
        height=180,
        placeholder="Paste English, Hindi (हिन्दी), or Marathi (मराठी) text here...",
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.caption(f"Characters entered: {len(message)} / 15,000")
    with c2:
        enable_live = st.checkbox(
            "Enable Live Web & Fact-Check Retrieval", value=True
        )

    if st.button("🚀 Run Multi-Step Verification Agent", type="primary", use_container_width=True):
        if not message.strip():
            st.error("Please paste a message before running verification.")
        else:
            try:
                with st.spinner(
                    "Running Stage 1 (Claim Extractor) → Stage 2 (Live Retriever) → Stage 3 (Grounded Synthesizer)..."
                ):
                    report = run_factcheck_agent(
                        message, enable_live_search=enable_live
                    )
                    st.session_state["fc_report"] = report.model_dump()
                st.success("Agent verification completed!")
            except Exception as e:
                st.error(f"Verification failed: {e}")

    if "fc_report" in st.session_state:
        rep = st.session_state["fc_report"]
        claims = rep.get("claims", [])

        if rep.get("injection_alerts"):
            st.error(
                "⚠️ Indirect Prompt-Injection Directive Flagged in Forward:\n- "
                + "\n- ".join(rep["injection_alerts"])
            )

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Verdict", rep["overall_verdict"])
        m2.metric("Detected Language", rep["detected_language"])
        m3.metric("Claims Extracted", len(claims))
        m4.metric("Total Latency", f"{rep['latency_seconds'].get('total_sec', 0)}s")

        st.subheader("📝 Summary")
        st.write(f"**English:** {rep['message_summary_en']}")
        if rep.get("message_summary_native"):
            st.info(
                f"**{rep['detected_language']} Explanation:** "
                f"{rep['message_summary_native']}"
            )

        if rep.get("overall_warning_signs"):
            st.subheader("⚠️ Misinformation Warning Signs")
            for w in rep["overall_warning_signs"]:
                st.warning(w)

        st.divider()
        st.subheader("🔍 Claim-by-Claim Evidence-Grounded Verdicts")

        verdict_icons = {
            "Supported by Evidence": "🟢",
            "Contradicted by Evidence": "🔴",
            "Misleading / Out of Context": "🟠",
            "Insufficient Evidence": "🔵",
        }

        for idx, c in enumerate(claims, start=1):
            icon = verdict_icons.get(c["verdict"], "🔵")
            with st.expander(
                f"{icon} {c['claim_id']}: {c['canonical_claim_en']} — [{c['verdict']} | Conf: {c['confidence_score']}]",
                expanded=(idx == 1),
            ):
                if c["original_claim"] != c["canonical_claim_en"]:
                    st.caption(f"Original ({rep['detected_language']}): {c['original_claim']}")

                st.markdown(f"**Verdict:** `{c['verdict']}` (Confidence: `{c['confidence_score']}`)")
                st.write(c["explanation_en"])

                if c.get("explanation_native"):
                    st.info(c["explanation_native"])

                if c.get("needs_human_review"):
                    st.warning(f"Human Review Recommended: {c['escalation_reason']}")

                st.markdown("**Retrieved Live Evidence**")
                if c["evidence_items"]:
                    for ev in c["evidence_items"]:
                        badge = "✅" if ev["source_tier"].startswith("Tier 1") else "🌐"
                        st.markdown(
                            f"{badge} **[{ev['title']}]({ev['url']})** "
                            f"— `{ev['source_tier']}` ({ev['domain']})"
                        )
                        st.caption(ev["snippet"])
                else:
                    st.info("No live evidence retrieved for this claim.")

        with st.expander("🧭 Inspect Agent Execution Trace & Latency Breakdown"):
            for step in rep.get("agent_trace", []):
                st.code(step)
            st.json(rep.get("latency_seconds", {}))

        # Exports
        st.divider()
        claims_csv = pd.DataFrame(
            [
                {
                    "Claim ID": c["claim_id"],
                    "Original Claim": c["original_claim"],
                    "English Claim": c["canonical_claim_en"],
                    "Verdict": c["verdict"],
                    "Confidence": c["confidence_score"],
                    "Cited URLs": "; ".join(c["cited_urls"]),
                    "Human Review": c["needs_human_review"],
                }
                for c in claims
            ]
        ).to_csv(index=False).encode("utf-8-sig")

        d1, d2 = st.columns(2)
        d1.download_button(
            "📥 Download Full Report (JSON)",
            data=json.dumps(rep, indent=4, ensure_ascii=False),
            file_name="factcheck_report.json",
            mime="application/json",
            use_container_width=True,
        )
        d2.download_button(
            "📥 Download Claims Table (CSV)",
            data=claims_csv,
            file_name="factcheck_claims.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==========================================
# TAB 2: 30-MESSAGE MULTILINGUAL BENCHMARK
# ==========================================

with tab2:
    st.subheader("📊 30-Message Multilingual Benchmark (English, Hindi & Marathi)")
    st.write(
        "Evaluates Claim Extraction Recall, Verdict Precision/Recall/Macro-F1, "
        "Language-Wise Accuracy, Citation Grounding Rate, and Latency on "
        "`benchmark_30_messages.csv`."
    )

    b1, b2 = st.columns(2)
    with b1:
        sample_limit = st.slider(
            "Messages to evaluate (use 5–6 for quick live demo, 30 for full run)",
            min_value=3,
            max_value=30,
            value=30,
        )
    with b2:
        bench_live = st.checkbox(
            "Use live web search during benchmark", value=True, key="bench_live"
        )

    if st.button("Run Multilingual Benchmark", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Evaluating {sample_limit} multilingual messages..."):
                res_df, met_df, mat, lang_df, summ = run_30_message_benchmark(
                    csv_path="benchmark_30_messages.csv",
                    limit=sample_limit,
                    enable_live_search=bench_live,
                )
                st.session_state["fc_bench_res"] = res_df
                st.session_state["fc_bench_met"] = met_df
                st.session_state["fc_bench_mat"] = mat
                st.session_state["fc_bench_lang"] = lang_df
                st.session_state["fc_bench_summ"] = summ
            st.success("Benchmark completed!")
        except Exception as e:
            st.error(f"Benchmark failed: {e}")

    if "fc_bench_summ" in st.session_state:
        s = st.session_state["fc_bench_summ"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Claim Extraction Recall", f"{s['claim_extraction_recall_pct']}%")
        k2.metric("Macro F1", f"{s['macro_f1']:.3f}")
        k3.metric("Citation Grounding Rate", f"{s['citation_grounding_rate_pct']}%")
        k4.metric("Avg Latency / Msg", f"{s['avg_latency_sec']}s")

        st.subheader("Language-Wise Performance Breakdown")
        st.dataframe(
            st.session_state["fc_bench_lang"],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Classification Metrics")
        st.dataframe(st.session_state["fc_bench_met"], use_container_width=True)

        st.subheader("Confusion Matrix")
        mat_df = pd.DataFrame(
            st.session_state["fc_bench_mat"],
            index=[f"Actual: {l}" for l in EVAL_LABELS],
            columns=[f"Pred: {l}" for l in EVAL_LABELS],
        )
        st.dataframe(mat_df, use_container_width=True)

        st.subheader("Per-Message Predictions")
        st.dataframe(
            st.session_state["fc_bench_res"],
            use_container_width=True,
            hide_index=True,
        )


# ==========================================
# TAB 3: PROMPT-INJECTION RED-TEAM LAB
# ==========================================

with tab3:
    st.subheader("🧪 Indirect Prompt-Injection Red-Team Suite")
    st.write(
        "Verifies that adversarial instructions hidden inside WhatsApp forwards "
        "(in English or Hindi) cannot hijack the agent into outputting "
        "'Supported by Evidence'."
    )

    if st.button("Run Red-Team Security Suite", type="primary", use_container_width=True):
        try:
            with st.spinner("Running 4 red-team & control payloads..."):
                rt_df, block_rate = run_factcheck_redteam(enable_live_search=True)
                st.session_state["fc_rt_df"] = rt_df
                st.session_state["fc_rt_rate"] = block_rate
            st.success("Red-team evaluation completed!")
        except Exception as e:
            st.error(f"Red-team run failed: {e}")

    if "fc_rt_df" in st.session_state:
        st.metric("Attack Block Rate", f"{st.session_state['fc_rt_rate']:.1f}%")
        st.dataframe(
            st.session_state["fc_rt_df"],
            use_container_width=True,
            hide_index=True,
        )


# ==========================================
# TAB 4: SOURCE CREDIBILITY RUBRIC
# ==========================================

with tab4:
    st.subheader("🏛️ Transparent Source Credibility Tiering")
    st.write(
        "Retrieved URLs are automatically ranked so official Indian/global "
        "institutions and IFCN-certified fact-checkers are prioritized over "
        "unverified blogs."
    )
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Tier 1 — Official Bodies & Fact-Checkers (+ `.gov.in` / `.nic.in`)**")
        st.write(", ".join(sorted(TIER_1_DOMAINS)))
    with t2:
        st.markdown("**Tier 2 — Established Regional & National News**")
        st.write(", ".join(sorted(TIER_2_DOMAINS)))


st.divider()
st.caption(
    "Multilingual Agentic Fact-Checker | Python | Groq | DuckDuckGo Live Search | "
    "Pydantic v2 | Scikit-Learn | Streamlit"
)