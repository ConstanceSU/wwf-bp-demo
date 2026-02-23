import re
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# CONFIG (assumes app.py is in demo/ with the csvs)
# -----------------------------
PATHS = {
    "company_exposure": "company_exposure.csv",
    "basin_risk": "basin_risk.csv",
    "country_risk_suite": "country_risk_suite.csv",
    "intervention_logic": "intervention_logic.csv",
    "nbs_projects": "nbs_projects.csv",
    "project_metrics": "project_metrics.csv",
    "sbtn_mapping": "sbtn_mapping.csv",  # if your file is named sbtn_mapping.csv
}

LATEST_YEAR_ONLY = True
TOP_K_PROJECTS = 5

# -----------------------------
# DATA LOADING (cached)
# -----------------------------
@st.cache_data
def load_data():
    df_ce = pd.read_csv(PATHS["company_exposure"])
    df_basin = pd.read_csv(PATHS["basin_risk"])
    df_country = pd.read_csv(PATHS["country_risk_suite"])
    df_logic = pd.read_csv(PATHS["intervention_logic"])
    df_projects = pd.read_csv(PATHS["nbs_projects"])
    df_metrics = pd.read_csv(PATHS["project_metrics"])
    df_map = pd.read_csv(PATHS["sbtn_mapping"])
    return df_ce, df_basin, df_country, df_logic, df_projects, df_metrics, df_map

# -----------------------------
# HELPERS
# -----------------------------
def split_pipe(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]

def safe_lower(x):
    return str(x).strip().lower()

def pick_company_from_text(text, company_list):
    """Try to detect a company name in user text."""
    t = safe_lower(text)
    for c in company_list:
        if safe_lower(c) in t:
            return c
    return None

def stage_from_sbtn_row(row):
    """
    A lightweight mapping based on the ambition board fields already used to make company_exposure.csv.
    But company_exposure.csv already contains sbtn_stage in your generated version.
    So we rely on df_ce['sbtn_stage'] directly.
    """
    return row

def score_project_row(r, risk_tol="medium"):
    """
    Simple scoring:
      - evidence: high>medium>low
      - confidence_score from metrics
      - cheaper is slightly better (optional)
      - adjust by risk tolerance: low tolerance prefers higher evidence/confidence
    """
    ev = safe_lower(r.get("evidence_level", "medium"))
    ev_score = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(ev, 0.6)

    conf = r.get("confidence_score", np.nan)
    conf_score = float(conf) if pd.notna(conf) else 0.6

    cost = r.get("estimated_cost_usd", np.nan)
    if pd.notna(cost) and cost > 0:
        cost_score = 1.0 / (1.0 + np.log10(cost))  # cheap -> higher
    else:
        cost_score = 0.4

    rt = safe_lower(risk_tol)
    if rt == "low":
        return 0.55 * ev_score + 0.35 * conf_score + 0.10 * cost_score
    if rt == "high":
        return 0.40 * ev_score + 0.30 * conf_score + 0.30 * cost_score
    return 0.50 * ev_score + 0.35 * conf_score + 0.15 * cost_score

def build_recommendation(company_name, df_ce, df_basin, df_logic, df_projects, df_metrics, df_map):
    # 1) company exposure
    ce = df_ce[df_ce["company_name"] == company_name].copy()
    if ce.empty:
        return None, "I couldn't find that company in company_exposure.csv."

    # risk tolerance (pick most common)
    risk_tol = ce["risk_tolerance"].mode().iloc[0] if "risk_tolerance" in ce.columns else "medium"
    sector = ce["sector"].mode().iloc[0] if "sector" in ce.columns else ""
    hq = ce["hq_country"].mode().iloc[0] if "hq_country" in ce.columns else ""

    # 2) join basin risk
    basins = ce.merge(df_basin, on="basin_id", how="left", suffixes=("", "_basin"))

    # derive main risk drivers (from basin_risk.csv already)
    basins["risk_driver_list"] = basins["risk_driver"].apply(split_pipe)

    # aggregate drivers weighted by revenue_share_percent
    driver_scores = {}
    for _, r in basins.iterrows():
        w = float(r.get("revenue_share_percent", 0) or 0)
        for d in r["risk_driver_list"]:
            driver_scores[d] = driver_scores.get(d, 0) + w

    if not driver_scores:
        driver_scores = {"low_materiality": 1.0}

    top_drivers = sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_driver_names = [d for d, _ in top_drivers if d != "low_materiality"]

    # 3) map risk drivers -> intervention types
    logic = df_logic.copy()
    logic = logic[logic["risk_driver"].isin(top_driver_names)].copy()
    if logic.empty:
        # fallback: use any intervention types if driver not found
        logic = df_logic.copy()

    # choose top intervention types (unique)
    intervention_types = logic["intervention_type"].dropna().unique().tolist()

    # 4) candidate projects: match by basin_id first, then intervention_type
    basin_ids = basins["basin_id"].dropna().unique().tolist()
    cand = df_projects[df_projects["basin_id"].isin(basin_ids)].copy()
    if not cand.empty:
        cand = cand[cand["intervention_type"].isin(intervention_types)].copy()

    # fallback if none: country-level match
    if cand.empty and "country_iso3" in df_projects.columns and "country_iso3" in basins.columns:
        iso3s = basins["country_iso3"].dropna().unique().tolist()
        cand = df_projects[df_projects["country_iso3"].isin(iso3s)].copy()
        cand = cand[cand["intervention_type"].isin(intervention_types)].copy()

    if cand.empty:
        return None, "I found no projects matching your exposed basins + intervention logic. (Try increasing mock projects.)"

    # 5) attach latest metrics
    m = df_metrics.copy()
    if LATEST_YEAR_ONLY and "year" in m.columns:
        max_year = m["year"].max()
        m = m[m["year"] == max_year].copy()

    # keep one metric row per project by taking highest confidence metric (demo)
    m = m.sort_values(["project_id", "confidence_score"], ascending=[True, False])
    m_best = m.groupby("project_id", as_index=False).head(1).copy()

    cand = cand.merge(m_best, on="project_id", how="left")

    # 6) map metric -> SBTN language
    cand = cand.merge(df_map, on="metric_name", how="left")

    # 7) score + rank
    cand["score"] = cand.apply(lambda r: score_project_row(r, risk_tol=risk_tol), axis=1)
    top = cand.sort_values("score", ascending=False).head(TOP_K_PROJECTS).copy()

    # Build assistant narrative
    # Basin summary
    basin_summary = basins[["basin_id", "country", "revenue_share_percent", "risk_driver"]].copy()
    basin_summary = basin_summary.sort_values("revenue_share_percent", ascending=False).head(5)

    # Construct SBTN-aligned “insight”
    sbtn_targets = ce["sbtn_target_areas"].mode().iloc[0] if "sbtn_target_areas" in ce.columns else "Unknown"
    sbtn_stage = ce["sbtn_stage"].mode().iloc[0] if "sbtn_stage" in ce.columns else "Unknown"

    narrative = []
    narrative.append(f"**Company:** {company_name}  \n**Sector:** {sector}  \n**HQ:** {hq}  \n**SBTN stage:** {sbtn_stage}  \n**Target areas:** {sbtn_targets}")
    narrative.append("")
    narrative.append("### What we see from your exposure + basin risk")
    drivers_text = ", ".join([f"`{d}`" for d in top_driver_names]) if top_driver_names else "`low_materiality`"
    narrative.append(f"- Main risk drivers (weighted by exposure): {drivers_text}")
    narrative.append(f"- Risk tolerance (demo): `{risk_tol}`")
    narrative.append("")
    narrative.append("### Recommended NbS projects (ranked)")
    narrative.append(
        "Below are the top candidates filtered by your exposed basins and intervention logic, "
        "then ranked by evidence/confidence/cost."
    )

    # Create a compact table for display
    show_cols = [
        "project_id","project_name","basin_id","country_iso3","intervention_type","stage","status",
        "metric_name","value","unit","sbtn_target_area","claim_type","attribution_risk","score"
    ]
    show_cols = [c for c in show_cols if c in top.columns]
    top_table = top[show_cols].copy()

    return {
        "narrative_md": "\n".join(narrative),
        "top_projects": top_table,
        "basin_summary": basin_summary
    }, None

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="WWF NbS Virtual Assistant (Demo)", layout="wide")
st.title("🌿 WWF NbS Virtual Assistant — Demo (Mock Data)")

df_ce, df_basin, df_country, df_logic, df_projects, df_metrics, df_map = load_data()

company_list = sorted(df_ce["company_name"].dropna().unique().tolist())
default_company = company_list[0] if company_list else None

with st.sidebar:
    st.header("Demo controls")

    selected_company = st.selectbox(
        "Select a company",
        company_list,
        index=0
    )

    st.caption("This is driven by company_exposure.csv (mock exposures + SBTN stage).")

    # ✅ ADD THIS BLOCK HERE
    st.session_state.setdefault("demo_query", "")

    st.subheader("Quick questions (1-click)")
    q1 = st.button("Top freshwater interventions")
    q2 = st.button("Where are my biggest basin risks?")
    q3 = st.button("Give me SBTN-ready claims I can report")
    q4 = st.button("Cheapest high-confidence projects")

    if q1: st.session_state.demo_query = "What are my top freshwater interventions?"
    if q2: st.session_state.demo_query = "Where are my biggest basin risks and why?"
    if q3: st.session_state.demo_query = "Draft SBTN-aligned claims from recommended projects."
    if q4: st.session_state.demo_query = "Show the cheapest high-confidence projects."

    run_now = st.button("Run recommendation")

    # ⬇ your existing divider
    st.divider()
    st.caption("Data files loaded:")
    for k, v in PATHS.items():
        st.write(f"- {v}")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi — tell me what you want to achieve (e.g., freshwater targets in priority basins), and I’ll suggest NbS options."}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------------
# 1️⃣ Capture chat input
# -------------------------------------------------------
user_text = st.chat_input(
    "Ask the assistant... (e.g., 'What are my top freshwater interventions?')"
)

# -------------------------------------------------------
# 2️⃣ Inject quick question if clicked
# -------------------------------------------------------
if not user_text and st.session_state.get("demo_query"):
    user_text = st.session_state.demo_query
    st.session_state.demo_query = ""

# -------------------------------------------------------
# 3️⃣ Inject default prompt if 'Run recommendation' clicked
# -------------------------------------------------------
if run_now and not user_text:
    user_text = "Give me recommended NbS projects and SBTN-ready claims."

# -------------------------------------------------------
# 4️⃣ Main execution block
# -------------------------------------------------------
if user_text:

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_text}
    )

    with st.chat_message("user"):
        st.markdown(user_text)

    # Detect company name inside text (or fallback to dropdown)
    detected = pick_company_from_text(user_text, company_list)
    company = detected or selected_company

    # Build recommendation
    result, err = build_recommendation(
        company,
        df_ce,
        df_basin,
        df_logic,
        df_projects,
        df_metrics,
        df_map,
    )

    # Assistant response
    with st.chat_message("assistant"):

        if err:
            st.markdown(f"⚠️ {err}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ {err}"}
            )

        else:
            # Narrative explanation
            st.markdown(result["narrative_md"])

            # Top projects
            st.markdown("**Top projects (click to sort):**")
            st.dataframe(result["top_projects"], use_container_width=True)

            # Basin transparency
            st.markdown("**Top exposed basins (for transparency):**")
            st.dataframe(result["basin_summary"], use_container_width=True)

            # Save assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": result["narrative_md"]}
            )

