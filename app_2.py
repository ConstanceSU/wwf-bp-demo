import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Nature Intelligence MVP",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4f4f4f;
        margin-bottom: 1.4rem;
    }
    .card {
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid #e6e6e6;
        background-color: #fafafa;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #6b6b6b;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .small-note {
        font-size: 0.9rem;
        color: #6b6b6b;
    }
    div.stButton > button {
        height: 86px;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 18px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def split_to_set(value):
    """
    Turn raw cell content into a clean set of tokens.
    Handles separators safely and removes junk values like N/A, nan, etc.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()

    if isinstance(value, (list, tuple, set)):
        raw_items = [str(v).strip().lower() for v in value]
    else:
        text = str(value).strip().lower()

        # keep "/" intact because values like n/a should not become n;a
        for sep in ["|", ",", ";"]:
            text = text.replace(sep, ";")

        raw_items = [item.strip() for item in text.split(";") if item.strip()]

    bad_values = {
        "", "n/a", "na", "nan", "none", "null", "unknown", "-", "--", "not available"
    }

    clean_items = set()
    for item in raw_items:
        if item in bad_values:
            continue
        if len(item) <= 1:
            continue
        clean_items.add(item)

    return clean_items

def safe_get(row, possible_cols, default="N/A"):
    for col in possible_cols:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip() != "":
            return row[col]
    return default

def find_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_columns(df, expected_cols):
    """
    Add missing columns as empty strings so app doesn't break.
    """
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    return df

def score_overlap(company_set, project_set):
    if not company_set or not project_set:
        return 0, set()
    overlap = company_set.intersection(project_set)
    return len(overlap), overlap

def format_currency(value):
    if pd.isna(value) or value is None or value == "":
        return "N/A"
    try:
        value = float(value)
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.1f}bn"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.1f}m"
        elif value >= 1_000:
            return f"${value/1_000:.0f}k"
        return f"${value:,.0f}"
    except:
        return str(value)

def get_financial_view(rec, df_finance):
    """
    Robust finance matching:
    1) exact project name
    2) exact project id
    3) same region + intervention type
    4) same intervention type
    """
    if df_finance is None or df_finance.empty:
        return None

    finance_df = df_finance.copy()

    finance_project_name_col = get_best_available_column(finance_df, [["project", "name"]])
    finance_project_id_col = get_best_available_column(finance_df, [["project", "id"]])
    finance_region_col = get_best_available_column(finance_df, [["region"]]) or get_best_available_column(finance_df, [["country"]])
    finance_intervention_col = get_best_available_column(finance_df, [["intervention"]])

    cost_col = detect_numeric_column(finance_df, ["project_cost", "estimated_project_cost", "total_cost", "cost"])

    project_name = normalize_string(rec.get("project_name"))
    project_id = normalize_string(rec.get("project_id"))
    region = normalize_string(rec.get("region"))
    intervention_type = normalize_string(rec.get("intervention_type"))

    match = pd.DataFrame()

    if finance_project_name_col and project_name:
        tmp = finance_df[
            finance_df[finance_project_name_col].astype(str).str.strip().str.lower() == project_name
        ]
        if not tmp.empty:
            match = tmp

    if match.empty and finance_project_id_col and project_id:
        tmp = finance_df[
            finance_df[finance_project_id_col].astype(str).str.strip().str.lower() == project_id
        ]
        if not tmp.empty:
            match = tmp

    if match.empty and finance_region_col and finance_intervention_col and region and intervention_type:
        tmp = finance_df[
            (finance_df[finance_region_col].astype(str).str.strip().str.lower() == region) &
            (finance_df[finance_intervention_col].astype(str).str.strip().str.lower() == intervention_type)
        ]
        if not tmp.empty:
            match = tmp

    if match.empty and finance_intervention_col and intervention_type:
        tmp = finance_df[
            finance_df[finance_intervention_col].astype(str).str.strip().str.lower() == intervention_type
        ]
        if not tmp.empty:
            match = tmp

    if match.empty:
        return None

    def pick_numeric_mean(df_slice, col):
        if col is None or col not in df_slice.columns:
            return None
        series = pd.to_numeric(df_slice[col], errors="coerce")
        if series.notna().sum() > 0:
            return float(series.mean())
        return None

    total_cost = pick_numeric_mean(match, cost_col)

    if total_cost is None:
        return None

    # ------------------------------------------
    # Derived demo assumptions
    # ------------------------------------------
    company_contribution = total_cost * 0.25

    if total_cost < 1_000_000:
        time_horizon = 2
    elif total_cost < 3_000_000:
        time_horizon = 3
    elif total_cost < 7_000_000:
        time_horizon = 5
    else:
        time_horizon = 7

    if company_contribution < 1_000_000:
        effort = "Low"
    elif company_contribution < 3_000_000:
        effort = "Medium"
    else:
        effort = "High"

    return {
        "total_cost": format_currency(total_cost),
        "company_contribution": format_currency(company_contribution),
        "time_horizon": time_horizon,
        "effort": effort
    }

def normalize_string(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().lower()

def find_columns_by_keywords(df, keywords, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    cols = []
    for col in df.columns:
        c = col.lower()
        if any(k in c for k in keywords) and not any(ex in c for ex in exclude_keywords):
            cols.append(col)
    return cols

def collect_row_tokens(row, columns):
    tokens = set()
    for col in columns:
        if col in row.index:
            tokens |= split_to_set(row[col])
    return tokens

def overlap_score(company_tokens, project_tokens, weight_per_match=1.0, max_matches=4):
    if not company_tokens or not project_tokens:
        return 0.0, set()

    overlap = company_tokens.intersection(project_tokens)
    score = min(len(overlap), max_matches) * weight_per_match
    return score, overlap

def get_best_available_column(df, keyword_groups):
    """
    keyword_groups example:
    [["project_name", "project"], ["id"]]
    returns best-matching column or None
    """
    lower_map = {c.lower(): c for c in df.columns}
    for group in keyword_groups:
        for c in df.columns:
            cl = c.lower()
            if all(k in cl for k in group):
                return c
    return None

def detect_numeric_column(df, include_keywords):
    candidates = []
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in include_keywords):
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append(col)

    if candidates:
        return candidates[0]

    # second pass: try coercion
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in include_keywords):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                return col

    return None

def format_currency(value):
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        value = float(value)
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.1f}bn"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.1f}m"
        elif value >= 1_000:
            return f"${value/1_000:.0f}k"
        return f"${value:,.0f}"
    except Exception:
        return str(value)
    
# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data():
    """
    Expected files in same folder as app.py:
    - company_exposure.csv
    - nbs_projects.csv
    - project_financial_output.csv   (optional)
    """
    base = Path(".")

    company_path = base / "company_exposure.csv"
    project_path = base / "nbs_projects.csv"
    finance_path = base / "project_financial_output.csv"

    if not company_path.exists() or not project_path.exists():
        return None, None, None

    df_company = pd.read_csv(company_path)
    df_projects = pd.read_csv(project_path)

    # Optional finance file
    if finance_path.exists():
        df_finance = pd.read_csv(finance_path)
    else:
        df_finance = pd.DataFrame()

    # Make app robust even if some columns are missing
    df_company = ensure_columns(df_company, [
        "company_name",
        "hotspots",
        "risk_exposure",
        "sbtn_targets",
        "business_priorities",
        "company_summary"
    ])

    df_projects = ensure_columns(df_projects, [
        "project_name",
        "region",
        "ecosystem",
        "intervention_type",
        "project_summary",
        "hotspot_tags",
        "risk_tags",
        "sbtn_tags",
        "business_tags"
    ])

    if not df_finance.empty:
        df_finance = ensure_columns(df_finance, [
            "project_name",
            "estimated_project_cost_usd",
            "suggested_company_contribution_usd",
            "time_horizon_years",
            "financial_effort"
        ])

    return df_company, df_projects, df_finance

# =========================================================
# RECOMMENDATION ENGINE
# =========================================================
def build_recommendation(company_name, df_company, df_projects, df_finance=None):
    company_col = find_first_existing(df_company, ["company_name", "company", "name"])
    if company_col is None:
        return None, "No company name column found in company_exposure.csv."

    company_match = df_company[
        df_company[company_col].astype(str).str.strip().str.lower() == company_name.strip().lower()
    ]

    if company_match.empty:
        return None, f"I couldn't find '{company_name}' in company_exposure.csv."

    company_row = company_match.iloc[0]

    # -----------------------------
    # Company-side signal extraction
    # -----------------------------
    hotspot_cols = find_columns_by_keywords(
        df_company,
        ["hotspot", "commodity", "basin", "region", "country", "operation", "sourcing", "exposure"],
        exclude_keywords=["company", "name", "summary"]
    )

    risk_cols = find_columns_by_keywords(
        df_company,
        ["risk", "water", "drought", "flood", "scarcity", "biodiversity", "pollution", "heat"],
        exclude_keywords=["score"]
    )

    sbtn_cols = find_columns_by_keywords(
        df_company,
        ["sbtn", "target", "freshwater", "land", "ocean", "nature"]
    )

    business_cols = find_columns_by_keywords(
        df_company,
        ["business", "priority", "brand", "supply", "resilience", "reputation", "compliance", "cost"]
    )

    company_hotspots = collect_row_tokens(company_row, hotspot_cols)
    company_risks = collect_row_tokens(company_row, risk_cols)
    company_sbtn = collect_row_tokens(company_row, sbtn_cols)
    company_business = collect_row_tokens(company_row, business_cols)

    # keep geography separately for stronger matching
    geography_cols = find_columns_by_keywords(
        df_company,
        ["region", "country", "basin", "geograph", "location"],
        exclude_keywords=["company"]
    )
    company_geographies = collect_row_tokens(company_row, geography_cols)

    # -----------------------------
    # Project-side signal extraction
    # -----------------------------
    project_name_col = find_first_existing(df_projects, ["project_name", "name"])
    region_col = find_first_existing(df_projects, ["region", "country", "location"])
    ecosystem_col = find_first_existing(df_projects, ["ecosystem", "ecosystem_type"])
    intervention_col = find_first_existing(df_projects, ["intervention_type", "intervention", "nbs_type"])
    summary_col = find_first_existing(df_projects, ["project_summary", "summary", "description"])
    project_id_col = find_first_existing(df_projects, ["project_id", "id"])

    scored_rows = []

    for _, proj in df_projects.iterrows():
        project_hotspot_cols = [c for c in df_projects.columns if any(k in c.lower() for k in [
            "hotspot", "commodity", "basin", "region", "country", "location", "ecosystem"
        ])]
        project_risk_cols = [c for c in df_projects.columns if any(k in c.lower() for k in [
            "risk", "water", "drought", "flood", "scarcity", "biodiversity", "pollution", "heat"
        ])]
        project_sbtn_cols = [c for c in df_projects.columns if any(k in c.lower() for k in [
            "sbtn", "target", "freshwater", "land", "ocean", "nature"
        ])]
        project_business_cols = [c for c in df_projects.columns if any(k in c.lower() for k in [
            "business", "priority", "brand", "supply", "resilience", "reputation", "compliance", "cost"
        ])]

        proj_hotspots = collect_row_tokens(proj, project_hotspot_cols)
        proj_risks = collect_row_tokens(proj, project_risk_cols)
        proj_sbtn = collect_row_tokens(proj, project_sbtn_cols)
        proj_business = collect_row_tokens(proj, project_business_cols)

        hotspot_score, hotspot_overlap = overlap_score(company_hotspots, proj_hotspots, weight_per_match=3.0, max_matches=4)
        risk_score, risk_overlap = overlap_score(company_risks, proj_risks, weight_per_match=2.8, max_matches=4)
        sbtn_score, sbtn_overlap = overlap_score(company_sbtn, proj_sbtn, weight_per_match=2.2, max_matches=3)
        business_score, business_overlap = overlap_score(company_business, proj_business, weight_per_match=1.8, max_matches=3)

        # Geography bonus
        proj_geo_tokens = set()
        if region_col and pd.notna(proj.get(region_col, None)):
            proj_geo_tokens |= split_to_set(proj[region_col])
        geography_overlap = company_geographies.intersection(proj_geo_tokens)
        geography_bonus = 3.5 if len(geography_overlap) > 0 else 0.0

        # Intervention bonus: reward named interventions slightly so all-zero ties are less likely
        intervention_bonus = 1.0 if intervention_col and pd.notna(proj.get(intervention_col, None)) else 0.0

        raw_score = (
            hotspot_score +
            risk_score +
            sbtn_score +
            business_score +
            geography_bonus +
            intervention_bonus
        )

        evidence_count = (
            len(hotspot_overlap) +
            len(risk_overlap) +
            len(sbtn_overlap) +
            len(business_overlap) +
            len(geography_overlap)
        )

        scored_rows.append({
            "project_id": proj[project_id_col] if project_id_col else "",
            "project_name": proj[project_name_col] if project_name_col else "Unknown project",
            "region": proj[region_col] if region_col else "N/A",
            "ecosystem": proj[ecosystem_col] if ecosystem_col else "N/A",
            "intervention_type": proj[intervention_col] if intervention_col else "N/A",
            "project_summary": proj[summary_col] if summary_col else "",
            "hotspot_overlap": hotspot_overlap,
            "risk_overlap": risk_overlap,
            "sbtn_overlap": sbtn_overlap,
            "business_overlap": business_overlap,
            "geography_overlap": geography_overlap,
            "hotspot_count": len(hotspot_overlap),
            "risk_count": len(risk_overlap),
            "sbtn_count": len(sbtn_overlap),
            "business_count": len(business_overlap),
            "evidence_count": evidence_count,
            "raw_score": raw_score
        })

    result_df = pd.DataFrame(scored_rows)

    if result_df.empty:
        return None, "No NbS projects found in nbs_projects.csv."

    # -----------------------------
    # More realistic fit score
    # -----------------------------
    max_score = result_df["raw_score"].max()
    min_score = result_df["raw_score"].min()

    if max_score == min_score:
        # no strong differentiation: still provide a usable but conservative range
        result_df = result_df.sort_values(
            by=["evidence_count", "project_name"],
            ascending=[False, True]
        ).reset_index(drop=True)
        n = len(result_df)
        if n == 1:
            result_df["fit_score"] = 78
        else:
            result_df["fit_score"] = [
                int(round(78 - (i * 8 / max(n - 1, 1))))
                for i in range(n)
            ]
    else:
        score_ratio = (result_df["raw_score"] - min_score) / (max_score - min_score)
        evidence_bonus = result_df["evidence_count"].clip(upper=6) / 6.0  # 0 to 1
        result_df["fit_score"] = (
            76 + (score_ratio * 16) + (evidence_bonus * 6)
        ).round(0).astype(int)

        result_df = result_df.sort_values(
            by=["raw_score", "evidence_count", "project_name"],
            ascending=[False, False, True]
        ).reset_index(drop=True)

    top = result_df.iloc[0].copy()

    reasons = []

    if len(top["hotspot_overlap"]) > 0:
        reasons.append(
            f"it aligns with {company_name}'s hotspot profile around {', '.join(sorted(list(top['hotspot_overlap']))[:2])}"
        )
    if len(top["risk_overlap"]) > 0:
        reasons.append(
            f"it addresses risk exposure linked to {', '.join(sorted(list(top['risk_overlap']))[:2])}"
        )
    if len(top["sbtn_overlap"]) > 0:
        reasons.append(
            f"it supports sustainability priorities including {', '.join(sorted(list(top['sbtn_overlap']))[:2])}"
        )
    if len(top["business_overlap"]) > 0:
        reasons.append(
            f"it also connects to business priorities such as {', '.join(sorted(list(top['business_overlap']))[:2])}"
        )
    if len(top["geography_overlap"]) > 0:
        reasons.append(
            f"it is geographically relevant to {', '.join(sorted(list(top['geography_overlap']))[:2])}"
        )

    if reasons:
        why_text = f"This project is recommended for {company_name} because " + "; ".join(reasons[:3]) + "."
    else:
        why_text = (
            f"This project is recommended for {company_name} because it shows the strongest overall fit "
            f"across the available hotspot, risk, sustainability, geography, and business-priority criteria."
        )

    rec = {
        "company_name": company_name,
        "project_id": top["project_id"],
        "project_name": top["project_name"],
        "region": top["region"],
        "ecosystem": top["ecosystem"],
        "intervention_type": top["intervention_type"],
        "fit_score": int(top["fit_score"]),
        "why_text": why_text,
        "project_summary": top["project_summary"],
        "hotspot_overlap": top["hotspot_overlap"],
        "risk_overlap": top["risk_overlap"],
        "sbtn_overlap": top["sbtn_overlap"],
        "business_overlap": top["business_overlap"],
        "geography_overlap": top["geography_overlap"],
        "full_ranking": result_df
    }

    rec["finance_info"] = get_financial_view(rec, df_finance)

    return rec, None

# =========================================================
# EXPLANATION TEXTS
# =========================================================
def generate_hotspot_text(rec):
    overlaps = sorted(list(rec["hotspot_overlap"]))
    region = rec.get("region", "the relevant geography")
    project_name = rec.get("project_name", "this project")
    ecosystem = rec.get("ecosystem", "the local ecosystem")
    intervention_type = rec.get("intervention_type", "the selected intervention").replace("_", " ")

    if overlaps:
        hotspot_text = ", ".join(overlaps[:3])

        return (
            f"Hotspot identified: {rec['company_name']} shows exposure linked to {hotspot_text}, "
            f"with relevance to {region}. \n\n"
            f"Why this project fits: {project_name} focuses on {intervention_type} in a {ecosystem} context, "
            f"which makes it a strong match for that hotspot profile. \n\n"
            f"Business relevance: this helps connect NbS investment to concrete operational or sourcing pressure points, "
            f"rather than presenting sustainability action in isolation."
        )

    return (
        f"Hotspot identified: no strong explicit hotspot tag match is visible in the current dataset. \n\n"
        f"Why this project still fits: it remains the best overall recommendation based on the broader scoring model. \n\n"
        f"Business relevance: this suggests the project is strategically relevant even where hotspot tagging is still incomplete."
    )

def generate_risk_text(rec):
    overlaps = list(rec["risk_overlap"])
    if overlaps:
        return (
            f"This project helps address nature-related risk exposure for {rec['company_name']}, "
            f"especially around {', '.join(overlaps[:3])}. "
            f"In the MVP logic, projects that reduce exposure to these risks receive a stronger ranking."
        )
    return (
        "This project was not selected only because of direct risk-tag overlap, "
        "but because it performs best across the broader recommendation criteria."
    )

def generate_sbtn_text(rec):
    overlaps = list(rec["sbtn_overlap"])
    if overlaps:
        return (
            f"This recommendation supports sustainability or SBTN-related priorities relevant to {rec['company_name']}, "
            f"including {', '.join(overlaps[:3])}. "
            f"For the MVP, this is treated as alignment with nature-related targets rather than formal target validation."
        )
    return (
        "The current dataset shows limited direct SBTN-tag overlap for this recommendation, "
        "so the fit is driven more by hotspot, risk, and business relevance."
    )

def generate_business_text(rec):
    overlaps = list(rec["business_overlap"])
    if overlaps:
        return (
            f"Beyond sustainability, this project can also support business priorities for {rec['company_name']}, "
            f"such as {', '.join(overlaps[:3])}. "
            f"This helps frame NbS investment as a business-relevant decision, not only a sustainability action."
        )
    return (
        "The business-priority connection is less explicit in the current dataset, "
        "but the project still stands out because of its stronger overall strategic fit."
    )

# =========================================================
# MAIN APP
# =========================================================
df_company, df_projects, df_finance = load_data()

st.markdown('<div class="main-title">Nature Intelligence Prototype</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A simple MVP to help a company identify which Nature-based Solution (NbS) project it should invest in.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# DATA CHECK
# ---------------------------------------------------------
if df_company is None or df_projects is None:
    st.error(
        "Missing input files. Please place these files in the same folder as app.py:\n\n"
        "- company_exposure.csv\n"
        "- nbs_projects.csv"
    )

    st.markdown("### Example structure for `company_exposure.csv`")
    st.code("""company_name,hotspots,risk_exposure,sbtn_targets,business_priorities,company_summary
Adidas,water stress;cotton;asia,water risk;biodiversity loss,freshwater;land,brand resilience;supply security,Global sportswear company with nature-related supply chain exposure
""")

    st.markdown("### Example structure for `nbs_projects.csv`")
    st.code("""project_name,region,ecosystem,intervention_type,project_summary,hotspot_tags,risk_tags,sbtn_tags,business_tags
Watershed Restoration Project,Asia,Freshwater,Restoration,Restores watersheds and improves ecosystem resilience,water stress;asia,water risk,freshwater,brand resilience;supply security
Agroforestry Landscape Program,Latin America,Forest,Agroforestry,Supports regenerative landscapes and biodiversity,cotton;land use,biodiversity loss,land,supply security
""")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Company Selection")

company_col = find_first_existing(df_company, ["company_name", "company", "name"])
company_list = sorted(df_company[company_col].dropna().astype(str).unique().tolist())

default_company = "Adidas" if "Adidas" in company_list else company_list[0]
selected_company = st.sidebar.selectbox(
    "Select company",
    options=company_list,
    index=company_list.index(default_company),
    key="selected_company"
)

if "last_company" not in st.session_state:
    st.session_state["last_company"] = selected_company

if st.session_state["last_company"] != selected_company:
    st.session_state["last_company"] = selected_company
    st.session_state.pop("recommendation", None)
    st.session_state.pop("explain_section", None)

show_top3 = st.sidebar.checkbox("Show top 3 projects", value=False)
show_debug = st.sidebar.checkbox("Show scoring table", value=False)

# ---------------------------------------------------------
# LANDING PAGE CONTENT
# ---------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)

    if st.button(
        "Which NbS project should I invest in?",
        type="primary",
        use_container_width=True
    ):
        recommendation, error = build_recommendation(
            selected_company,
            df_company,
            df_projects,
            df_finance
        )

        if error:
            st.error(error)
        else:
            st.session_state["recommendation"] = recommendation

with right:
    st.markdown("## What this MVP does")
    st.write(
        "- identifies one best-fit NbS project\n"
        "- explains why it is relevant for the selected company\n"
        "- connects the recommendation to hotspots, risks, targets, and business priorities"
    )

# ---------------------------------------------------------
# OUTPUT CARD
# ---------------------------------------------------------
if "recommendation" in st.session_state:
    rec = st.session_state["recommendation"]

    st.markdown("## Recommendation")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.markdown(f"### Recommended project: **{rec['project_name']}**")
        st.write(rec["why_text"])

        if rec["project_summary"]:
            st.markdown('<div class="small-note">Project context</div>', unsafe_allow_html=True)
            st.write(rec["project_summary"])

    with col2:
        st.metric("Fit score", f"{rec['fit_score']} / 100")
        st.write(f"**Region:** {rec.get('region') if rec.get('region') not in ['', None] else 'N/A'}")
        st.write(f"**Ecosystem:** {rec.get('ecosystem') if rec.get('ecosystem') not in ['', None] else 'N/A'}")
        st.write(f"**Intervention type:** {rec.get('intervention_type') if rec.get('intervention_type') not in ['', None] else 'N/A'}")

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # INDICATIVE INVESTMENT
    # -----------------------------------------------------
    st.markdown("## Indicative investment")

    finance = rec.get("finance_info", None)

    if finance is not None:
        f1, f2, f3, f4 = st.columns(4)

        with f1:
            st.metric("Estimated project cost", finance.get("total_cost", "N/A"))

        with f2:
            st.metric("Suggested company contribution", finance.get("company_contribution", "N/A"))

        with f3:
            horizon_val = finance.get("time_horizon", "N/A")
            st.metric("Time horizon", f"{horizon_val} years" if str(horizon_val) != "N/A" else "N/A")

        with f4:
            st.metric("Financial effort", finance.get("effort", "N/A"))

        st.caption(
            "Suggested company contribution, time horizon, and financial effort are indicative values derived from estimated project cost using MVP assumptions."
        )
    else:
        st.info(
            "No matched finance row was found for this project. "
            "Add project-level or intervention-level cost data in project_financial_output.csv to activate the quantitative view."
        )

    # Optional top 3
    if show_top3:
        st.markdown("### Top ranked projects")
        top3 = rec["full_ranking"][[
            "project_name", "region", "ecosystem", "intervention_type", "fit_score"
        ]].head(3)
        st.dataframe(top3, use_container_width=True)

    # -----------------------------------------------------
    # EXPLANATION BUTTONS
    # -----------------------------------------------------
    st.markdown("## Why this project?")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Hotspots", use_container_width=True):
            st.session_state["explain_section"] = "hotspots"

    with c2:
        if st.button("Risk exposure", use_container_width=True):
            st.session_state["explain_section"] = "risk"

    with c3:
        if st.button("SBTN / sustainability fit", use_container_width=True):
            st.session_state["explain_section"] = "sbtn"

    with c4:
        if st.button("Business priorities", use_container_width=True):
            st.session_state["explain_section"] = "business"

    if "explain_section" in st.session_state:
        section = st.session_state["explain_section"]

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if section == "hotspots":
            st.markdown("### Hotspots")
            st.write(generate_hotspot_text(rec))

        elif section == "risk":
            st.markdown("### Risk exposure")
            st.write(generate_risk_text(rec))

        elif section == "sbtn":
            st.markdown("### SBTN / sustainability fit")
            st.write(generate_sbtn_text(rec))

        elif section == "business":
            st.markdown("### Business priorities")
            st.write(generate_business_text(rec))

        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # DEBUG / SCORING TABLE
    # -----------------------------------------------------
    if show_debug:
        st.markdown("## Scoring logic table")
        debug_df = rec["full_ranking"][[
            "project_name",
            "hotspot_count",
            "risk_count",
            "sbtn_count",
            "business_count",
            "raw_score",
            "fit_score"
        ]]
        st.dataframe(debug_df, use_container_width=True)