"""
Indonesia Climate Hazard Assessment Dashboard
Baseline 2000-2020 → Projected 2020-2040
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indonesia Climate Hazard Assessment",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DELTAS = {
    "ssp126": {"temperature": 0.5,  "precipitation": 1.5,  "soil_wetness": -0.012, "slr_cm": 5.5},
    "ssp245": {"temperature": 0.7,  "precipitation": 2.0,  "soil_wetness": -0.018, "slr_cm": 7.0},
    "ssp585": {"temperature": 0.9,  "precipitation": 3.0,  "soil_wetness": -0.025, "slr_cm": 9.0},
}

SCENARIO_LABELS = {
    "ssp126": "SSP1-2.6 — Low (Strong mitigation)",
    "ssp245": "SSP2-4.5 — Medium (Moderate mitigation)",
    "ssp585": "SSP5-8.5 — High (Business as Usual)",
}

RISK_COLORS = {"low": "#16a34a", "moderate": "#d97706", "high": "#dc2626", "critical": "#7c3aed"}

# ── Scoring (mirrors app/scoring.py) ─────────────────────────────────────────
def _stepped(v, bp):
    for upper, score, level in bp:
        if v < upper:
            return score, level
    return bp[-1][1], bp[-1][2]

def score_temperature(d):
    return _stepped(d, [(0.5,15,"low"),(1.0,35,"moderate"),(1.5,55,"moderate"),(2.0,75,"high"),(float("inf"),95,"critical")])

def score_precipitation(d):
    return _stepped(abs(d), [(2,10,"low"),(4,30,"moderate"),(7,55,"moderate"),(11,75,"high"),(float("inf"),95,"critical")])

def score_drought(d):
    return _stepped(-d, [(0.01,10,"low"),(0.02,30,"moderate"),(0.04,55,"moderate"),(0.06,75,"high"),(float("inf"),95,"critical")])

def score_sea_level(d):
    return _stepped(d, [(5,15,"low"),(8,35,"moderate"),(12,60,"moderate"),(16,80,"high"),(float("inf"),95,"critical")])

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_baselines() -> dict:
    p = Path("data/baselines.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text())

@st.cache_data
def load_cities() -> pd.DataFrame:
    p = Path("data/cities.csv")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def build_assessment_df(baselines: dict, scenario: str) -> pd.DataFrame:
    """Compute risk scores for all cities in the fixture."""
    d = DELTAS[scenario]
    rows = []
    cities_df = load_cities()
    city_meta = {f"{r.lat:.2f},{r.lon:.2f}": r for r in cities_df.itertuples()} if not cities_df.empty else {}

    for loc_key, loc in baselines.get("locations", {}).items():
        meta = city_meta.get(loc_key)
        province = meta.province if meta else "Unknown"

        t_score, t_level = score_temperature(d["temperature"])
        p_score, p_level = score_precipitation(d["precipitation"])
        w_score, w_level = score_drought(d["soil_wetness"])
        s_score, s_level = score_sea_level(d["slr_cm"])
        composite = round((t_score + p_score + w_score + s_score) / 4)

        rows.append({
            "city":              loc["city"],
            "province":          province,
            "lat":               float(loc_key.split(",")[0]),
            "lon":               float(loc_key.split(",")[1]),
            "baseline_temp_c":   loc["T2M_annual_mean"],
            "future_temp_c":     round(loc["T2M_annual_mean"] + d["temperature"], 2),
            "delta_temp_c":      d["temperature"],
            "temp_score":        t_score,
            "temp_level":        t_level,
            "baseline_precip":   loc["PRECTOTCORR_p99"],
            "delta_precip_pct":  d["precipitation"],
            "precip_score":      p_score,
            "precip_level":      p_level,
            "baseline_wetness":  loc["GWETROOT_dry_season"],
            "delta_wetness":     d["soil_wetness"],
            "drought_score":     w_score,
            "drought_level":     w_level,
            "slr_delta_cm":      d["slr_cm"],
            "slr_score":         s_score,
            "slr_level":         s_level,
            "composite_score":   composite,
            "composite_level":   "low" if composite < 25 else "moderate" if composite < 50 else "high" if composite < 75 else "critical",
        })
    return pd.DataFrame(rows).sort_values("composite_score", ascending=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Indonesia Climate Hazard")
    st.caption("2000–2020 → 2020–2040")
    st.divider()

    scenario = st.selectbox("Emissions scenario", list(SCENARIO_LABELS.keys()),
                            index=1, format_func=lambda x: SCENARIO_LABELS[x])

    baselines = load_baselines()
    cities_df = load_cities()

    if baselines:
        df = build_assessment_df(baselines, scenario)
        provinces = sorted(df["province"].dropna().unique())
        selected_province = st.selectbox("Province", ["All provinces"] + provinces)
        if selected_province != "All provinces":
            df_filtered = df[df["province"] == selected_province]
        else:
            df_filtered = df

        city_names = df_filtered["city"].tolist()
        selected_city = st.selectbox("City / Regency", city_names)
        city_row = df_filtered[df_filtered["city"] == selected_city].iloc[0]
    else:
        st.warning("No baseline data found. Run `scripts/fetch_baselines.py` or trigger the GitHub Actions workflow.")
        st.stop()

    st.divider()
    ts = baselines.get("metadata", {}).get("fetched_at", "seeded")
    if ts == "seeded":
        st.warning("⚠ Using seeded values — trigger GitHub Actions to refresh with real climate data.")
    else:
        st.caption(f"Data refreshed: {ts[:10]}")
    st.caption(f"{len(df)} cities loaded /n Created by Ainur Ridho")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title(f"{selected_city} - Climate Hazard")
st.caption(f"{city_row['province']} · Baseline 2000–2020 → Projected 2020–2040 · {SCENARIO_LABELS[scenario]}")

# ── Metric cards ──────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

def metric_card(col, label, baseline, future, delta, score, level):
    color = RISK_COLORS[level]
    col.metric(label, f"{future}", f"{delta:+}", help=f"Baseline: {baseline}")
    col.markdown(f"<div style='background:{color};color:white;padding:2px 8px;border-radius:4px;"
                 f"font-size:0.8rem;font-weight:600;display:inline-block'>Score {score} · {level}</div>",
                 unsafe_allow_html=True)

metric_card(col1, "Surface Temperature",
            f"{city_row['baseline_temp_c']:.1f}°C",
            f"{city_row['future_temp_c']:.1f}°C",
            city_row['delta_temp_c'],
            city_row['temp_score'], city_row['temp_level'])

metric_card(col2, "Rainfall (p99)",
            f"{city_row['baseline_precip']:.1f} mm/d",
            f"{city_row['baseline_precip']*(1+city_row['delta_precip_pct']/100):.1f} mm/d",
            city_row['delta_precip_pct'],
            city_row['precip_score'], city_row['precip_level'])

metric_card(col3, "Drought",
            f"{city_row['baseline_wetness']:.3f}",
            f"{city_row['baseline_wetness']+city_row['delta_wetness']:.3f}",
            city_row['delta_wetness'],
            city_row['drought_score'], city_row['drought_level'])

col4.metric("Sea Level Rise", f"+{city_row['slr_delta_cm']:.1f} cm", "by 2040")
col4.markdown(f"<div style='background:{RISK_COLORS[city_row['slr_level']]};color:white;"
              f"padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:600;"
              f"display:inline-block'>Score {city_row['slr_score']} · {city_row['slr_level']}</div>",
              unsafe_allow_html=True)

comp_color = RISK_COLORS[city_row['composite_level']]
col5.metric("Composite Risk", f"{city_row['composite_score']}/100")
col5.markdown(f"<div style='background:{comp_color};color:white;padding:2px 8px;"
              f"border-radius:4px;font-size:0.8rem;font-weight:600;display:inline-block'>"
              f"{city_row['composite_level'].title()}</div>", unsafe_allow_html=True)

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Map", "Province Comparison", "Table (All Cities)"])

with tab1:
    fig_map = px.scatter_mapbox(
        df, lat="lat", lon="lon", color="composite_score",
        color_continuous_scale=["#16a34a", "#d97706", "#dc2626"],
        range_color=[0, 100],
        size_max=14, zoom=4, center={"lat": -2.5, "lon": 118.0},
        hover_name="city",
        hover_data={"province": True, "composite_score": True,
                    "temp_score": True, "precip_score": True,
                    "drought_score": True, "slr_score": True,
                    "lat": False, "lon": False},
        labels={"composite_score": "Composite Hazard"},
        mapbox_style="carto-positron",
        height=520,
    )
    fig_map.update_layout(margin={"r": 0, "l": 0, "b": 0, "t": 0},
                          coloraxis_colorbar=dict(title="Hazard Score"))
    st.plotly_chart(fig_map, width="stretch")

with tab2:
    prov_df = df.groupby("province").agg(
        cities=("city", "count"),
        avg_composite=("composite_score", "mean"),
        avg_temp_score=("temp_score", "mean"),
        avg_precip_score=("precip_score", "mean"),
        avg_drought_score=("drought_score", "mean"),
        avg_slr_score=("slr_score", "mean"),
    ).round(1).reset_index().sort_values("avg_composite", ascending=True)

    fig_prov = go.Figure()
    hazards = [
        ("avg_temp_score",    "Temperature", "#ef4444"),
        ("avg_precip_score",  "Precipitation", "#3b82f6"),
        ("avg_drought_score", "Drought", "#f59e0b"),
        ("avg_slr_score",     "Sea Level", "#06b6d4"),
    ]
    for col, label, color in hazards:
        fig_prov.add_trace(go.Bar(
            y=prov_df["province"], x=prov_df[col],
            name=label, orientation="h", marker_color=color,
        ))
    fig_prov.update_layout(
        barmode="group", height=600,
        xaxis_title="Average Risk Score", yaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin={"l": 160, "r": 20, "t": 40, "b": 40},
    )
    st.plotly_chart(fig_prov, width="stretch")

with tab3:
    display_cols = {
        "city": "City", "province": "Province",
        "baseline_temp_c": "Baseline T (°C)", "delta_temp_c": "ΔT (°C)",
        "temp_score": "Temp Score", "precip_score": "Precip Score",
        "drought_score": "Drought Score", "slr_score": "SLR Score",
        "composite_score": "Composite",
    }
    display_df = df[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Composite": st.column_config.ProgressColumn(
                "Composite", min_value=0, max_value=100
            ),
        },
    )
    st.download_button(
        "⬇ Download CSV Table", display_df.to_csv(index=False),
        file_name=f"indonesia_climate_hazard_{scenario}.csv", mime="text/csv",
    )
