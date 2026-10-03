"""
⚡ Household Energy Analytics - Streamlit Dashboard
====================================================
Interactive dashboard for analysing household electricity consumption.

Professional dark-themed UI built with Streamlit + custom CSS.

Run:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import analysis

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="⚡ Household Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------
# Single source of truth for the dashboard's visual identity.
ACCENT = "#f5b942"        # amber — energy
ACCENT_SOFT = "#f5b94220"
BG = "#0e1117"            # Streamlit dark background
CARD_BG = "#161b24"
CARD_BORDER = "#232a38"
TEXT = "#e8eaed"
TEXT_MUTED = "#9aa3b2"
POS = "#4ade80"           # green
NEG = "#f87171"           # red
BLUE = "#60a5fa"
PURPLE = "#a78bfa"

PLOTLY_TEMPLATE = "plotly_dark"

CHART_COLORS = ["#f5b942", "#60a5fa", "#4ade80", "#a78bfa", "#f87171", "#fb923c"]


def apply_custom_css():
    """Inject the dashboard's custom stylesheet."""
    st.markdown(
        """
        <style>
        /* ---------- Base ---------- */
        .stApp { background-color: #0e1117; }
        #MainMenu, footer { visibility: hidden; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #131722 0%, #0e1117 100%);
            border-right: 1px solid #232a38;
        }

        /* ---------- Header banner ---------- */
        .hero {
            background: linear-gradient(135deg, #1a2233 0%, #141a28 55%, #1c1f14 100%);
            border: 1px solid #2a3348;
            border-radius: 16px;
            padding: 28px 34px;
            margin-bottom: 8px;
            position: relative;
            overflow: hidden;
        }
        .hero::before {
            content: "";
            position: absolute; top: -60%; right: -5%;
            width: 420px; height: 420px;
            background: radial-gradient(circle, #f5b94214 0%, transparent 70%);
            pointer-events: none;
        }
        .hero-title {
            font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px;
            color: #e8eaed; margin: 0; line-height: 1.2;
        }
        .hero-title .bolt { color: #f5b942; }
        .hero-sub {
            font-size: 0.95rem; color: #9aa3b2; margin-top: 8px; font-weight: 400;
        }
        .hero-badge {
            display: inline-block; margin-top: 14px;
            background: #f5b94216; color: #f5b942;
            border: 1px solid #f5b94240; border-radius: 999px;
            padding: 4px 14px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.4px;
        }

        /* ---------- KPI cards ---------- */
        div[data-testid="stMetric"] {
            background: #161b24;
            border: 1px solid #232a38;
            border-radius: 14px;
            padding: 18px 20px 14px 20px;
        }
        div[data-testid="stMetric"]:hover { border-color: #f5b94266; transition: 0.2s; }
        div[data-testid="stMetric"] label p {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            letter-spacing: 1.2px !important;
            text-transform: uppercase !important;
            color: #9aa3b2 !important;
        }
        div[data-testid="stMetric"] value {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
            color: #f5b942 !important;
        }

        /* ---------- Section headers ---------- */
        .section-header {
            display: flex; align-items: center; gap: 12px;
            margin: 26px 0 4px 0;
        }
        .section-header .bar {
            width: 5px; height: 26px; border-radius: 3px;
            background: linear-gradient(180deg, #f5b942, #f5b94260);
        }
        .section-header h2 {
            font-size: 1.25rem; font-weight: 700; color: #e8eaed;
            margin: 0; letter-spacing: -0.2px;
        }
        .section-note { color: #9aa3b2; font-size: 0.88rem; margin: 2px 0 14px 17px; }

        /* ---------- Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px; background: #131722; padding: 6px;
            border-radius: 14px; border: 1px solid #232a38;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 9px; padding: 8px 18px;
            font-size: 0.88rem; font-weight: 600; color: #9aa3b2;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background: #f5b94214 !important; color: #f5b942 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; }
        .stTabs [data-baseweb="tab-border"] { display: none; }

        /* ---------- Dataframes / tables ---------- */
        [data-testid="stDataFrame"] { border: 1px solid #232a38; border-radius: 12px; }

        /* ---------- Info / caption blocks ---------- */
        .note-box {
            background: #f5b9420d; border: 1px solid #f5b94233;
            border-left: 4px solid #f5b942;
            border-radius: 10px; padding: 12px 18px;
            color: #c9ced8; font-size: 0.88rem; margin: 10px 0;
        }
        .note-box b { color: #f5b942; }

        .stat-table {
            width: 100%; border-collapse: collapse; font-size: 0.9rem;
        }
        .stat-table td {
            padding: 9px 14px; border-bottom: 1px solid #232a38;
        }
        .stat-table td:first-child { color: #9aa3b2; }
        .stat-table td:last-child {
            text-align: right; color: #e8eaed; font-weight: 700;
            font-variant-numeric: tabular-nums;
        }
        .stat-table tr:last-child td { border-bottom: none; }
        .stat-card {
            background: #161b24; border: 1px solid #232a38;
            border-radius: 14px; padding: 8px 10px;
        }
        .stat-card h4 {
            color: #f5b942; font-size: 0.75rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1.1px;
            margin: 8px 14px 4px 14px;
        }

        /* ---------- Insight cards ---------- */
        .insight-item {
            display: flex; gap: 14px; align-items: flex-start;
            background: #161b24; border: 1px solid #232a38;
            border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;
        }
        .insight-num {
            min-width: 30px; height: 30px; border-radius: 8px;
            background: #f5b94214; color: #f5b942;
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 0.85rem;
            border: 1px solid #f5b94240; margin-top: 2px;
        }
        .insight-text { color: #c9ced8; font-size: 0.95rem; line-height: 1.55; }
        .insight-text b { color: #f5b942; }

        /* ---------- Sidebar polish ---------- */
        section[data-testid="stSidebar"] h1 { font-size: 1.1rem; }
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%; background: #f5b94214; color: #f5b942;
            border: 1px solid #f5b94240; border-radius: 10px; font-weight: 700;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: #f5b9422a; border-color: #f5b94280;
        }
        .sidebar-brand {
            display: flex; align-items: center; gap: 10px;
            padding: 4px 2px 14px 2px; border-bottom: 1px solid #232a38;
            margin-bottom: 14px;
        }
        .sidebar-brand .logo {
            width: 38px; height: 38px; border-radius: 10px;
            background: linear-gradient(135deg, #f5b942, #d97706);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem;
        }
        .sidebar-brand .name { color: #e8eaed; font-weight: 800; font-size: 0.98rem; }
        .sidebar-brand .tagline { color: #9aa3b2; font-size: 0.72rem; }

        /* ---------- Footer ---------- */
        .footer {
            border-top: 1px solid #232a38; margin-top: 34px; padding-top: 16px;
            text-align: center; color: #6b7280; font-size: 0.8rem;
        }

        /* ---------- Charts spacing ---------- */
        .stPlotlyChart { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle, badge):
    """Render the gradient header banner."""
    st.markdown(
        f"""
        <div class="hero">
            <h1 class="hero-title"><span class="bolt">⚡</span> {title}</h1>
            <div class="hero-sub">{subtitle}</div>
            <span class="hero-badge">{badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(icon, title, note=None):
    """Render a styled section header with an accent bar."""
    st.markdown(
        f"""
        <div class="section-header">
            <div class="bar"></div>
            <h2>{icon}&ensp;{title}</h2>
        </div>
        {"<div class='section-note'>" + note + "</div>" if note else ""}
        """,
        unsafe_allow_html=True,
    )


def note_box(html):
    st.markdown(f'<div class="note-box">{html}</div>', unsafe_allow_html=True)


def style_plotly_fig(fig, height=380):
    """Apply the dashboard's uniform Plotly styling."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=54, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family="Inter, Segoe UI, sans-serif", size=13),
        title=dict(font=dict(size=16, color=TEXT), x=0.01),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=CARD_BORDER, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor="#232a3840", zerolinecolor="#232a38")
    fig.update_yaxes(gridcolor="#232a3840", zerolinecolor="#232a38")
    return fig


apply_custom_css()

# ----------------------------------------------------------------------
# Data loading (cached so filters feel instant)
# ----------------------------------------------------------------------
# Resolve the dataset path from this file's location, not the process
# working directory. This makes the app work whether it is launched
# from the repo root (Streamlit Cloud) or from the project folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "electricity_consumption.csv")


@st.cache_data
def load_data_cached(path):
    return analysis.load_data(path)


if not st.session_state.get("data_loaded", False):
    try:
        df_full = load_data_cached(DATA_PATH)
        st.session_state["df_full"] = df_full
        st.session_state["data_loaded"] = True
    except FileNotFoundError:
        st.error(f"Dataset not found at: {DATA_PATH}")
        st.stop()
else:
    df_full = st.session_state["df_full"]

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo">⚡</div>
            <div>
                <div class="name">Energy Analytics</div>
                <div class="tagline">HOUSEHOLD CONSUMPTION DASHBOARD</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### ⚙️ Filters")

    # Reset button restores defaults
    if st.button("🔄 Reset Filters", use_container_width=True):
        st.session_state.pop("filters", None)
        st.rerun()

    # Date range
    date_min = df_full["Date"].min()
    date_max = df_full["Date"].max()
    date_range = st.date_input(
        "📅 Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    # Day type
    day_types = st.multiselect(
        "🗓️ Day Type",
        options=["Weekday", "Weekend"],
        default=["Weekday", "Weekend"],
    )

    # Months
    all_months = analysis.MONTH_ORDER
    months = st.multiselect(
        "📆 Month",
        options=all_months,
        default=[m for m in all_months if m in set(df_full["Month"])],
    )

    # Hour range
    hour_range = st.slider(
        "⏰ Hour Range",
        min_value=0, max_value=23,
        value=(0, 23),
    )

    # High-consumption detection method
    detect_method = st.radio(
        "🔍 Detection Method",
        options=["iqr", "std"],
        format_func=lambda m: "IQR (Q3 + 1.5×IQR)" if m == "iqr" else "Mean + 2×Std Dev",
        index=0,
    )

# ----------------------------------------------------------------------
# Apply filters
# ----------------------------------------------------------------------
if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_date, end_date = pd.Timestamp(date_range[0]), date_max

mask = (
    (df_full["Date"] >= start_date)
    & (df_full["Date"] <= end_date)
    & df_full["Day_Type"].isin(day_types)
    & df_full["Month"].isin(months)
    & df_full["Hour"].between(hour_range[0], hour_range[1])
)
df = df_full.loc[mask].copy()

if df.empty:
    st.warning("No records match the selected filters. Please widen the filter selection.")
    st.stop()

# ----------------------------------------------------------------------
# Header banner
# ----------------------------------------------------------------------
hero(
    "Household Energy Analytics",
    "Consumption patterns, peak-hour analysis and unusual-usage detection "
    "for a full year of household electricity readings.",
    f"{len(df_full):,} RECORDS · {len(df_full.columns)} VARIABLES · SIMULATED DATASET",
)

# ----------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------
df_flagged, hc_threshold, hc_method_desc = analysis.detect_high_consumption(df, method=detect_method)
n_high = int(df_flagged["High_Consumption"].sum())
stats = analysis.calculate_statistics(df)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("TOTAL ENERGY", f"{stats['total_kwh']:,.1f} kWh")
kpi2.metric("AVG CONSUMPTION", f"{stats['avg_kwh']:.2f} kWh")
kpi3.metric("PEAK HOUR", f"{stats['peak_hour']:02d}:00")
kpi4.metric("HIGH-CONSUMPTION", f"{n_high}")

st.caption(
    f"Showing {stats['n_records']:,} of {len(df_full):,} records · "
    f"Detection: {hc_method_desc} · Threshold {hc_threshold:.2f} kWh"
)
st.markdown("---")

# ----------------------------------------------------------------------
# Basic statistics
# ----------------------------------------------------------------------
section("📋", "Basic Statistics", "Descriptive statistics computed from the filtered dataset.")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div class="stat-card">
            <h4>⚡ Energy Statistics (kWh per record)</h4>
            <table class="stat-table">
                <tr><td>Total consumption</td><td>{stats['total_kwh']:,.2f} kWh</td></tr>
                <tr><td>Average</td><td>{stats['avg_kwh']:.3f}</td></tr>
                <tr><td>Median</td><td>{stats['median_kwh']:.3f}</td></tr>
                <tr><td>Minimum</td><td>{stats['min_kwh']:.3f}</td></tr>
                <tr><td>Maximum</td><td>{stats['max_kwh']:.3f}</td></tr>
                <tr><td>Std deviation</td><td>{stats['std_kwh']:.3f}</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="stat-card">
            <h4>🔌 Electrical &amp; Time Patterns</h4>
            <table class="stat-table">
                <tr><td>Average voltage</td><td>{stats['avg_voltage']:.1f} V</td></tr>
                <tr><td>Average current</td><td>{stats['avg_current']:.2f} A</td></tr>
                <tr><td>Peak consumption hour</td><td>{stats['peak_hour']:02d}:00</td></tr>
                <tr><td>Lowest consumption hour</td><td>{stats['lowest_hour']:02d}:00</td></tr>
                <tr><td>Weekday average</td><td>{stats['weekday_avg_kwh']:.3f} kWh</td></tr>
                <tr><td>Weekend average</td><td>{stats['weekend_avg_kwh']:.3f} kWh</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# Tabbed analysis sections
# ----------------------------------------------------------------------
tab_hourly, tab_daily, tab_compare, tab_voltage, tab_rooms, tab_heat, tab_high, tab_data = st.tabs([
    "⏰ Hourly", "📅 Daily", "⚖️ Weekday vs Weekend", "⚡ Voltage", "🏠 Rooms",
    "🔥 Day × Hour", "🚩 High Consumption", "🗂️ Data Explorer",
])

# ----------------------------------------------------------------------
# Tab 1 - Hourly consumption
# ----------------------------------------------------------------------
with tab_hourly:
    section("📈", "Hourly Consumption Pattern",
            "Average power for each hour of the day — identifies the household's peak electricity hours.")

    hourly = analysis.calculate_hourly_consumption(df)
    fig = px.line(hourly, x="Hour", y="Avg_Power_kW", markers=True,
                  labels={"Avg_Power_kW": "Avg Power (kW)"}).update_traces(
        line=dict(color=ACCENT, width=3), marker=dict(size=8, color=ACCENT,
                                                      line=dict(color=BG, width=2)))
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

    section("🏷️", "Peak-Hour Classification")
    note_box(
        "Hours are classified with <b>data-driven thresholds</b>, not manual judgement — "
        "<b>Peak</b>: avg ≥ 75th percentile of hourly averages · "
        "<b>Low</b>: avg ≤ 25th percentile · <b>Normal</b>: in between."
    )
    peak_df = analysis.calculate_peak_hours(df)
    color_map = {"Peak": NEG, "Normal": BLUE, "Low": POS}
    fig2 = px.bar(peak_df.sort_values("Hour"), x="Hour", y="Avg_Power_kW",
                  color="Classification", color_discrete_map=color_map)
    fig2.update_xaxes(dtick=1)
    st.plotly_chart(style_plotly_fig(fig2, height=360), use_container_width=True)

    st.dataframe(peak_df.round(3), use_container_width=True, height=320)

# ----------------------------------------------------------------------
# Tab 2 - Daily consumption
# ----------------------------------------------------------------------
with tab_daily:
    section("📈", "Daily Consumption Trend",
            "Total energy consumed per day over the selected period.")

    daily = analysis.calculate_daily_consumption(df)
    fig = px.area(daily, x="Date", y="Total_kWh").update_traces(
        line=dict(color=ACCENT, width=2),
        fillcolor=f"rgba(245, 185, 66, 0.12)")
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

    daily_flagged, day_threshold, day_method = analysis.detect_high_consumption_days(df)
    n_flagged_days = int(daily_flagged["High_Consumption"].sum())

    section("🚩", "Unusually High-Consumption Days")
    note_box(
        f"Method: <b>{day_method}</b>. Days above <b>{day_threshold:.2f} kWh</b> are flagged. "
        "These identify days worth investigating — <b>not confirmed wastage</b>."
    )
    fig = px.scatter(daily_flagged, x="Date", y="Total_kWh", color="High_Consumption",
                     color_discrete_map={True: NEG, False: BLUE}).update_traces(marker=dict(size=7))
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

    if n_flagged_days > 0:
        st.markdown(f"**{n_flagged_days} unusual day(s) detected:**")
        st.dataframe(
            daily_flagged.loc[daily_flagged["High_Consumption"],
                              ["Date", "Total_kWh", "Avg_kWh", "Max_kWh"]].round(3),
            use_container_width=True,
        )

# ----------------------------------------------------------------------
# Tab 3 - Weekday vs Weekend
# ----------------------------------------------------------------------
with tab_compare:
    section("⚖️", "Weekday vs Weekend Consumption",
            "How electricity usage differs between weekdays and weekends.")

    comparison = analysis.calculate_weekday_weekend(df)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(comparison, x="Day_Type", y="Avg_Energy_kWh", color="Day_Type",
                     color_discrete_map={"Weekday": BLUE, "Weekend": ACCENT})
        st.plotly_chart(style_plotly_fig(fig, height=340), use_container_width=True)
    with col2:
        fig = px.bar(comparison, x="Day_Type", y="Total_Energy_kWh", color="Day_Type",
                     color_discrete_map={"Weekday": BLUE, "Weekend": ACCENT})
        st.plotly_chart(style_plotly_fig(fig, height=340), use_container_width=True)

    section("🕐", "Hourly Profile")
    hourly_split = analysis.calculate_hourly_by_day_type(df)
    fig = go.Figure()
    for day_type, color in [("Weekday", BLUE), ("Weekend", ACCENT)]:
        if day_type in hourly_split.columns:
            fig.add_trace(go.Scatter(
                x=hourly_split["Hour"], y=hourly_split[day_type],
                mode="lines+markers", name=day_type,
                line=dict(color=color, width=3)))
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

    st.dataframe(comparison.round(3), use_container_width=True)

# ----------------------------------------------------------------------
# Tab 4 - Voltage vs Power
# ----------------------------------------------------------------------
with tab_voltage:
    section("⚡", "Voltage vs Power Consumption",
            "Scatter plot of supply voltage against power consumption.")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.scatter(df, x="Voltage_V", y="Power_Consumption_kW", opacity=0.3).update_traces(
            marker=dict(color=ACCENT, size=6))
        st.plotly_chart(style_plotly_fig(fig, height=420), use_container_width=True)
    with col2:
        corr = analysis.voltage_power_correlation(df)
        st.markdown(
            f"""
            <div class="stat-card">
                <h4>Correlation Analysis</h4>
                <table class="stat-table">
                    <tr><td>Pearson r (V vs P)</td><td>{corr:.3f}</td></tr>
                    <tr><td>Strength</td><td>{'Weak' if abs(corr) < 0.3 else 'Moderate' if abs(corr) < 0.7 else 'Strong'}</td></tr>
                    <tr><td>Direction</td><td>{'Negative' if corr < 0 else 'Positive'}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        note_box(
            "⚠️ <b>Correlation ≠ causation.</b> The coefficient only indicates the strength and "
            "direction of the association <b>in this dataset</b> — it does not prove that voltage "
            "changes cause consumption changes."
        )

    section("🔗", "Correlation Matrix")
    corr_matrix = analysis.calculate_correlations(df)
    fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(style_plotly_fig(fig, height=460), use_container_width=True)

    section("📊", "Consumption Distribution")
    fig = px.histogram(df, x="Power_Consumption_kW", nbins=50,
                       color_discrete_sequence=[ACCENT])
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

# ----------------------------------------------------------------------
# Tab 5 - Room-wise consumption
# ----------------------------------------------------------------------
with tab_rooms:
    section("🏠", "Room-wise Estimated Consumption",
            "Which usage category contributes most to estimated household consumption.")
    note_box(
        "These are <b>estimated usage categories</b> derived from the simulated dataset — "
        "not metered sub-circuit measurements."
    )

    rooms = analysis.calculate_room_consumption(df)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(rooms, x="Room", y="Total_kWh", color="Room",
                     color_discrete_sequence=CHART_COLORS)
        st.plotly_chart(style_plotly_fig(fig, height=360), use_container_width=True)
    with col2:
        fig = px.pie(rooms, names="Room", values="Total_kWh", hole=0.55,
                     color_discrete_sequence=CHART_COLORS)
        fig.update_traces(textinfo="percent", textfont_color=TEXT)
        st.plotly_chart(style_plotly_fig(fig, height=360), use_container_width=True)

    section("🕐", "Hourly Pattern per Room")
    hourly_rooms = analysis.calculate_hourly_room_consumption(df)
    fig = go.Figure()
    room_colors = [("Kitchen", NEG), ("Living_Room", BLUE), ("Bedroom", POS), ("Other", PURPLE)]
    for room, color in room_colors:
        fig.add_trace(go.Scatter(
            x=hourly_rooms["Hour"], y=hourly_rooms[room],
            mode="lines+markers", name=room.replace("_", " "),
            line=dict(color=color, width=3)))
    fig.update_layout(xaxis=dict(dtick=1))
    st.plotly_chart(style_plotly_fig(fig), use_container_width=True)

    st.dataframe(rooms.round(3), use_container_width=True)

# ----------------------------------------------------------------------
# Tab 6 - Day × Hour heatmap
# ----------------------------------------------------------------------
with tab_heat:
    section("🔥", "Day × Hour Consumption Heatmap",
            "Average power for every weekday × hour combination — the project's headline visual.")

    matrix = analysis.calculate_day_hour_matrix(df)
    fig = px.imshow(matrix, text_auto=".2f", aspect="auto",
                    color_continuous_scale="YlOrRd")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(style_plotly_fig(fig, height=480), use_container_width=True)

# ----------------------------------------------------------------------
# Tab 7 - Potential high-consumption periods
# ----------------------------------------------------------------------
with tab_high:
    section("🚩", "Potential High-Consumption Periods")
    note_box(
        "Records where consumption is unusually high compared with normal observations. "
        "These indicate <b>unusual patterns that may be worth investigating</b> — they are "
        "<b>not confirmed electricity wastage</b> (high usage can be legitimate: guests, "
        "seasonal appliance use, etc.)."
    )
    st.markdown(
        f"**Detection method:** {hc_method_desc} &nbsp;·&nbsp; "
        f"**Threshold:** {hc_threshold:.2f} kWh &nbsp;·&nbsp; "
        f"**Flagged:** {n_high} records ({n_high / len(df) * 100:.1f}% of filtered data)"
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_flagged, x="Date", y="Energy_Consumption_kWh",
                         color="High_Consumption",
                         color_discrete_map={True: NEG, False: BLUE}).update_traces(marker=dict(size=7))
        fig.add_hline(y=hc_threshold, line_dash="dash", line_color=ACCENT,
                      annotation_text=f"Threshold {hc_threshold:.2f} kWh",
                      annotation_font_color=ACCENT)
        st.plotly_chart(style_plotly_fig(fig, height=380), use_container_width=True)
    with col2:
        high_hours = (df_flagged.loc[df_flagged["High_Consumption"]]
                      .groupby("Hour")["Record_ID"].count().reset_index())
        if not high_hours.empty:
            fig = px.bar(high_hours, x="Hour", y="Record_ID",
                         color_discrete_sequence=[NEG])
            fig.update_xaxes(dtick=1)
            st.plotly_chart(style_plotly_fig(fig, height=380), use_container_width=True)
        else:
            st.info("No high-consumption records in the current selection.")

    high_df = df_flagged.loc[df_flagged["High_Consumption"]]
    if not high_df.empty:
        st.markdown(f"**{len(high_df)} flagged records** (first 200 shown)")
        st.dataframe(high_df[["Record_ID", "Date", "Time", "Day", "Day_Type",
                              "Hour", "Energy_Consumption_kWh"]].head(200).round(3),
                     use_container_width=True, height=350)

# ----------------------------------------------------------------------
# Tab 8 - Data explorer
# ----------------------------------------------------------------------
with tab_data:
    section("🗂️", "Dataset Explorer",
            f"Filtered dataset: {len(df):,} rows × {len(df.columns)} columns.")

    c1, c2, c3 = st.columns(3)
    c1.metric("MISSING VALUES", int(df.isnull().sum().sum()))
    c2.metric("DUPLICATE ROWS", int(df.duplicated().sum()))
    c3.metric("DUPLICATE IDs", int(df["Record_ID"].duplicated().sum()))

    section("📊", "Descriptive Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)

    section("👁️", "Data Preview")
    st.dataframe(df.head(500), use_container_width=True, height=420)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered dataset as CSV", csv,
                       "electricity_filtered.csv", "text/csv")

# ----------------------------------------------------------------------
# Key insights
# ----------------------------------------------------------------------
st.markdown("---")
section("💡", "Key Insights",
        "All values are computed dynamically from the currently filtered dataset.")

insights = analysis.generate_insights(df)
for i, text in enumerate(insights, start=1):
    st.markdown(
        f"""
        <div class="insight-item">
            <div class="insight-num">{i}</div>
            <div class="insight-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
        ⚡ Household Energy Analytics — a Data Analysis &amp; Visualization (DAV) project.<br>
        High-consumption flags identify unusual patterns for investigation, not confirmed wastage.
    </div>
    """,
    unsafe_allow_html=True,
)
