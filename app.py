"""SiteGuard: PPE compliance command center for infrastructure sites.

A Streamlit dashboard on top of a YOLOv8 PPE model. It is organised around how
infrastructure projects run safety: work zones with their own mandatory PPE,
cameras assigned to zones, shifts, an incident register with severity, and a
view of the health of the detection system itself.
"""

import html
import os
import platform
import tempfile
import time
from collections import deque
from datetime import datetime, timedelta

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SiteGuard · Infrastructure Safety Command Center",
    layout="wide",
    page_icon="🦺",
    initial_sidebar_state="expanded",
)

from ultralytics import YOLO  # noqa: E402  (imported after page config on purpose)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
MODEL_PATH = "best.pt"
LOG_FILE = "violation_logs.csv"
LOG_COLUMNS = ["Timestamp", "Violation", "Confidence", "Zone", "Camera", "Shift", "Severity"]
LEGACY_COLUMNS = ["Timestamp", "Violation", "Confidence"]
UNASSIGNED = "Unassigned"

# Work zones typical of a linear infrastructure project, with their default PPE rules.
ZONES = {
    "Z1 · Tunnelling & Underground": {"ppe": ["Hardhat", "Safety Vest", "Mask"], "risk": "Very High"},
    "Z2 · Steel & Formwork Erection": {"ppe": ["Hardhat", "Safety Vest"], "risk": "Very High"},
    "Z3 · Foundation & Piling": {"ppe": ["Hardhat", "Safety Vest"], "risk": "High"},
    "Z4 · Concrete Batching Plant": {"ppe": ["Hardhat", "Safety Vest", "Mask"], "risk": "High"},
    "Z5 · Road Works & Traffic Diversion": {"ppe": ["Hardhat", "Safety Vest"], "risk": "High"},
    "Z6 · Material Yard & Stores": {"ppe": ["Hardhat", "Safety Vest"], "risk": "Medium"},
}

PPE_ITEMS = ["Hardhat", "Safety Vest", "Mask"]
PPE_ICONS = {"Hardhat": "⛑️", "Safety Vest": "🦺", "Mask": "😷"}

# Severity follows the hazard each missing item exposes a worker to on site.
SEVERITY = {
    "NO-Hardhat": "Critical",     # falling objects, head strike
    "NO-Safety Vest": "High",     # struck by plant / vehicles
    "NO-Mask": "Medium",          # dust, silica, fumes
}
SEVERITY_WEIGHT = {"Critical": 5, "High": 3, "Medium": 1}
SEVERITY_BADGE = {"Critical": "🔴 Critical", "High": "🟠 High", "Medium": "🟡 Medium"}
SEVERITY_COLOR = {"Critical": "#dc3c3d", "High": "#f18a59", "Medium": "#fbb315"}

# Categorical slots validated against the dark panel surface (#111A2E).
TYPE_COLORS = {"NO-Hardhat": "#338ced", "NO-Safety Vest": "#e76624", "NO-Mask": "#00a888"}

SITE_ASSETS = {"Person": "Workers", "machinery": "Plant & Machinery", "vehicle": "Vehicles", "Safety Cone": "Safety Cones"}

INK = "#E6EAF2"
INK_MUTED = "#8B95A8"
GRID = "rgba(255,255,255,0.07)"


def _stretch_kwargs():
    """Full-width keyword for both old (use_container_width) and new (width) Streamlit APIs."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}


STRETCH = _stretch_kwargs()


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection model…")
def load_model(path):
    return YOLO(path)


if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {os.path.abspath(MODEL_PATH)}")
    st.stop()

model = load_model(MODEL_PATH)
CLASS_NAMES = model.names


@st.cache_resource(show_spinner=False)
def inference_device():
    """Return (ultralytics device argument, human label) for the best available accelerator."""
    try:
        import torch

        if torch.cuda.is_available():
            return "0", "CUDA · " + torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", "Apple MPS"
    except Exception:
        pass
    return "cpu", "CPU"


# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("alerts", deque(maxlen=8))
    ss.setdefault("last_logged", {})
    ss.setdefault("frames_processed", 0)
    ss.setdefault("latency_ms", deque(maxlen=120))
    ss.setdefault("session_started", datetime.now())
    ss.setdefault("session_incidents", 0)


init_state()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def shift_for(ts):
    """Three-shift roster used on most 24x7 infrastructure sites."""
    h = ts.hour
    if 6 <= h < 14:
        return "A · 06–14"
    if 14 <= h < 22:
        return "B · 14–22"
    return "C · 22–06"


def esc(value):
    return html.escape(str(value))


def mask_url(url):
    """Hide credentials in RTSP URLs before showing them on screen."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://•••@{rest.split('@', 1)[1]}"
    return url


def fmt_age(delta):
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


# --------------------------------------------------------------------------------------
# Incident log (CSV) — backward compatible with the original 3-column format
# --------------------------------------------------------------------------------------
def _empty_logs():
    return pd.DataFrame(columns=LOG_COLUMNS)


def load_logs():
    if not os.path.exists(LOG_FILE):
        return _empty_logs()
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception:
        return _empty_logs()
    if not all(c in df.columns for c in LEGACY_COLUMNS):
        return _empty_logs()

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    # Very old logs may store confidence as a percentage.
    df.loc[df["Confidence"] > 1, "Confidence"] = df["Confidence"] / 100

    for col in ("Zone", "Camera"):
        if col not in df.columns:
            df[col] = UNASSIGNED
        df[col] = df[col].fillna(UNASSIGNED)
    if "Shift" not in df.columns:
        df["Shift"] = df["Timestamp"].map(shift_for)
    if "Severity" not in df.columns:
        df["Severity"] = df["Violation"].map(SEVERITY).fillna("Medium")
    return df[LOG_COLUMNS]


def append_log(row):
    """Append one incident, upgrading a legacy log file to the new schema first."""
    new = pd.DataFrame([row], columns=LOG_COLUMNS)
    if os.path.exists(LOG_FILE):
        try:
            header = pd.read_csv(LOG_FILE, nrows=0).columns.tolist()
        except Exception:
            header = []
        if header == LOG_COLUMNS:
            new.to_csv(LOG_FILE, mode="a", header=False, index=False)
            return
        if all(c in header for c in LEGACY_COLUMNS):
            existing = load_logs()
            existing["Timestamp"] = existing["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            pd.concat([existing, new], ignore_index=True).to_csv(LOG_FILE, index=False)
            return
    new.to_csv(LOG_FILE, index=False)


# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
    --bg: #0A101C; --panel: #111A2E; --panel-2: #16213A; --panel-3: #1B2844;
    --border: rgba(255,255,255,0.08); --border-strong: rgba(255,255,255,0.14);
    --text: #E6EAF2; --muted: #8B95A8; --accent: #F5A524;
    --ok: #2BC48A; --warn: #fbb315; --serious: #f18a59; --crit: #dc3c3d;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}
.stApp {
    background:
        linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px) 0 0 / 32px 32px,
        linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px) 0 0 / 32px 32px,
        radial-gradient(1200px 600px at 0% -10%, #14223D 0%, var(--bg) 60%) fixed;
    color: var(--text);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.4rem; padding-bottom: 2rem; max-width: 1500px; }
h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -0.01em; }

/* ---------- command bar ---------- */
.cmdbar {
    display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 1.4rem;
    background: linear-gradient(180deg, #131E35 0%, #0F182B 100%);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 1rem 1.4rem; margin-bottom: 1rem;
    box-shadow: 0 12px 32px rgba(0,0,0,0.35);
    position: relative; overflow: hidden;
}
.cmdbar::before {
    content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px;
    background: repeating-linear-gradient(135deg, var(--accent) 0 14px, #111 14px 28px);
    opacity: .9;
}
.brand { display: flex; align-items: center; gap: .8rem; }
.brand-mark {
    width: 44px; height: 44px; border-radius: 10px; display: grid; place-items: center;
    background: rgba(245,165,36,0.12); border: 1px solid rgba(245,165,36,0.4); font-size: 1.4rem;
}
.brand-name { font-weight: 800; font-size: 1.15rem; letter-spacing: .14em; color: var(--text); }
.brand-sub { color: var(--muted); font-size: .78rem; letter-spacing: .04em; }
.ctx { display: flex; flex-wrap: wrap; gap: .5rem 1.6rem; }
.ctx-item { display: flex; flex-direction: column; min-width: 0; }
.ctx-label { color: var(--muted); font-size: .66rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.ctx-value { color: var(--text); font-size: .92rem; font-weight: 600; white-space: nowrap; }
.ctx-value.mono { font-family: var(--mono); font-size: .86rem; }
.pills { display: flex; gap: .5rem; flex-wrap: wrap; justify-content: flex-end; }
.pill {
    display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
    padding: 5px 12px; border-radius: 999px; font-weight: 700; font-size: .7rem; letter-spacing: .1em;
    border: 1px solid var(--border-strong); color: var(--muted); background: rgba(255,255,255,0.03);
}
.pill.ok { color: var(--ok); border-color: rgba(43,196,138,.35); background: rgba(43,196,138,.1); }
.pill.warn { color: var(--warn); border-color: rgba(251,179,21,.35); background: rgba(251,179,21,.08); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pill.ok .dot { animation: pulse 1.8s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(43,196,138,.6);} 70% { box-shadow: 0 0 0 8px rgba(43,196,138,0);} 100% { box-shadow: 0 0 0 0 rgba(43,196,138,0);} }
@media (max-width: 1100px) { .cmdbar { grid-template-columns: 1fr; } .pills { justify-content: flex-start; } }

/* ---------- KPI tiles ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: .8rem; margin-bottom: 1.2rem; }
@media (max-width: 1200px) { .kpi-grid { grid-template-columns: repeat(3, minmax(0,1fr)); } }
@media (max-width: 700px)  { .kpi-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
.kpi {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: .85rem 1rem; position: relative; overflow: hidden; min-width: 0;
}
.kpi::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--kpi-color, var(--accent)); }
.kpi-label { color: var(--muted); font-size: .68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; }
.kpi-value { color: var(--text); font-size: 1.55rem; font-weight: 750; margin-top: .25rem; line-height: 1.15;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }
.kpi-sub { color: var(--muted); font-size: .76rem; margin-top: .2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ---------- risk badge ---------- */
.risk { display: inline-flex; align-items: center; gap: 6px; font-weight: 800; letter-spacing: .06em; }
.risk-LOW { color: var(--ok); } .risk-MODERATE { color: var(--warn); }
.risk-HIGH { color: var(--serious); } .risk-SEVERE { color: var(--crit); }

/* ---------- panels ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel); border: 1px solid var(--border) !important; border-radius: 12px !important;
}
.panel-title { display: flex; align-items: center; justify-content: space-between; margin: 0 0 .6rem 0; }
.panel-title h4 { margin: 0 !important; padding: 0 !important; font-size: .95rem !important; letter-spacing: .02em; }
.panel-title .tag { color: var(--muted); font-size: .72rem; font-family: var(--mono); }

/* camera tile header */
.cam-head {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    background: #0B1322; border: 1px solid var(--border); border-bottom: none;
    border-radius: 10px 10px 0 0; padding: .5rem 2.6rem .5rem .8rem; font-family: var(--mono); font-size: .78rem; color: var(--muted);
}
.cam-head b { color: var(--text); }
.rec { color: var(--crit); font-weight: 700; display: inline-flex; align-items: center; gap: 6px; }
.rec .dot { animation: blink 1.2s steps(2, start) infinite; }
@keyframes blink { to { visibility: hidden; } }
.cam-empty {
    display: grid; place-items: center; text-align: center; aspect-ratio: 16 / 9;
    background:
        repeating-linear-gradient(0deg, rgba(255,255,255,0.025) 0 1px, transparent 1px 4px),
        #0B1322;
    border: 1px solid var(--border); border-radius: 0 0 10px 10px; color: var(--muted);
}
.cam-empty .big { font-size: 2.2rem; margin-bottom: .4rem; }
div[data-testid="stImage"] img { border-radius: 0 0 10px 10px; border: 1px solid var(--border); }

/* telemetry grid */
.tele { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: .5rem; margin-bottom: .8rem; }
.tele-cell { background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; padding: .55rem .7rem; }
.tele-k { color: var(--muted); font-size: .64rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.tele-v { color: var(--text); font-family: var(--mono); font-size: 1.05rem; font-weight: 700; }

/* PPE status board */
.ppe-row {
    display: grid; grid-template-columns: 1.6rem 1fr auto; align-items: center; gap: .6rem;
    padding: .5rem .7rem; border-radius: 8px; margin-bottom: .4rem;
    background: var(--panel-2); border: 1px solid var(--border);
}
.ppe-name { font-weight: 600; font-size: .88rem; }
.ppe-meta { color: var(--muted); font-size: .72rem; font-family: var(--mono); }
.badge { font-size: .66rem; font-weight: 800; letter-spacing: .08em; padding: 3px 8px; border-radius: 6px; white-space: nowrap; }
.b-ok { color: var(--ok); background: rgba(43,196,138,.12); }
.b-crit { color: #ff8a8b; background: rgba(220,60,61,.16); }
.b-adv { color: var(--warn); background: rgba(251,179,21,.12); }
.b-idle { color: var(--muted); background: rgba(255,255,255,.05); }

/* asset census */
.census { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: .4rem; margin-bottom: .8rem; }
.census-cell { text-align: center; background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; padding: .45rem .2rem; }
.census-v { font-family: var(--mono); font-weight: 700; font-size: 1.05rem; }
.census-k { color: var(--muted); font-size: .62rem; letter-spacing: .06em; text-transform: uppercase; }

/* alert feed */
.alert {
    display: grid; grid-template-columns: auto 1fr auto; gap: .6rem; align-items: center;
    border: 1px solid var(--border); border-left: 3px solid var(--sev, var(--crit));
    background: var(--panel-2); padding: .5rem .7rem; border-radius: 8px; margin-bottom: .4rem; font-size: .82rem;
}
.alert .what { font-weight: 700; }
.alert .where { color: var(--muted); font-size: .72rem; font-family: var(--mono); }
.alert .when { color: var(--muted); font-size: .72rem; font-family: var(--mono); white-space: nowrap; }
.empty-note { color: var(--muted); font-size: .82rem; padding: .4rem 0; }

/* zone rule card */
.zone-card { background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: .7rem .8rem; margin-top: .4rem; }
.zone-card .k { color: var(--muted); font-size: .66rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.chip { display: inline-block; margin: .25rem .25rem 0 0; padding: 3px 8px; border-radius: 6px; font-size: .74rem;
        background: rgba(255,255,255,.05); border: 1px solid var(--border); color: var(--text); }
.chip.on { border-color: rgba(245,165,36,.45); background: rgba(245,165,36,.1); }

/* buttons / tabs / misc */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px; font-weight: 650; border: 1px solid var(--border-strong);
}
.stButton > button[kind="primary"], .stDownloadButton > button {
    background: var(--accent) !important; color: #111 !important; border: none !important;
}
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px 8px 0 0; padding: .55rem 1rem; color: var(--muted); font-weight: 650; }
.stTabs [aria-selected="true"] { color: var(--text) !important; background: var(--panel); }
[data-testid="stMetric"] { background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: .7rem .9rem; }
.stDataFrame { border-radius: 10px; overflow: hidden; }

section[data-testid="stSidebar"] { background: #0C1424 !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] img { border-radius: 10px; }
.side-h {
    margin: 1.1rem 0 .5rem; color: var(--muted); font-size: .68rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase;
    display: flex; align-items: center; gap: .5rem;
}
.side-h::after { content: ""; flex: 1; height: 1px; background: var(--border); }

.health-row { display: flex; justify-content: space-between; gap: 1rem; padding: .45rem 0; border-bottom: 1px dashed var(--border); font-size: .86rem; }
.health-row:last-child { border-bottom: none; }
.health-row .k { color: var(--muted); }
.health-row .v { font-family: var(--mono); color: var(--text); text-align: right; }
.bar { height: 6px; border-radius: 4px; background: rgba(255,255,255,.07); overflow: hidden; margin-top: 4px; }
.bar > span { display: block; height: 100%; border-radius: 4px; }

.app-footer { text-align: center; font-size: .8em; color: var(--muted); margin: 2.4rem 0 1.4rem; }
.app-footer a { color: var(--muted); text-decoration: none; }
.app-footer a:hover { color: var(--accent); }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Sidebar: site context and detection policy
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.image("home.jpeg", **STRETCH)

    st.markdown('<div class="side-h">Site context</div>', unsafe_allow_html=True)
    project = st.text_input("Project", value="Metro Viaduct · Package C3")
    zone = st.selectbox("Work zone", list(ZONES.keys()), index=0)
    camera_id = st.text_input("Camera ID", value="CAM-01", max_chars=16).strip() or "CAM-01"

    zone_rule = ZONES[zone]
    mandatory = st.multiselect(
        "Mandatory PPE in this zone",
        PPE_ITEMS,
        default=zone_rule["ppe"],
        key=f"ppe_{zone}",
        help="Only missing mandatory PPE is logged as an incident. Other misses are shown as advisories.",
    )
    chips = "".join(
        f'<span class="chip {"on" if p in mandatory else ""}">{PPE_ICONS[p]} {p}</span>' for p in PPE_ITEMS
    )
    st.markdown(
        f'<div class="zone-card"><div class="k">Zone risk class</div>'
        f'<div style="font-weight:700;margin:.15rem 0 .3rem">{esc(zone_rule["risk"])}</div>'
        f'<div class="k">PPE matrix</div>{chips}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-h">Video source</div>', unsafe_allow_html=True)
    source_type = st.radio(
        "Input source",
        ["Browser Webcam (Photo)", "Upload Video", "Upload Image", "RTSP IP Camera", "OpenCV Webcam (Local Only)"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-h">Detection policy</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)
    cooldown_s = st.slider(
        "Alert cooldown (s)", 0, 30, 5,
        help="The same violation from the same camera is logged at most once per cooldown window, "
             "so a single worker is not logged on every frame.",
    )
    frame_skip = st.slider("Process every Nth frame", 1, 10, 2, help="Higher values reduce CPU load on edge devices.")


# --------------------------------------------------------------------------------------
# Command bar
# --------------------------------------------------------------------------------------
now = datetime.now()
device_arg, device = inference_device()
st.markdown(
    f"""
<div class="cmdbar">
  <div class="brand">
    <div class="brand-mark">🦺</div>
    <div><div class="brand-name">SITEGUARD</div><div class="brand-sub">Infrastructure Safety Command Center</div></div>
  </div>
  <div class="ctx">
    <div class="ctx-item"><span class="ctx-label">Project</span><span class="ctx-value">{esc(project)}</span></div>
    <div class="ctx-item"><span class="ctx-label">Zone</span><span class="ctx-value">{esc(zone)}</span></div>
    <div class="ctx-item"><span class="ctx-label">Camera</span><span class="ctx-value mono">{esc(camera_id)}</span></div>
    <div class="ctx-item"><span class="ctx-label">Shift</span><span class="ctx-value mono">{shift_for(now)}</span></div>
    <div class="ctx-item"><span class="ctx-label">Local time</span><span class="ctx-value mono">{now.strftime("%d %b %Y · %H:%M")}</span></div>
  </div>
  <div class="pills">
    <span class="pill ok"><span class="dot"></span>MODEL ONLINE</span>
    <span class="pill">{esc(device.upper())}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------------------
def risk_level(score):
    if score >= 60:
        return "SEVERE"
    if score >= 30:
        return "HIGH"
    if score >= 10:
        return "MODERATE"
    return "LOW"


def render_kpis(df):
    total = len(df)
    last24 = df[df["Timestamp"] >= now - timedelta(hours=24)] if total else df
    today = int((df["Timestamp"].dt.date == now.date()).sum()) if total else 0
    score = int(last24["Severity"].map(SEVERITY_WEIGHT).fillna(1).sum()) if total else 0
    level = risk_level(score)

    crit = df[df["Severity"] == "Critical"] if total else df
    if len(crit):
        last_crit = crit["Timestamp"].max()
        days_safe = (now.date() - last_crit.date()).days
        safe_val, safe_sub = f"{days_safe:,}", "Last critical " + last_crit.strftime("%d %b %Y")
    else:
        safe_val, safe_sub = "—", "No Critical incidents logged"

    if total:
        top = df["Violation"].value_counts()
        top_name, top_count = str(top.index[0]).replace("NO-", ""), int(top.iloc[0])
        conf = df["Confidence"].mean()
        avg_conf = f"{conf * 100:.0f}%" if pd.notna(conf) else "—"
        last_seen = fmt_age(now - df["Timestamp"].max())
    else:
        top_name, top_count, avg_conf, last_seen = "—", 0, "—", "never"

    risk_html = f'<span class="risk risk-{level}">● {level}</span>'
    cards = [
        ("Site Risk (24h)", risk_html, f"Weighted score {score}", SEVERITY_COLOR.get(
            {"SEVERE": "Critical", "HIGH": "High", "MODERATE": "Medium"}.get(level, ""), "#2BC48A")),
        ("Incidents Today", f"{today:,}", now.strftime("%d %b %Y"), "#F5A524"),
        ("Total Incidents", f"{total:,}", f"Last logged {last_seen}", "#7C8CFF"),
        ("Days Without Critical", safe_val, safe_sub, "#dc3c3d"),
        ("Most Missed PPE", esc(top_name), f"{top_count:,} incidents" if total else "No data yet", "#338ced"),
        ("Avg. Confidence", avg_conf, "Detection certainty", "#2BC48A"),
    ]
    tiles = "".join(
        f'<div class="kpi" style="--kpi-color:{c}"><div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>'
        for lbl, val, sub, c in cards
    )
    st.markdown(f'<div class="kpi-grid">{tiles}</div>', unsafe_allow_html=True)


logs = load_logs()
render_kpis(logs)


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------
def analyse(results):
    """Count each class in a result and work out per-item PPE status."""
    counts = {}
    detections = []
    for box in results.boxes:
        name = CLASS_NAMES[int(box.cls[0])]
        conf = float(box.conf[0])
        counts[name] = counts.get(name, 0) + 1
        detections.append((name, conf))

    ppe = {}
    for item in PPE_ITEMS:
        worn, missing = counts.get(item, 0), counts.get("NO-" + item, 0)
        if missing and item in mandatory:
            status = "VIOLATION"
        elif missing:
            status = "ADVISORY"
        elif worn:
            status = "COMPLIANT"
        else:
            status = "NOT SEEN"
        ppe[item] = {"worn": worn, "missing": missing, "status": status}
    return counts, detections, ppe


def record_incidents(detections):
    """Log mandatory-PPE violations, honouring the per-camera cooldown."""
    logged = 0
    t = time.time()
    for name, conf in detections:
        if not name.upper().startswith("NO-"):
            continue
        item = name[3:]
        if item not in mandatory:
            continue
        key = (camera_id, name)
        if cooldown_s and t - st.session_state.last_logged.get(key, 0) < cooldown_s:
            continue
        st.session_state.last_logged[key] = t
        ts = datetime.now()
        severity = SEVERITY.get(name, "Medium")
        append_log({
            "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Violation": name,
            "Confidence": round(conf, 2),
            "Zone": zone,
            "Camera": camera_id,
            "Shift": shift_for(ts),
            "Severity": severity,
        })
        st.session_state.alerts.appendleft({"ts": ts, "name": name, "conf": conf, "severity": severity})
        st.session_state.session_incidents += 1
        logged += 1
    return logged


def run_inference(frame):
    start = time.time()
    try:
        results = model(frame, conf=conf_threshold, device=device_arg, verbose=False)[0]
    except Exception:
        # Some accelerators reject individual ops; CPU always works.
        results = model(frame, conf=conf_threshold, device="cpu", verbose=False)[0]
    latency = (time.time() - start) * 1000
    st.session_state.frames_processed += 1
    st.session_state.latency_ms.append(latency)
    counts, detections, ppe = analyse(results)
    new_incidents = record_incidents(detections)
    return results.plot(), {
        "latency": latency, "counts": counts, "ppe": ppe,
        "objects": len(detections), "new_incidents": new_incidents,
    }


# --------------------------------------------------------------------------------------
# Live panel renderers
# --------------------------------------------------------------------------------------
def cam_header(source_label, live):
    rec = '<span class="rec"><span class="dot"></span>REC</span>' if live else '<span>STANDBY</span>'
    return (
        f'<div class="cam-head"><span>{rec}&nbsp;&nbsp;<b>{esc(camera_id)}</b> · {esc(zone)}</span>'
        f'<span>{esc(source_label)}</span></div>'
    )


def cam_placeholder(message, icon="📷"):
    return f'<div class="cam-empty"><div><div class="big">{icon}</div>{esc(message)}</div></div>'


def telemetry_html(info=None, fps=None):
    lat = f'{info["latency"]:.0f} ms' if info else "—"
    objs = str(info["objects"]) if info else "—"
    fps_s = f"{fps:.1f}" if fps else "—"
    inc = str(st.session_state.session_incidents)
    cells = [("Inference", lat), ("Throughput", f"{fps_s} fps"), ("Objects", objs), ("Session incidents", inc)]
    return '<div class="tele">' + "".join(
        f'<div class="tele-cell"><div class="tele-k">{k}</div><div class="tele-v">{v}</div></div>' for k, v in cells
    ) + "</div>"


def census_html(info=None):
    counts = info["counts"] if info else {}
    return '<div class="census">' + "".join(
        f'<div class="census-cell"><div class="census-v">{counts.get(cls, 0) if info else "—"}</div>'
        f'<div class="census-k">{label}</div></div>'
        for cls, label in SITE_ASSETS.items()
    ) + "</div>"


def ppe_board_html(info=None):
    rows = []
    for item in PPE_ITEMS:
        req = "Mandatory" if item in mandatory else "Optional"
        if info:
            p = info["ppe"][item]
            status = p["status"]
            meta = f'{req} · worn {p["worn"]} · missing {p["missing"]}'
        else:
            status, meta = "NOT SEEN", f"{req} · awaiting feed"
        cls = {"VIOLATION": "b-crit", "ADVISORY": "b-adv", "COMPLIANT": "b-ok"}.get(status, "b-idle")
        icon = {"VIOLATION": "✕", "ADVISORY": "!", "COMPLIANT": "✓"}.get(status, "–")
        rows.append(
            f'<div class="ppe-row"><span>{PPE_ICONS[item]}</span>'
            f'<div><div class="ppe-name">{item}</div><div class="ppe-meta">{meta}</div></div>'
            f'<span class="badge {cls}">{icon} {status}</span></div>'
        )
    return "".join(rows)


def alerts_html():
    alerts = list(st.session_state.alerts)
    if not alerts:
        return '<div class="empty-note">No incidents raised this session.</div>'
    out = []
    for a in alerts:
        color = SEVERITY_COLOR.get(a["severity"], "#dc3c3d")
        out.append(
            f'<div class="alert" style="--sev:{color}"><span>{SEVERITY_BADGE[a["severity"]].split()[0]}</span>'
            f'<div><div class="what">{esc(a["name"])} · {a["severity"]}</div>'
            f'<div class="where">{esc(camera_id)} · {a["conf"] * 100:.0f}% conf</div></div>'
            f'<span class="when">{a["ts"].strftime("%H:%M:%S")}</span></div>'
        )
    return "".join(out)


def panel_title(title, tag=""):
    return f'<div class="panel-title"><h4>{title}</h4><span class="tag">{esc(tag)}</span></div>'


class LivePanel:
    """Placeholders on the right of the feed, updated in place every frame."""

    def __init__(self, column):
        with column:
            with st.container(border=True):
                st.markdown(panel_title("Telemetry", "edge node"), unsafe_allow_html=True)
                self.tele = st.empty()
                st.markdown(panel_title("Site census", "this frame"), unsafe_allow_html=True)
                self.census = st.empty()
            with st.container(border=True):
                st.markdown(panel_title("PPE compliance", zone.split(" · ")[0]), unsafe_allow_html=True)
                self.ppe = st.empty()
            with st.container(border=True):
                st.markdown(panel_title("Alert feed", "latest first"), unsafe_allow_html=True)
                self.alerts = st.empty()
        self.update()

    def update(self, info=None, fps=None):
        self.tele.markdown(telemetry_html(info, fps), unsafe_allow_html=True)
        self.census.markdown(census_html(info), unsafe_allow_html=True)
        self.ppe.markdown(ppe_board_html(info), unsafe_allow_html=True)
        self.alerts.markdown(alerts_html(), unsafe_allow_html=True)


def show_frame(slot, frame_bgr):
    slot.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", **STRETCH)


def stream(video_source, label, feed_slot, head_slot, panel):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        head_slot.markdown(cam_header(label, False), unsafe_allow_html=True)
        feed_slot.markdown(cam_placeholder("Unable to open video source", "⚠️"), unsafe_allow_html=True)
        return
    head_slot.markdown(cam_header(label, True), unsafe_allow_html=True)
    fails, idx, info, t_last = 0, 0, None, time.time()
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 10:
                    break
                continue
            fails = 0
            idx += 1
            if idx % frame_skip:
                continue
            annotated, info = run_inference(frame)
            t_now = time.time()
            fps = 1.0 / max(t_now - t_last, 1e-6)
            t_last = t_now
            show_frame(feed_slot, annotated)
            panel.update(info, fps)
    finally:
        cap.release()
    head_slot.markdown(cam_header(label, False), unsafe_allow_html=True)
    st.toast("Stream ended", icon="📴")


def single_image(frame, feed_slot, head_slot, panel, label):
    head_slot.markdown(cam_header(label, False), unsafe_allow_html=True)
    annotated, info = run_inference(frame)
    show_frame(feed_slot, annotated)
    panel.update(info)
    if info["new_incidents"]:
        st.toast(f'{info["new_incidents"]} incident(s) logged', icon="⚠️")


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def style_chart(chart, height=260):
    return (
        chart.properties(height=height)
        .configure(background="transparent", font="system-ui, sans-serif")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=INK_MUTED, titleColor=INK_MUTED, gridColor=GRID, domainColor=GRID,
            tickColor=GRID, labelFontSize=11, titleFontSize=11, titleFontWeight=600,
        )
        .configure_legend(
            labelColor=INK, titleColor=INK_MUTED, orient="top", labelFontSize=11,
            titleFontSize=11, symbolType="square",
        )
    )


def type_scale(types):
    known = [t for t in TYPE_COLORS if t in types]
    return alt.Scale(domain=known, range=[TYPE_COLORS[t] for t in known])


def trend_chart(df):
    span = df["Timestamp"].max() - df["Timestamp"].min()
    if span <= timedelta(hours=3):
        freq, label, fmt = "5min", "5-minute window", "%H:%M"
    elif span <= timedelta(days=2):
        freq, label, fmt = "h", "Hour", "%d %b %H:%M"
    else:
        freq, label, fmt = "D", "Day", "%d %b"
    d = df.assign(Bucket=df["Timestamp"].dt.floor(freq))
    buckets = pd.date_range(d["Bucket"].min(), d["Bucket"].max(), freq=freq)
    if len(buckets) > 120:  # keep the bars readable on long histories
        buckets = buckets[-120:]
        d = d[d["Bucket"] >= buckets[0]]
    types = [t for t in TYPE_COLORS if t in set(d["Violation"])] or sorted(d["Violation"].unique())
    agg = (
        d.groupby(["Bucket", "Violation"]).size()
        .reindex(pd.MultiIndex.from_product([buckets, types], names=["Bucket", "Violation"]), fill_value=0)
        .reset_index(name="Incidents")
    )
    agg["Label"] = agg["Bucket"].dt.strftime(fmt)
    order = list(dict.fromkeys(agg["Label"]))
    step = max(1, len(order) // 14)
    chart = alt.Chart(agg).mark_bar(stroke="#111A2E", strokeWidth=1).encode(
        x=alt.X("Label:O", title=label, sort=order,
                axis=alt.Axis(labelAngle=0, values=order[::step], labelOverlap=True)),
        y=alt.Y("Incidents:Q", title="Incidents", stack="zero"),
        color=alt.Color("Violation:N", scale=type_scale(types), title=None, sort=types),
        order=alt.Order("Violation:N"),
        tooltip=[alt.Tooltip("Label:N", title=label), "Violation:N", "Incidents:Q"],
    )
    return style_chart(chart, 280)


def type_chart(df):
    agg = df["Violation"].value_counts().rename_axis("Violation").reset_index(name="Incidents")
    agg["Severity"] = agg["Violation"].map(SEVERITY).fillna("Medium")
    base = alt.Chart(agg).encode(
        y=alt.Y("Violation:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK, labelFontSize=12)),
        x=alt.X("Incidents:Q", title="Incidents"),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("Violation:N", scale=type_scale(agg["Violation"].unique()), legend=None),
        tooltip=["Violation:N", "Severity:N", "Incidents:Q"],
    )
    labels = base.mark_text(align="left", dx=6, color=INK, fontSize=12, fontWeight=600).encode(text="Incidents:Q")
    return style_chart(bars + labels, 200)


def zone_heatmap(df):
    agg = df.groupby(["Zone", "Violation"]).size().reset_index(name="Incidents")
    base = alt.Chart(agg).encode(
        x=alt.X("Violation:N", title=None, axis=alt.Axis(labelAngle=0, labelColor=INK)),
        y=alt.Y("Zone:N", title=None, axis=alt.Axis(labelColor=INK, labelLimit=260)),
    )
    cells = base.mark_rect(cornerRadius=4, stroke="#111A2E", strokeWidth=2).encode(
        color=alt.Color(
            "Incidents:Q",
            scale=alt.Scale(range=["#1b3358", "#1b64be", "#338ced", "#9dc8f7"]),
            legend=alt.Legend(title="Incidents", orient="right", gradientLength=120),
        ),
        tooltip=["Zone:N", "Violation:N", "Incidents:Q"],
    )
    text = base.mark_text(fontSize=12, fontWeight=700).encode(
        text="Incidents:Q",
        color=alt.condition(alt.datum.Incidents > float(agg["Incidents"].max()) * 0.6, alt.value("#0B1220"), alt.value(INK)),
    )
    return style_chart(cells + text, max(120, 46 * agg["Zone"].nunique()))


def shift_chart(df):
    d = df.assign(Hour=df["Timestamp"].dt.hour)
    agg = d.groupby("Hour").size().reindex(range(24), fill_value=0).reset_index(name="Incidents")
    agg["Shift"] = agg["Hour"].map(lambda h: shift_for(datetime(2000, 1, 1, h)))
    chart = alt.Chart(agg).mark_bar(cornerRadiusEnd=3, color="#338ced").encode(
        x=alt.X("Hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Incidents:Q", title="Incidents"),
        tooltip=["Hour:O", "Shift:N", "Incidents:Q"],
    )
    return style_chart(chart, 220)


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab_live, tab_register, tab_analytics, tab_system = st.tabs(
    ["🎥  Live Monitoring", "📋  Incident Register", "📊  Safety Analytics", "🖥️  System Health"]
)

# ---------------------------------- Incident register ---------------------------------
with tab_register:
    if logs.empty:
        st.info("No incidents logged yet. Incidents appear here as soon as a mandatory-PPE violation is detected.")
    else:
        f1, f2, f3, f4 = st.columns([1.3, 1.4, 1, 1])
        min_d, max_d = logs["Timestamp"].min().date(), logs["Timestamp"].max().date()
        dates = f1.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max(max_d, now.date()))
        zones_f = f2.multiselect("Zone", sorted(logs["Zone"].unique()), placeholder="All zones")
        sev_f = f3.multiselect("Severity", ["Critical", "High", "Medium"], placeholder="All")
        cams_f = f4.multiselect("Camera", sorted(logs["Camera"].unique()), placeholder="All cameras")

        view = logs.copy()
        if isinstance(dates, (list, tuple)) and len(dates) == 2:
            view = view[(view["Timestamp"].dt.date >= dates[0]) & (view["Timestamp"].dt.date <= dates[1])]
        if zones_f:
            view = view[view["Zone"].isin(zones_f)]
        if sev_f:
            view = view[view["Severity"].isin(sev_f)]
        if cams_f:
            view = view[view["Camera"].isin(cams_f)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Incidents in view", f"{len(view):,}")
        m2.metric("🔴 Critical", f'{int((view["Severity"] == "Critical").sum()):,}')
        m3.metric("🟠 High", f'{int((view["Severity"] == "High").sum()):,}')
        m4.metric("🟡 Medium", f'{int((view["Severity"] == "Medium").sum()):,}')

        table = view.sort_values("Timestamp", ascending=False).copy()
        table.insert(0, "ID", ["INC-" + str(i + 1).zfill(5) for i in table.index])
        table["Severity"] = table["Severity"].map(SEVERITY_BADGE).fillna(table["Severity"])
        st.dataframe(
            table[["ID", "Timestamp", "Severity", "Violation", "Zone", "Camera", "Shift", "Confidence"]],
            **STRETCH,
            hide_index=True,
            height=420,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent"),
            },
        )

        export = view.copy()
        export["Timestamp"] = export["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button(
            "📥 Export filtered register (CSV)",
            data=export.to_csv(index=False),
            file_name=f"incident_register_{now.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            **STRETCH,
        )
        with d2.popover("🗑️ Clear register", **STRETCH):
            st.warning("This permanently deletes every logged incident.")
            if st.button("Yes, delete all incidents", type="primary"):
                os.remove(LOG_FILE)
                st.session_state.alerts.clear()
                st.rerun()

# ---------------------------------- Safety analytics ----------------------------------
with tab_analytics:
    if logs.empty:
        st.info("Analytics appear once incidents have been logged.")
    else:
        st.caption(
            f'Based on {len(logs):,} incidents from {logs["Timestamp"].min():%d %b %Y} '
            f'to {logs["Timestamp"].max():%d %b %Y}.'
        )
        with st.container(border=True):
            st.markdown(panel_title("Incident trend by violation type", "stacked counts"), unsafe_allow_html=True)
            st.altair_chart(trend_chart(logs), **STRETCH, theme=None)

        a1, a2 = st.columns(2, gap="medium")
        with a1:
            with st.container(border=True):
                st.markdown(panel_title("Incidents by violation type"), unsafe_allow_html=True)
                st.altair_chart(type_chart(logs), **STRETCH, theme=None)
            with st.container(border=True):
                st.markdown(panel_title("Incidents by hour of day", "shift A 06–14 · B 14–22 · C 22–06"),
                            unsafe_allow_html=True)
                st.altair_chart(shift_chart(logs), **STRETCH, theme=None)
        with a2:
            with st.container(border=True):
                st.markdown(panel_title("Zone × violation hotspots"), unsafe_allow_html=True)
                st.altair_chart(zone_heatmap(logs), **STRETCH, theme=None)
            with st.container(border=True):
                st.markdown(panel_title("Severity mix"), unsafe_allow_html=True)
                sev = logs["Severity"].value_counts()
                rows = ""
                for name in ["Critical", "High", "Medium"]:
                    n = int(sev.get(name, 0))
                    pct = n / len(logs) * 100
                    rows += (
                        f'<div class="health-row"><span class="k">{SEVERITY_BADGE[name]}</span>'
                        f'<span class="v">{n:,} · {pct:.0f}%</span></div>'
                        f'<div class="bar"><span style="width:{pct:.1f}%;background:{SEVERITY_COLOR[name]}"></span></div>'
                    )
                st.markdown(rows, unsafe_allow_html=True)
                st.caption("Critical: no hardhat (falling objects). High: no hi-vis vest (plant and vehicle "
                           "movement). Medium: no mask (dust and silica).")

# ---------------------------------- System health -------------------------------------
with tab_system:
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.2)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(os.path.abspath("."))
        resources = [
            ("CPU", cpu, f"{cpu:.0f}%"),
            ("Memory", mem.percent, f"{mem.used / 2**30:.1f} / {mem.total / 2**30:.1f} GB"),
            ("Disk", disk.percent, f"{disk.used / 2**30:.0f} / {disk.total / 2**30:.0f} GB"),
        ]
    except Exception:
        resources = []

    lat = list(st.session_state.latency_ms)
    avg_lat = sum(lat) / len(lat) if lat else None

    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        with st.container(border=True):
            st.markdown(panel_title("Edge node resources", platform.node() or "host"), unsafe_allow_html=True)
            if resources:
                out = ""
                for name, pct, detail in resources:
                    color = "#2BC48A" if pct < 70 else ("#fbb315" if pct < 90 else "#dc3c3d")
                    out += (
                        f'<div class="health-row"><span class="k">{name}</span><span class="v">{detail}</span></div>'
                        f'<div class="bar"><span style="width:{pct:.0f}%;background:{color}"></span></div>'
                    )
                st.markdown(out, unsafe_allow_html=True)
            else:
                st.caption("Resource metrics unavailable (psutil not installed).")
    with s2:
        with st.container(border=True):
            st.markdown(panel_title("Detection model", os.path.basename(MODEL_PATH)), unsafe_allow_html=True)
            size_mb = os.path.getsize(MODEL_PATH) / 2**20
            rows = [
                ("Architecture", "YOLOv8 · " + str(getattr(model, "task", "detect"))),
                ("Weights", f"{size_mb:.1f} MB"),
                ("Classes", str(len(CLASS_NAMES))),
                ("Inference device", device),
                ("Confidence threshold", f"{conf_threshold:.2f}"),
                ("Avg. latency (session)", f"{avg_lat:.0f} ms" if avg_lat else "—"),
                ("Frames processed", f"{st.session_state.frames_processed:,}"),
            ]
            st.markdown("".join(
                f'<div class="health-row"><span class="k">{k}</span><span class="v">{esc(v)}</span></div>' for k, v in rows
            ), unsafe_allow_html=True)
    with s3:
        with st.container(border=True):
            st.markdown(panel_title("Data & runtime"), unsafe_allow_html=True)
            log_kb = os.path.getsize(LOG_FILE) / 1024 if os.path.exists(LOG_FILE) else 0
            try:
                import ultralytics
                ul_ver = ultralytics.__version__
            except Exception:
                ul_ver = "—"
            rows = [
                ("Incident log", f"{LOG_FILE} · {log_kb:.0f} KB"),
                ("Records", f"{len(logs):,}"),
                ("Session uptime", fmt_age(now - st.session_state.session_started).replace(" ago", "")),
                ("Python", platform.python_version()),
                ("Streamlit", st.__version__),
                ("Ultralytics", ul_ver),
                ("OpenCV", cv2.__version__),
            ]
            st.markdown("".join(
                f'<div class="health-row"><span class="k">{k}</span><span class="v">{esc(v)}</span></div>' for k, v in rows
            ), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(panel_title("Model class map", "what the detector can see"), unsafe_allow_html=True)
        groups = {
            "Compliant PPE": [n for n in CLASS_NAMES.values() if n in PPE_ITEMS],
            "PPE violations": [n for n in CLASS_NAMES.values() if n.upper().startswith("NO-")],
            "Site assets": [n for n in CLASS_NAMES.values() if n in SITE_ASSETS],
        }
        cols = st.columns(len(groups))
        for col, (title, names) in zip(cols, groups.items()):
            col.markdown(
                f'<div class="zone-card"><div class="k">{title}</div>'
                + "".join(f'<span class="chip">{esc(n)}</span>' for n in names) + "</div>",
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown(panel_title("Zone PPE policy", "defaults · override per zone in the sidebar"),
                    unsafe_allow_html=True)
        policy = pd.DataFrame([
            {"Zone": z, "Risk class": r["risk"], **{p: ("✅ Required" if p in r["ppe"] else "—") for p in PPE_ITEMS}}
            for z, r in ZONES.items()
        ])
        st.dataframe(policy, **STRETCH, hide_index=True)


st.markdown(
    """
<div class='app-footer'>
    SiteGuard UI · built on the AI PPE Detection project by Group ID: G00171 (Intel AI for Manufacturing) ·
    <a href='https://github.com/darshan1654/AI-PPE-Detection' target='_blank'>original repository</a>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------- Live monitoring -----------------------------------
with tab_live:
    def render_live():
        """Rendered last: a running stream blocks the script, so every other tab must already be drawn."""
        feed_col, side_col = st.columns([2.1, 1], gap="medium")
        with feed_col:
            head_slot = st.empty()
            feed_slot = st.empty()
            controls = st.container()
        panel = LivePanel(side_col)
        head_slot.markdown(cam_header(source_type, False), unsafe_allow_html=True)
        feed_slot.markdown(cam_placeholder("No active feed. Choose a source below to start monitoring."),
                           unsafe_allow_html=True)

        with controls:
            if source_type == "Browser Webcam (Photo)":
                captured = st.camera_input("Capture a site snapshot (browser permission required)")
                if captured:
                    frame = cv2.imdecode(np.frombuffer(captured.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    single_image(frame, feed_slot, head_slot, panel, "Browser snapshot")

            elif source_type == "Upload Image":
                up = st.file_uploader("Upload a site image", type=["jpg", "jpeg", "png"])
                if up:
                    frame = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    single_image(frame, feed_slot, head_slot, panel, up.name)

            elif source_type == "Upload Video":
                up = st.file_uploader("Upload site footage", type=["mp4", "avi", "mov", "mkv"])
                c1, c2 = st.columns(2)
                use_sample = c2.button("▶ Run bundled sample.mp4", **STRETCH,
                                       disabled=not os.path.exists("sample.mp4"))
                if up and c1.button("▶ Start analysis", type="primary", **STRETCH):
                    path = os.path.join(tempfile.gettempdir(), os.path.basename(up.name))
                    with open(path, "wb") as f:
                        f.write(up.getvalue())
                    st.button("⏹ Stop", key="stop_upload")
                    stream(path, up.name, feed_slot, head_slot, panel)
                elif use_sample:
                    st.button("⏹ Stop", key="stop_sample")
                    stream("sample.mp4", "sample.mp4", feed_slot, head_slot, panel)

            elif source_type == "RTSP IP Camera":
                rtsp_url = st.text_input(
                    "RTSP stream URL",
                    placeholder="rtsp://user:password@192.168.1.100:554/stream1",
                    type="password",
                    help="Credentials are masked on screen and never written to the incident log.",
                )
                if rtsp_url and st.button("📡 Connect stream", type="primary"):
                    st.button("⏹ Disconnect", key="stop_rtsp")
                    stream(rtsp_url, mask_url(rtsp_url), feed_slot, head_slot, panel)

            elif source_type == "OpenCV Webcam (Local Only)":
                if os.environ.get("IS_STREAMLIT_CLOUD"):
                    st.error("The OpenCV webcam is only available when the app runs on a local machine.")
                    st.info("Tip: use **Browser Webcam (Photo)** in cloud deployments.")
                else:
                    st.caption("Uses the camera attached to the machine running this app (device 0).")
                    if st.button("🎥 Start local camera", type="primary"):
                        st.button("⏹ Stop", key="stop_local")
                        stream(0, "Local camera 0", feed_slot, head_slot, panel)

    render_live()
