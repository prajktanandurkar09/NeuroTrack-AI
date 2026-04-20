import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from text_emotion import predict_emotion, get_all_scores
from face_emotion import predict_face_emotion, get_face_features
from emotion_fusion import fuse_emotions, fusion_report
from utils.task_mapper import recommend_task, get_focus_score, get_energy_label
from utils.session_manager import add_student_record, get_classroom_stats, export_history_csv

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroTrack AI",
    page_icon="NT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #f1f1f5;
}
.stApp {
    background: linear-gradient(135deg, #0f0f13 0%, #13131e 50%, #0f0f13 100%);
}
.main { background: transparent; }

/* All native text */
h1, h2, h3, h4, h5, h6 { color: #f1f1f5 !important; }
.stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em { color: #f1f1f5 !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color: rgba(255,255,255,0.5) !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }

/* st.metric */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.76rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    color: #ffffff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 1.9rem !important;
}

/* Widget labels */
.stTextArea label, .stTextInput label,
.stSelectbox label, .stMultiSelect label,
.stCameraInput label,
label[data-testid="stWidgetLabel"] p {
    color: rgba(255,255,255,0.78) !important;
    font-size: 0.85rem !important;
    font-weight: 500;
}

/* ── TEXTBOX FIX: force dark background + light text on ALL states ── */
.stTextArea textarea,
.stTextArea textarea:focus,
.stTextArea textarea:active,
.stTextArea textarea:hover {
    background-color: #1a1a2e !important;
    background: #1a1a2e !important;
    border: 1.5px solid rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
    color: #f1f1f5 !important;
    caret-color: #8b5cf6 !important;
    outline: none !important;
    -webkit-text-fill-color: #f1f1f5 !important;
    box-shadow: none !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.20) !important;
}
.stTextArea textarea::placeholder {
    color: rgba(255,255,255,0.30) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.30) !important;
}
/* Also fix single-line text input */
.stTextInput input,
.stTextInput input:focus,
.stTextInput input:active {
    background-color: #1a1a2e !important;
    background: #1a1a2e !important;
    border: 1.5px solid rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
    color: #f1f1f5 !important;
    caret-color: #8b5cf6 !important;
    -webkit-text-fill-color: #f1f1f5 !important;
}
.stTextInput input:focus { border-color: #6366f1 !important; }
.stTextInput input::placeholder {
    color: rgba(255,255,255,0.30) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.30) !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] [data-baseweb="select"] {
    background: #1a1a2e !important;
    border: 1.5px solid rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
}
[data-testid="stMultiSelect"] span { color: #ffffff !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: rgba(99,102,241,0.35) !important;
    border: none !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span { color: #ffffff !important; }

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border: none !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.22);
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.42) !important;
    background: linear-gradient(135deg, #4f52e0, #7c3aed) !important;
}
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Tabs */
            /* FIX TAB BAR PROPERLY */
.stTabs {
    margin-top: 20px;
}

.stTabs [data-baseweb="tab-list"] {
    display: flex;
    gap: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 6px;
}

.stTabs [data-baseweb="tab"] {
    flex: 1;
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    color: rgba(255,255,255,0.6);
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 500 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: rgba(255,255,255,0.85) !important; }
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.28) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    height: 10px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 10px !important;
}

/* Alerts */
.stAlert p { color: #f1f1f5 !important; font-weight: 500; }

/* Expander */
[data-testid="stExpander"] details {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: rgba(255,255,255,0.82) !important;
    font-weight: 500 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0c0c12 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #f1f1f5 !important; }
[data-testid="stSidebar"] .stCaption p { color: rgba(255,255,255,0.4) !important; }
[data-testid="stSidebar"] .stTextInput input {
    background-color: #1a1a2e !important;
    border: 1.5px solid rgba(99,102,241,0.35) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.07) !important; }

/* Spinner */
[data-testid="stSpinner"] p { color: rgba(255,255,255,0.7) !important; }

/* Custom classes */
.neuro-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-size: 2.4rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1; color: #ffffff;
}
.metric-label {
    font-size: 0.75rem; color: rgba(255,255,255,0.45);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 6px;
}
.section-title {
    font-size: 1.05rem; font-weight: 600;
    color: rgba(255,255,255,0.92); margin-bottom: 12px;
}
.result-box {
    border-radius: 14px; padding: 20px 24px;
    margin: 16px 0; border-left: 4px solid;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Plotly dark theme
# ──────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Space Grotesk, sans-serif", color="#d1d1db", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
        font=dict(color="#d1d1db", size=11),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.08)",
    ),
    hoverlabel=dict(bgcolor="#1e1e2e", bordercolor="#6366f1", font=dict(color="#ffffff", size=12)),
)
PLOTLY_XAXIS = dict(
    showgrid=False, zeroline=False, color="#9ca3af",
    tickfont=dict(color="#9ca3af", size=11),
    title_font=dict(color="#c4c4d0", size=12),
    linecolor="rgba(255,255,255,0.08)",
)
PLOTLY_YAXIS = dict(
    showgrid=True, gridcolor="rgba(255,255,255,0.06)", gridwidth=1,
    zeroline=False, range=[0, 100], color="#9ca3af",
    tickfont=dict(color="#9ca3af", size=11),
    title_font=dict(color="#c4c4d0", size=12),
    linecolor="rgba(255,255,255,0.08)",
)

# ──────────────────────────────────────────────
# EMOTION CONFIG  (no emojis in labels)
# ──────────────────────────────────────────────
EMOTION_CONFIG = {
    "happy":    {"color": "#22c55e", "bg": "rgba(34,197,94,0.1)"},
    "neutral":  {"color": "#3b82f6", "bg": "rgba(59,130,246,0.1)"},
    "sad":      {"color": "#8b5cf6", "bg": "rgba(139,92,246,0.1)"},
    "stressed": {"color": "#f59e0b", "bg": "rgba(245,158,11,0.1)"},
    "angry":    {"color": "#ef4444", "bg": "rgba(239,68,68,0.1)"},
}

def get_cfg(emotion):
    return EMOTION_CONFIG.get(emotion, {"color": "#6b7280", "bg": "rgba(107,114,128,0.1)"})

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
defaults = {
    "history": [], "classroom_scores": [],
    "text_result": None, "face_result": None,
    "final_result": None, "student_name": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## NeuroTrack AI")
    st.markdown("<p style='color:rgba(255,255,255,0.4);font-size:0.8rem;margin-top:-8px;'>Classroom Emotion Intelligence</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Student Info")
    student_name = st.text_input("Student Name", placeholder="Enter your name...", value=st.session_state.student_name)
    st.session_state.student_name = student_name
    st.divider()

    st.markdown("### Quick Demo")
    st.caption("Simulate classroom entries")
    dc = st.columns(2)
    with dc[0]:
        if st.button("Happy"):
            st.session_state.classroom_scores.append(90)
            add_student_record(st.session_state, f"Demo-{len(st.session_state.history)+1}", "happy", 90, 90, "demo")
        if st.button("Sad"):
            st.session_state.classroom_scores.append(40)
            add_student_record(st.session_state, f"Demo-{len(st.session_state.history)+1}", "sad", 80, 40, "demo")
    with dc[1]:
        if st.button("Stress"):
            st.session_state.classroom_scores.append(30)
            add_student_record(st.session_state, f"Demo-{len(st.session_state.history)+1}", "stressed", 85, 30, "demo")
        if st.button("Angry"):
            st.session_state.classroom_scores.append(22)
            add_student_record(st.session_state, f"Demo-{len(st.session_state.history)+1}", "angry", 85, 22, "demo")

    st.divider()
    if st.session_state.history:
        st.download_button(
            label="Export Session CSV",
            data=export_history_csv(st.session_state.history),
            file_name=f"neurotrack_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True,
        )
    if st.button("Reset Session", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

    st.divider()
    st.markdown("<div style='color:rgba(255,255,255,0.22);font-size:0.68rem;text-align:center;'>NeuroTrack AI v2.0<br>Emotion-Aware Learning System</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 24px;'>
  <h1 style='font-size:2.8rem;font-weight:700;
     background:linear-gradient(135deg,#6366f1,#8b5cf6,#a78bfa);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;'>
    NeuroTrack AI
  </h1>
  <p style='color:rgba(255,255,255,0.45);font-size:1rem;margin-top:8px;'>
    Classroom Energy and Emotion Intelligence System
  </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Emotion Analysis", "Classroom Dashboard", "Session History"])




# ══════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════
with tab1:
    col_text, col_face = st.columns(2, gap="large")

    with col_text:
        st.markdown("<div class='section-title'>Text Analysis</div>", unsafe_allow_html=True)
        text_input = st.text_area(
            "How are you feeling?",
            placeholder="Describe your mood. E.g. I feel stressed about the exam today",
            height=140, label_visibility="collapsed",
        )
        if st.button("Analyze Text", use_container_width=True, key="btn_text"):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    result = predict_emotion(text_input)
                    st.session_state.text_result = result
                    all_scores = get_all_scores(text_input)
                emotion, conf = result
                cfg = get_cfg(emotion)
                st.markdown(f"""
                <div class='result-box' style='background:{cfg["bg"]};border-color:{cfg["color"]};'>
                  <div style='font-size:1.4rem;font-weight:700;color:{cfg["color"]};'>{emotion.title()}</div>
                  <div style='color:rgba(255,255,255,0.55);font-size:0.85rem;margin-top:4px;'>Confidence: {conf}%</div>
                </div>""", unsafe_allow_html=True)
                if all_scores:
                    st.markdown("<div class='section-title' style='font-size:0.9rem;margin-top:16px;'>Emotion Breakdown</div>", unsafe_allow_html=True)
                    for emo, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                        c = get_cfg(emo)
                        row = st.columns([2, 5, 1])
                        row[0].markdown(f"<span style='color:{c['color']};font-size:0.9rem;'>{emo.title()}</span>", unsafe_allow_html=True)
                        row[1].progress(int(score))
                        row[2].markdown(f"<span style='font-family:monospace;font-size:0.82rem;color:#e2e2ea;'>{score:.0f}%</span>", unsafe_allow_html=True)
            else:
                st.warning("Please enter some text first.")

    with col_face:
        st.markdown("<div class='section-title'>Face Analysis</div>", unsafe_allow_html=True)
        img_input = st.camera_input("Capture your face", label_visibility="collapsed")
        if img_input:
            with st.spinner("Analyzing face..."):
                face_result = predict_face_emotion(img_input)
                st.session_state.face_result = face_result
                features = get_face_features(img_input)
            emotion, conf = face_result
            cfg = get_cfg(emotion)
            st.markdown(f"""
            <div class='result-box' style='background:{cfg["bg"]};border-color:{cfg["color"]};'>
              <div style='font-size:1.4rem;font-weight:700;color:{cfg["color"]};'>{emotion.title()}</div>
              <div style='color:rgba(255,255,255,0.55);font-size:0.85rem;margin-top:4px;'>Confidence: {conf}%</div>
            </div>""", unsafe_allow_html=True)
            with st.expander("Face Analysis Details"):
                fc = st.columns(2)
                fc[0].metric("Brightness",     features.get("brightness", "N/A"))
                fc[0].metric("Contrast",       features.get("contrast", "N/A"))
                fc[1].metric("Edge Density",   features.get("edge_density", "N/A"))
                fc[1].metric("Faces Detected", features.get("faces_detected", 0))

    st.divider()
    st.markdown("<div class='section-title' style='font-size:1.2rem;'>Final Emotion Fusion</div>", unsafe_allow_html=True)
    gc, _ = st.columns([1, 2])
    with gc:
        generate = st.button("Generate Final Result", use_container_width=True, key="btn_final")

    if generate:
        text_r = st.session_state.text_result
        face_r = st.session_state.face_result
        if not text_r and not face_r:
            st.warning("Please analyse text or capture a face first.")
        else:
            report        = fusion_report(text_r, face_r)
            final_emotion = report["final_emotion"]
            final_conf    = report["final_confidence"]
            st.session_state.final_result = (final_emotion, final_conf)
            focus_score = get_focus_score(final_emotion)
            task_data   = recommend_task(final_emotion)
            cfg         = get_cfg(final_emotion)
            name   = st.session_state.student_name or f"Student-{len(st.session_state.history)+1}"
            source = "both" if (text_r and face_r) else ("text" if text_r else "face")
            st.session_state.classroom_scores.append(focus_score)
            add_student_record(st.session_state, name, final_emotion, final_conf, focus_score, source)

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f"""<div class='metric-card'>
                  <div class='metric-value' style='color:{cfg["color"]};font-size:1.6rem;'>{final_emotion.title()}</div>
                  <div class='metric-label'>Detected Emotion</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""<div class='metric-card'>
                  <div class='metric-value' style='color:#818cf8;'>{final_conf}%</div>
                  <div class='metric-label'>Confidence</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                sc = "#22c55e" if focus_score >= 70 else "#f59e0b" if focus_score >= 40 else "#ef4444"
                st.markdown(f"""<div class='metric-card'>
                  <div class='metric-value' style='color:{sc};'>{focus_score}</div>
                  <div class='metric-label'>Focus Score</div>
                </div>""", unsafe_allow_html=True)
            with r4:
                ml = "Agreement" if report["sources_agree"] else "Weighted Fusion"
                st.markdown(f"""<div class='metric-card'>
                  <div class='metric-value' style='font-size:1.1rem;color:#a78bfa;'>{ml}</div>
                  <div class='metric-label'>Fusion Method</div>
                  <div style='color:rgba(255,255,255,0.38);font-size:0.73rem;margin-top:4px;'>Source: {source.title()}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#e2e2ea;font-weight:600;'>Focus Score: {focus_score}/100</span>", unsafe_allow_html=True)
            st.progress(focus_score / 100)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""<div class='neuro-card' style='border-color:{cfg["color"]}44;'>
              <div style='color:{cfg["color"]};font-weight:700;font-size:1rem;margin-bottom:10px;'>Recommended Actions</div>
              <div style='color:#e2e2ea;font-weight:500;font-size:0.95rem;'>{task_data["primary"]}</div>
            </div>""", unsafe_allow_html=True)

            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("<span style='color:#e2e2ea;font-weight:600;'>Suggested Tasks:</span>", unsafe_allow_html=True)
                for t in task_data["tasks"]:
                    st.markdown(f"- {t}")
            with tc2:
                st.markdown("<span style='color:#e2e2ea;font-weight:600;'>Study Tip:</span>", unsafe_allow_html=True)
                st.info(task_data["study_tip"])
                st.markdown("<span style='color:#e2e2ea;font-weight:600;'>Break Suggestion:</span>", unsafe_allow_html=True)
                st.success(task_data["break_suggestion"])

# ══════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<h3 style='color:#f1f1f5;'>Live Classroom Energy Monitor</h3>", unsafe_allow_html=True)

    scores  = st.session_state.classroom_scores
    history = st.session_state.history

    if not scores:
        st.markdown("""<div style='text-align:center;padding:60px;'>
          <div style='color:rgba(255,255,255,0.35);font-size:0.95rem;'>
            No data yet. Analyse emotions or use Quick Demo buttons in the sidebar.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        stats      = get_classroom_stats(history)
        avg_energy = sum(scores) / len(scores)
        energy_label, energy_desc = get_energy_label(avg_energy)

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, label, val, tip in [
            (m1, "Energy",   f"{avg_energy:.1f}", energy_label.split(" ", 1)[-1]),
            (m2, "Readings", str(stats["total_readings"]),   "total entries"),
            (m3, "Trend",    stats["trend"].title(),         "vs earlier"),
            (m4, "Peak",     str(stats.get("max_focus", 0)), "max focus"),
            (m5, "Low",      str(stats.get("min_focus", 0)), "min focus"),
        ]:
            with col:
                st.metric(label=label, value=val, help=tip)

        st.divider()
        cc1, cc2 = st.columns(2)

        with cc1:
            st.markdown("<span style='color:#e2e2ea;font-weight:600;'>Focus Score Timeline</span>", unsafe_allow_html=True)
            if history:
                df = pd.DataFrame(history)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(df)+1)), y=df["focus_score"],
                    mode="lines+markers", name="Focus Score",
                    line=dict(color="#6366f1", width=2.5),
                    marker=dict(size=8, color="#8b5cf6", line=dict(color="#ffffff", width=1.5)),
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.09)",
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(df)+1)), y=[avg_energy]*len(df),
                    mode="lines", name="Average",
                    line=dict(color="#f59e0b", width=1.5, dash="dash"),
                ))
                fig.update_layout(
                    **PLOTLY_BASE,
                    xaxis=dict(**PLOTLY_XAXIS, title="Student #"),
                    yaxis=dict(**PLOTLY_YAXIS, title="Focus Score"),
                )
                st.plotly_chart(fig, use_container_width=True)

        with cc2:
            st.markdown("<span style='color:#e2e2ea;font-weight:600;'>Emotion Distribution</span>", unsafe_allow_html=True)
            if stats["emotion_distribution"]:
                emo_dist = stats["emotion_distribution"]
                labels   = list(emo_dist.keys())
                values   = list(emo_dist.values())
                colors   = [get_cfg(e)["color"] for e in labels]
                disp_lbl = [e.title() for e in labels]
                fig2 = go.Figure(go.Pie(
                    labels=disp_lbl, values=values, hole=0.55,
                    marker=dict(colors=colors, line=dict(color="#0f0f13", width=2)),
                    textinfo="percent+label",
                    textfont=dict(size=12, color="#ffffff"),
                    insidetextfont=dict(color="#ffffff"),
                    outsidetextfont=dict(color="#e2e2ea"),
                ))
                fig2.update_layout(**PLOTLY_BASE, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""<div class='neuro-card' style='text-align:center;border-color:rgba(99,102,241,0.3);'>
          <div style='font-size:1.5rem;color:#f1f1f5;font-weight:600;'>{energy_label}</div>
          <div style='color:rgba(255,255,255,0.55);margin-top:8px;font-size:0.9rem;'>{energy_desc}</div>
        </div>""", unsafe_allow_html=True)
        st.progress(int(avg_energy) / 100)

        if stats.get("dominant_emotion"):
            dom = stats["dominant_emotion"]
            dc  = get_cfg(dom)
            dt  = recommend_task(dom)
            st.markdown(f"""<div class='neuro-card' style='border-color:{dc["color"]}55;margin-top:12px;'>
              <div style='color:{dc["color"]};font-weight:600;font-size:0.95rem;'>
                Class is predominantly <strong style='color:{dc["color"]};'>{dom.title()}</strong>
              </div>
              <div style='color:rgba(255,255,255,0.62);margin-top:8px;font-size:0.88rem;'>
                Teacher Tip: {dt["primary"]}
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<h3 style='color:#f1f1f5;'>Session History</h3>", unsafe_allow_html=True)
    history = st.session_state.history

    if not history:
        st.markdown("""<div style='text-align:center;padding:60px;'>
          <div style='color:rgba(255,255,255,0.35);margin-top:12px;'>No history yet. Start analysing emotions.</div>
        </div>""", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(history)
        fc1, fc2, _ = st.columns([2, 2, 1])
        with fc1:
            emotion_filter = st.multiselect("Filter by emotion", options=df["emotion"].unique().tolist(), default=df["emotion"].unique().tolist())
        with fc2:
            source_filter = st.multiselect("Filter by source", options=df["source"].unique().tolist(), default=df["source"].unique().tolist())

        filtered_df = df[df["emotion"].isin(emotion_filter) & df["source"].isin(source_filter)]

        hdr = st.columns([1, 2, 2, 2, 3, 1])
        for col, txt in zip(hdr, ["Time", "Name", "Emotion", "Focus", "Details", ""]):
            col.markdown(f"<span style='color:rgba(255,255,255,0.35);font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;'>{txt}</span>", unsafe_allow_html=True)
        st.markdown("<div style='border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:8px;'></div>", unsafe_allow_html=True)

        for _, row in filtered_df.iterrows():
            cfg = get_cfg(row["emotion"])
            bar = "#" * int(row["focus_score"] / 10) + "-" * (10 - int(row["focus_score"] / 10))
            h = st.columns([1, 2, 2, 2, 3, 1])
            h[0].markdown(f"<span style='color:rgba(255,255,255,0.35);font-size:0.78rem;'>{row['timestamp']}</span>", unsafe_allow_html=True)
            h[1].markdown(f"<span style='color:#e2e2ea;font-weight:600;'>{row['name']}</span>", unsafe_allow_html=True)
            h[2].markdown(f"<span style='color:{cfg['color']};font-weight:500;'>{row['emotion'].title()}</span>", unsafe_allow_html=True)
            h[3].markdown(f"<span style='font-family:monospace;color:#6366f1;font-size:0.78rem;'>{bar}</span> <span style='color:#e2e2ea;font-size:0.82rem;'>{row['focus_score']}</span>", unsafe_allow_html=True)
            h[4].markdown(f"<span style='color:rgba(255,255,255,0.38);font-size:0.74rem;'>Conf: {row['confidence']}% | Source: {row['source']}</span>", unsafe_allow_html=True)

        st.divider()
        if st.session_state.history:
            st.download_button(
                label="Download Session CSV",
                data=export_history_csv(st.session_state.history),
                file_name=f"neurotrack_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )


        