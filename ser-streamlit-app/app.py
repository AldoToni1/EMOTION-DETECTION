"""Aplikasi Streamlit untuk Speech Emotion Recognition berbasis WavLM."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoFeatureExtractor

from model import MODEL_PATH, PRETRAINED_MODEL, load_model
from utils import (
    ID2LABEL,
    LABEL2ID,
    get_audio_info,
    get_transcription_waveform,
    load_audio,
    predict_emotion,
    preprocess_audio,
    transcribe_audio,
)

# ---------------------------------------------------------------------------
# Konfigurasi STT (Speech-to-Text) — Whisper via transformers
# Ganti ke "openai/whisper-base" agar lebih cepat di CPU (akurasi sedikit turun)
# ---------------------------------------------------------------------------
ENABLE_STT = True
WHISPER_MODEL = "openai/whisper-small"
WHISPER_LANGUAGE = "indonesian"

EMOTION_ICONS = {
    "netral": "😐",
    "senang": "😊",
    "sedih": "😢",
    "marah": "😠",
    "takut": "😨",
    "jijik": "🤢",
}

EMOTION_COLORS = {
    "netral": "#94a3b8",
    "senang": "#fbbf24",
    "sedih": "#60a5fa",
    "marah": "#f87171",
    "takut": "#38bdf8",
    "jijik": "#06b6d4",
}

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(59,130,246,0.12));
            border: 1px solid rgba(96,165,250,0.22);
            border-radius: 16px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            color: rgba(226,232,240,0.82);
            margin: 0 0 1rem 0;
            line-height: 1.55;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(37,99,235,0.22);
            color: #93c5fd;
            border: 1px solid rgba(96,165,250,0.35);
            border-radius: 999px;
            padding: 0.28rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .section-card {
            background: rgba(30,41,59,0.55);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .section-step {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #60a5fa;
            margin-bottom: 0.35rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 650;
            margin: 0 0 0.35rem 0;
        }
        .section-desc {
            font-size: 0.92rem;
            color: rgba(203,213,225,0.78);
            margin: 0 0 0.75rem 0;
        }
        .meta-label {
            font-size: 0.75rem;
            color: rgba(148,163,184,0.95);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.15rem;
        }
        .meta-value {
            font-size: 1rem;
            font-weight: 600;
            color: #f8fafc;
            word-break: break-word;
        }
        .empty-state {
            text-align: center;
            padding: 2.5rem 1.5rem;
            border: 1px dashed rgba(148,163,184,0.28);
            border-radius: 16px;
            background: rgba(15,23,42,0.35);
            margin: 1rem 0 1.5rem 0;
        }
        .empty-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
        .empty-title { font-size: 1.05rem; font-weight: 650; margin-bottom: 0.25rem; }
        .empty-desc { font-size: 0.92rem; color: rgba(203,213,225,0.72); }
        .result-card {
            background: linear-gradient(160deg, rgba(30,41,59,0.92), rgba(15,23,42,0.82));
            border: 1px solid rgba(96,165,250,0.32);
            border-radius: 20px;
            padding: 2.75rem 2rem;
            margin: 1rem 0 1.25rem 0;
            text-align: center;
            box-shadow: 0 12px 40px rgba(15,23,42,0.45);
        }
        .result-inner {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .result-emoji {
            font-size: 4.5rem;
            line-height: 1;
            margin-bottom: 0.65rem;
            filter: drop-shadow(0 4px 12px rgba(0,0,0,0.25));
        }
        .result-label {
            font-size: 2.1rem;
            font-weight: 700;
            margin: 0 0 0.85rem 0;
            text-transform: capitalize;
            letter-spacing: -0.01em;
        }
        .result-conf-label {
            font-size: 0.78rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #94a3b8;
            margin-bottom: 0.35rem;
        }
        .result-confidence {
            font-size: 4.25rem;
            font-weight: 800;
            margin: 0;
            line-height: 1;
            background: linear-gradient(135deg, #93c5fd, #60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .result-dominance {
            font-size: 0.82rem;
            color: #94a3b8;
            margin: 0 0 0.5rem 0;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 650;
        }
        .result-rank-note {
            font-size: 0.92rem;
            color: rgba(203,213,225,0.82);
            margin: 0.85rem 0 0 0;
            line-height: 1.45;
        }
        .result-margin {
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(37,99,235,0.18);
            border: 1px solid rgba(96,165,250,0.28);
            font-size: 0.82rem;
            color: #93c5fd;
            font-weight: 600;
        }
        .top3-card {
            background: rgba(15,23,42,0.55);
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            text-align: center;
            min-height: 118px;
        }
        .top3-rank {
            font-size: 0.72rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .top3-emoji { font-size: 1.6rem; margin: 0.15rem 0; }
        .top3-label { font-size: 1rem; font-weight: 650; text-transform: capitalize; }
        .top3-pct {
            font-size: 1.15rem;
            font-weight: 700;
            color: #60a5fa;
            margin-top: 0.15rem;
        }
        .top3-conf-label {
            font-size: 0.68rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .transcript-card {
            background: linear-gradient(145deg, rgba(30,41,59,0.85), rgba(15,23,42,0.78));
            border: 1px solid rgba(96,165,250,0.28);
            border-left: 4px solid #3b82f6;
            border-radius: 14px;
            padding: 1.1rem 1.35rem;
            margin: 0.5rem 0 1rem 0;
        }
        .transcript-head {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #60a5fa;
            margin-bottom: 0.45rem;
        }
        .transcript-text {
            font-size: 1.05rem;
            color: #f1f5f9;
            line-height: 1.55;
            font-style: italic;
        }
        .transcript-empty {
            font-size: 0.95rem;
            color: rgba(148,163,184,0.85);
            font-style: italic;
        }
        .prob-row-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.92rem;
            margin-bottom: 0.2rem;
        }
        .prob-bar-wrap {
            background: rgba(51,65,85,0.65);
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
            margin-bottom: 0.85rem;
        }
        .prob-bar-fill {
            height: 10px;
            border-radius: 999px;
        }
        .sidebar-pill {
            display: inline-block;
            background: rgba(51,65,85,0.75);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 999px;
            padding: 0.18rem 0.62rem;
            margin: 0.12rem 0.18rem 0.12rem 0;
            font-size: 0.78rem;
            text-transform: capitalize;
        }
        .sidebar-header {
            background: linear-gradient(145deg, rgba(37,99,235,0.28), rgba(29,78,216,0.18));
            border: 1px solid rgba(96,165,250,0.22);
            border-radius: 14px;
            padding: 1rem 1rem 0.85rem 1rem;
            margin-bottom: 0.85rem;
            text-align: center;
        }
        .sidebar-header-icon {
            font-size: 1.75rem;
            margin-bottom: 0.25rem;
        }
        .sidebar-header-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.01em;
        }
        .sidebar-header-sub {
            font-size: 0.78rem;
            color: rgba(203,213,225,0.72);
            margin: 0.25rem 0 0 0;
            line-height: 1.4;
        }
        .sidebar-stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
            margin-bottom: 0.85rem;
        }
        .sidebar-stat-card {
            background: rgba(15,23,42,0.55);
            border: 1px solid rgba(148,163,184,0.14);
            border-radius: 12px;
            padding: 0.65rem 0.7rem;
        }
        .sidebar-stat-icon {
            font-size: 1rem;
            margin-bottom: 0.15rem;
        }
        .sidebar-stat-label {
            font-size: 0.68rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.1rem;
        }
        .sidebar-stat-value {
            font-size: 0.82rem;
            font-weight: 650;
            color: #f1f5f9;
            line-height: 1.25;
        }
        .sidebar-status-card {
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }
        .sidebar-status-ok {
            background: rgba(37,99,235,0.12);
            border: 1px solid rgba(96,165,250,0.32);
        }
        .sidebar-status-fail {
            background: rgba(248,113,113,0.1);
            border: 1px solid rgba(248,113,113,0.28);
        }
        .sidebar-status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .sidebar-status-dot.ok {
            background: #3b82f6;
            box-shadow: 0 0 8px rgba(59,130,246,0.65);
        }
        .sidebar-status-dot.fail {
            background: #f87171;
            box-shadow: 0 0 8px rgba(248,113,113,0.55);
        }
        .sidebar-status-text {
            font-size: 0.84rem;
            font-weight: 650;
            color: #f8fafc;
        }
        .sidebar-status-sub {
            font-size: 0.72rem;
            color: rgba(203,213,225,0.65);
            margin-top: 0.05rem;
        }
        .sidebar-section-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #60a5fa;
            margin: 0.15rem 0 0.55rem 0;
        }
        .sidebar-emotion-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.45rem;
            margin-bottom: 0.85rem;
        }
        .sidebar-emotion-item {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(15,23,42,0.45);
            border: 1px solid rgba(148,163,184,0.12);
            border-left: 3px solid var(--emotion-color, #2563eb);
            border-radius: 10px;
            padding: 0.42rem 0.55rem;
            font-size: 0.78rem;
            text-transform: capitalize;
        }
        .sidebar-emotion-emoji { font-size: 1rem; line-height: 1; }
        .sidebar-emotion-id {
            font-size: 0.65rem;
            color: #64748b;
            margin-left: auto;
        }
        .sidebar-howto {
            background: rgba(30,41,59,0.45);
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.75rem;
        }
        .sidebar-howto-step {
            display: flex;
            align-items: flex-start;
            gap: 0.55rem;
            margin-bottom: 0.55rem;
        }
        .sidebar-howto-step:last-child { margin-bottom: 0; }
        .sidebar-howto-num {
            background: rgba(37,99,235,0.25);
            color: #93c5fd;
            border-radius: 999px;
            width: 1.35rem;
            height: 1.35rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.72rem;
            font-weight: 700;
            flex-shrink: 0;
        }
        .sidebar-howto-text {
            font-size: 0.78rem;
            color: rgba(226,232,240,0.85);
            line-height: 1.35;
            padding-top: 0.05rem;
        }
        .sidebar-divider {
            border: none;
            border-top: 1px solid rgba(148,163,184,0.12);
            margin: 0.65rem 0;
        }
        .status-ok { color: #60a5fa; font-weight: 650; }
        .status-fail { color: #f87171; font-weight: 650; }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(12,30,58,0.92));
        }
        div[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
            border: none !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 0.72rem 1rem !important;
            font-weight: 650 !important;
            box-shadow: 0 8px 24px rgba(37,99,235,0.28) !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
            box-shadow: 0 10px 28px rgba(37,99,235,0.36) !important;
        }
        div.stButton > button:disabled {
            opacity: 0.55 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Memuat model WavLM...")
def load_ser_model(model_path: str, device_name: str):
    device = torch.device(device_name)
    model = load_model(model_path, device)
    return model, device


@st.cache_resource(show_spinner="Memuat feature extractor...")
def load_feature_extractor():
    return AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL)


@st.cache_resource(show_spinner="Memuat model Whisper (STT)...")
def load_asr_pipeline(model_name: str, device_name: str):
    """Cache pipeline Whisper untuk transkrip audio ke teks."""
    from transformers import pipeline

    device = 0 if device_name == "cuda" else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )


def format_file_size(size_bytes: int | None) -> str:
    if not size_bytes:
        return "—"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def summarize_prediction(result: dict) -> dict:
    """Ringkas prediksi untuk tampilan ranking & margin (UI only)."""
    prob_df = result["probabilities_df"]
    top_pct = float(prob_df.iloc[0]["Persentase (%)"])

    second_label = None
    second_pct = 0.0
    if len(prob_df) > 1:
        second_label = str(prob_df.iloc[1]["Emosi"])
        second_pct = float(prob_df.iloc[1]["Persentase (%)"])

    margin_pp = top_pct - second_pct

    if margin_pp >= 20:
        separation = "Pemisahan kuat dari emosi lain"
    elif margin_pp >= 10:
        separation = "Pemisahan cukup jelas dari emosi lain"
    else:
        separation = "Pemisahan tipis — emosi lain masih dekat"

    return {
        "top_label": result["predicted_label"],
        "top_pct": top_pct,
        "second_label": second_label,
        "second_pct": second_pct,
        "margin_pp": margin_pp,
        "separation": separation,
        "num_classes": len(prob_df),
    }


def render_transcript_card(text: str) -> None:
    if text:
        body = f'<div class="transcript-text">"{text}"</div>'
    else:
        body = (
            '<div class="transcript-empty">Tidak ada ucapan yang terdeteksi '
            "(audio mungkin tanpa kata-kata yang jelas).</div>"
        )
    st.markdown(
        f"""
        <div class="transcript-card">
            <div class="transcript-head">📝 Transkrip Ucapan (Speech-to-Text)</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Speech Emotion Recognition</div>
            <p class="hero-subtitle">
                Unggah suara, dengarkan preview, lalu sistem akan memprediksi emosi dominan
                sekaligus menampilkan transkrip teks dari audio.
            </p>
            <span class="hero-badge">WavLM + Whisper STT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">🎧</div>
            <div class="empty-title">Belum ada audio yang diunggah.</div>
            <div class="empty-desc">Mulai dengan mengunggah file .wav atau .mp3.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metadata_card(
    filename: str,
    duration_sec: float,
    sample_rate: int,
    channels: int,
    file_size: str,
) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-step">Metadata Audio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="meta-label">Nama File</div><div class="meta-value">{filename}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="meta-label">Durasi</div><div class="meta-value">{duration_sec} dtk</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="meta-label">Sample Rate</div><div class="meta-value">{sample_rate} Hz</div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="meta-label">Channel</div><div class="meta-value">{channels}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="meta-label">Ukuran File</div><div class="meta-value">{file_size}</div>',
        unsafe_allow_html=True,
    )


def render_top3_cards(prob_df: pd.DataFrame) -> None:
    top3 = prob_df.head(3).reset_index(drop=True)
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if i >= len(top3):
            break
        row = top3.iloc[i]
        emotion = row["Emosi"]
        pct = row["Persentase (%)"]
        col.markdown(
            f"""
            <div class="top3-card">
                <div class="top3-rank">#{i + 1}</div>
                <div class="top3-emoji">{EMOTION_ICONS.get(emotion, "🎭")}</div>
                <div class="top3-label">{emotion}</div>
                <div class="top3-conf-label">Confidence</div>
                <div class="top3-pct">{pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_bars(prob_df: pd.DataFrame, highlight: str | None = None) -> None:
    for _, row in prob_df.iterrows():
        emotion = row["Emosi"]
        pct = float(row["Persentase (%)"])
        color = EMOTION_COLORS.get(emotion, "#2563eb")
        weight = "700" if emotion == highlight else "500"
        st.markdown(
            f"""
            <div class="prob-row-label">
                <span style="font-weight:{weight}; text-transform:capitalize;">
                    {EMOTION_ICONS.get(emotion, "")} {emotion}
                </span>
                <span style="font-weight:650; color:#60a5fa;">{pct:.1f}%</span>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result_card(summary: dict) -> None:
    label = summary["top_label"]
    confidence = summary["top_pct"] / 100
    icon = EMOTION_ICONS.get(label, "🎭")
    accent = EMOTION_COLORS.get(label, "#60a5fa")

    margin_html = ""
    if summary["second_label"]:
        margin_html = (
            f'<div class="result-margin">+{summary["margin_pp"]:.1f} p.p. dari '
            f'{summary["second_label"]} (#2 · {summary["second_pct"]:.1f}%)</div>'
        )

    st.markdown(
        f"""
        <div class="result-card" style="border-color: {accent}44;">
            <div class="result-inner">
                <div class="result-dominance">Emosi Dominan · Tertinggi dari {summary["num_classes"]} kelas</div>
                <div class="result-emoji">{icon}</div>
                <p class="result-label">{label}</p>
                <div class="result-conf-label">Skor Tertinggi</div>
                <div class="result-confidence">{summary["top_pct"]:.1f}%</div>
                <p class="result-rank-note">{summary["separation"]}</p>
                {margin_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_model_ready(device_name: str) -> tuple[bool, str | None]:
    try:
        load_ser_model(str(MODEL_PATH), device_name)
        load_feature_extractor()
        return True, None
    except FileNotFoundError as exc:
        return False, str(exc)
    except RuntimeError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Gagal memuat model: {exc}"


def render_sidebar(device_name: str) -> None:
    model_ready, model_error = check_model_ready(device_name)
    device_icon = "🚀" if device_name == "cuda" else "💻"
    device_label = "GPU (CUDA)" if device_name == "cuda" else "CPU"

    status_class = "sidebar-status-ok" if model_ready else "sidebar-status-fail"
    dot_class = "ok" if model_ready else "fail"
    status_text = "Model Siap" if model_ready else "Model Gagal"
    status_sub = "Checkpoint berhasil dimuat" if model_ready else "Periksa file model"

    emotion_items = "".join(
        f'<div class="sidebar-emotion-item" style="--emotion-color:{EMOTION_COLORS[label]};">'
        f'<span class="sidebar-emotion-emoji">{EMOTION_ICONS[label]}</span>'
        f"<span>{label}</span>"
        f'<span class="sidebar-emotion-id">{idx}</span></div>'
        for idx in sorted(ID2LABEL)
        for label in [ID2LABEL[idx]]
    )

    sidebar_html = (
        f'<div class="sidebar-header">'
        f'<div class="sidebar-header-icon">🎙️</div>'
        f'<p class="sidebar-header-title">SER Assistant</p>'
        f'<p class="sidebar-header-sub">Deteksi emosi suara berbasis deep learning</p>'
        f"</div>"
        f'<div class="sidebar-stat-grid">'
        f'<div class="sidebar-stat-card"><div class="sidebar-stat-icon">🧠</div>'
        f'<div class="sidebar-stat-label">Backbone</div>'
        f'<div class="sidebar-stat-value">WavLM Base Plus</div></div>'
        f'<div class="sidebar-stat-card"><div class="sidebar-stat-icon">🎭</div>'
        f'<div class="sidebar-stat-label">Kelas</div>'
        f'<div class="sidebar-stat-value">6 Emosi</div></div>'
        f'<div class="sidebar-stat-card"><div class="sidebar-stat-icon">{device_icon}</div>'
        f'<div class="sidebar-stat-label">Perangkat</div>'
        f'<div class="sidebar-stat-value">{device_label}</div></div>'
        f'<div class="sidebar-stat-card"><div class="sidebar-stat-icon">📁</div>'
        f'<div class="sidebar-stat-label">Format</div>'
        f'<div class="sidebar-stat-value">.wav / .mp3</div></div>'
        f"</div>"
        f'<div class="sidebar-status-card {status_class}">'
        f'<div class="sidebar-status-dot {dot_class}"></div>'
        f"<div><div class=\"sidebar-status-text\">{status_text}</div>"
        f'<div class="sidebar-status-sub">{status_sub}</div></div>'
        f"</div>"
        f'<div class="sidebar-section-label">Peta Emosi</div>'
        f'<div class="sidebar-emotion-grid">{emotion_items}</div>'
    )

    with st.sidebar:
        st.markdown(sidebar_html, unsafe_allow_html=True)

        if not model_ready and model_error:
            st.error(model_error)

        with st.expander("⚙️ Detail Teknis"):
            st.caption(f"Backbone: {PRETRAINED_MODEL}")
            st.caption("Mode: inferensi saja (bukan training)")
            st.code(str(MODEL_PATH), language=None)


def run_prediction(uploaded_file, device_name: str) -> tuple[dict, dict]:
    with st.spinner("Menganalisis pola emosi dari audio..."):
        model, device = load_ser_model(str(MODEL_PATH), device_name)
        processor = load_feature_extractor()
        uploaded_file.seek(0)
        waveform, sample_rate = load_audio(uploaded_file)
        processed_waveform, preprocess_info = preprocess_audio(waveform, sample_rate)
        result = predict_emotion(model, processor, processed_waveform, device)
    return result, preprocess_info


def main() -> None:
    inject_custom_css()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    render_sidebar(device_name)
    render_hero()

    model_ready, model_error = check_model_ready(device_name)
    if not model_ready:
        st.error(
            "Model gagal dimuat sehingga prediksi tidak dapat dijalankan.\n\n"
            f"{model_error or 'Periksa file checkpoint di folder models/'}"
        )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-step">Langkah 1</div>
            <div class="section-title">Unggah Audio</div>
            <p class="section-desc">Gunakan file .wav atau .mp3 dengan suara yang jelas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Pilih file audio",
        type=["wav", "mp3"],
        label_visibility="collapsed",
        help="Format yang didukung: .wav dan .mp3",
    )

    if uploaded_file is None:
        render_empty_state()
        return

    display_name = Path(uploaded_file.name).name if uploaded_file.name else "unknown"

    try:
        uploaded_file.seek(0)
        audio_info = get_audio_info(uploaded_file)
        if display_name != "unknown":
            audio_info["filename"] = display_name
    except Exception:
        st.error(
            "File audio tidak dapat diproses. "
            "Coba gunakan file .wav atau .mp3 dengan durasi pendek dan kualitas suara jelas."
        )
        return

    render_metadata_card(
        filename=audio_info["filename"],
        duration_sec=audio_info["duration_sec"],
        sample_rate=audio_info["sample_rate"],
        channels=audio_info["channels"],
        file_size=format_file_size(getattr(uploaded_file, "size", None)),
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="section-step">Langkah 2</div>
            <div class="section-title">Pratinjau Audio</div>
            <p class="section-desc">Pastikan audio dapat diputar sebelum melakukan prediksi.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file.seek(0)
    st.audio(uploaded_file)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    predict_clicked = st.button(
        "Analisis Emosi",
        type="primary",
        use_container_width=True,
        disabled=not model_ready,
    )

    if not predict_clicked:
        return

    try:
        uploaded_file.seek(0)
        result, preprocess_info = run_prediction(uploaded_file, device_name)
    except FileNotFoundError as exc:
        st.error(f"File model tidak ditemukan.\n\n{exc}")
        return
    except RuntimeError as exc:
        error_text = str(exc)
        if "tidak cocok" in error_text.lower() or "checkpoint" in error_text.lower():
            st.error(f"Checkpoint tidak cocok dengan arsitektur model.\n\n{error_text}")
        elif "HuggingFace" in error_text or "pretrained" in error_text.lower():
            st.error(
                "Gagal memuat model HuggingFace. "
                "Periksa koneksi internet untuk unduhan pertama kali.\n\n"
                f"{error_text}"
            )
        else:
            st.error(f"Terjadi kesalahan saat memuat model.\n\n{error_text}")
        return
    except Exception:
        st.error(
            "File audio tidak dapat diproses. "
            "Coba gunakan file .wav atau .mp3 dengan durasi pendek dan kualitas suara jelas."
        )
        return

    st.markdown(
        """
        <div class="section-card">
            <div class="section-step">Langkah 3</div>
            <div class="section-title">Hasil Prediksi</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = summarize_prediction(result)
    render_result_card(summary)

    if preprocess_info["trimmed"]:
        st.warning(
            f"Audio dipotong menjadi maksimal {preprocess_info['max_duration_sec']} detik "
            f"untuk analisis emosi (durasi asli: {preprocess_info['original_duration_sec']} dtk). "
            "Transkrip tetap memakai audio penuh."
        )

    if ENABLE_STT:
        try:
            with st.spinner("Mentranskrip ucapan ke teks (Whisper)..."):
                asr = load_asr_pipeline(WHISPER_MODEL, device_name)
                uploaded_file.seek(0)
                stt_waveform = get_transcription_waveform(uploaded_file)
                transcript = transcribe_audio(asr, stt_waveform, WHISPER_LANGUAGE)
            render_transcript_card(transcript)
        except Exception as exc:
            st.info(
                "Transkrip teks tidak tersedia (model STT gagal dimuat atau audio tidak dapat "
                f"ditranskrip).\n\nDetail: {exc}"
            )

    st.markdown("#### Top 3 Emosi")
    render_top3_cards(result["probabilities_df"])

    st.markdown("#### Confidence Semua Kelas")
    render_probability_bars(result["probabilities_df"], highlight=result["predicted_label"])

    with st.expander("Detail Teknis"):
        st.markdown("**Probabilitas Semua Kelas**")
        display_df = result["probabilities_df"][["Emosi", "Persentase (%)"]].copy()
        display_df["Persentase (%)"] = display_df["Persentase (%)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("**Mapping Label**")
        mapping_df = pd.DataFrame(
            [{"ID": idx, "Label": label} for label, idx in sorted(LABEL2ID.items(), key=lambda x: x[1])]
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        st.markdown("**Raw Logits**")
        logits_df = pd.DataFrame(
            {
                "Emosi": [ID2LABEL[i] for i in range(len(result["logits"]))],
                "Logit": result["logits"],
            }
        )
        st.dataframe(logits_df, use_container_width=True, hide_index=True)

        st.markdown("**Probabilitas (Raw)**")
        prob_raw_df = pd.DataFrame(
            {
                "Emosi": [ID2LABEL[i] for i in range(len(result["probabilities"]))],
                "Probabilitas": result["probabilities"],
            }
        )
        st.dataframe(prob_raw_df, use_container_width=True, hide_index=True)

        st.markdown("**Path Model**")
        st.code(str(MODEL_PATH), language=None)


if __name__ == "__main__":
    main()
