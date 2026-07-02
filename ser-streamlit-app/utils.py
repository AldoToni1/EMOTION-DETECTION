"""Utilitas audio preprocessing dan inferensi emosi."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from model import WavLMSERModel

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 4.0
MAX_SAMPLES = int(TARGET_SAMPLE_RATE * MAX_DURATION_SECONDS)

LABEL2ID = {
    "netral": 0,
    "senang": 1,
    "sedih": 2,
    "marah": 3,
    "takut": 4,
    "jijik": 5,
}

ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def _load_via_soundfile(path: str) -> tuple[torch.Tensor, int]:
    """Fallback decode WAV/FLAC via soundfile."""
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, int(sample_rate)


def _load_via_librosa(path: str) -> tuple[torch.Tensor, int]:
    """Fallback decode MP3/audio umum via librosa (lazy import)."""
    import librosa

    audio_np, sample_rate = librosa.load(path, sr=None, mono=False)
    waveform = torch.from_numpy(audio_np).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, int(sample_rate)


def _decode_audio_file(path: str) -> tuple[torch.Tensor, int]:
    """torchaudio → librosa → soundfile."""
    try:
        return torchaudio.load(path)
    except Exception:
        try:
            return _load_via_librosa(path)
        except Exception:
            return _load_via_soundfile(path)


def load_audio(file: io.BytesIO | str | Path) -> tuple[torch.Tensor, int]:
    """Muat audio dari file upload Streamlit (.wav / .mp3)."""
    if isinstance(file, (str, Path)):
        path = str(file)
        waveform, sample_rate = _decode_audio_file(path)
    else:
        suffix = ".wav"
        if hasattr(file, "name") and file.name:
            suffix = Path(file.name).suffix or suffix

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        try:
            waveform, sample_rate = _decode_audio_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, int(sample_rate)


def preprocess_audio(
    audio: torch.Tensor,
    sample_rate: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Konversi ke mono 16 kHz dan potong/pad maksimal 6 detik."""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    original_duration = audio.shape[-1] / sample_rate

    if sample_rate != TARGET_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, TARGET_SAMPLE_RATE)

    if audio.shape[-1] > MAX_SAMPLES:
        audio = audio[..., :MAX_SAMPLES]
        trimmed = True
    else:
        trimmed = original_duration > MAX_DURATION_SECONDS

    duration_after = audio.shape[-1] / TARGET_SAMPLE_RATE
    info = {
        "original_sample_rate": sample_rate,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "original_duration_sec": round(original_duration, 3),
        "processed_duration_sec": round(duration_after, 3),
        "trimmed": trimmed,
        "max_duration_sec": MAX_DURATION_SECONDS,
    }
    return audio.squeeze(0), info


def get_audio_info(file: io.BytesIO | str | Path) -> dict[str, Any]:
    """Ambil metadata audio tanpa preprocessing penuh."""
    if hasattr(file, "name") and file.name:
        filename = Path(file.name).name
    elif isinstance(file, (str, Path)):
        filename = Path(file).name
    else:
        filename = "unknown"
    waveform, sample_rate = load_audio(file)
    duration_sec = round(waveform.shape[-1] / sample_rate, 3)
    channels = 1 if waveform.ndim == 1 else waveform.shape[0]

    return {
        "filename": filename,
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "channels": channels,
        "num_samples": int(waveform.shape[-1]),
    }


def get_transcription_waveform(file: io.BytesIO | str | Path) -> "np.ndarray":
    """Ambil waveform mono 16 kHz PENUH (tanpa potong) untuk transkrip STT."""
    waveform, sample_rate = load_audio(file)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    return waveform.squeeze(0).numpy().astype("float32")


def transcribe_audio(asr_pipeline: Any, waveform: "np.ndarray", language: str = "indonesian") -> str:
    """Transkrip audio ke teks menggunakan pipeline Whisper (ASR)."""
    output = asr_pipeline(
        {"raw": waveform, "sampling_rate": TARGET_SAMPLE_RATE},
        generate_kwargs={"language": language, "task": "transcribe"},
        chunk_length_s=30,
    )
    return str(output.get("text", "")).strip()


STT_FALLBACK_MESSAGE = "Transkripsi sementara tidak tersedia. Silakan coba lagi nanti."

_whisper_pipeline: Any | None = None
_whisper_config_key: tuple[str, str] | None = None


def create_whisper_pipeline(model_name: str, device_name: str) -> Any:
    """Buat pipeline Whisper ASR — hanya dipanggil on-demand, bukan saat startup."""
    from transformers import pipeline

    device = 0 if device_name == "cuda" else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )


def get_whisper_pipeline(
    model_name: str,
    device_name: str,
    *,
    loader: Any | None = None,
) -> Any:
    """Lazy-load pipeline Whisper; pertama kali dipanggil saat user minta transkrip."""
    global _whisper_pipeline, _whisper_config_key

    if loader is not None:
        return loader(model_name, device_name)

    key = (model_name, device_name)
    if _whisper_pipeline is None or _whisper_config_key != key:
        _whisper_pipeline = create_whisper_pipeline(model_name, device_name)
        _whisper_config_key = key
    return _whisper_pipeline


def safe_transcribe(
    file: io.BytesIO | str | Path,
    model_name: str,
    device_name: str,
    language: str = "indonesian",
    *,
    pipeline_loader: Any | None = None,
) -> tuple[str, bool]:
    """Transkrip audio ke teks. Tidak pernah raise; mengembalikan (teks, sukses)."""
    try:
        if hasattr(file, "seek"):
            file.seek(0)
        waveform = get_transcription_waveform(file)
        asr = get_whisper_pipeline(model_name, device_name, loader=pipeline_loader)
        text = transcribe_audio(asr, waveform, language)
        cleaned = str(text).strip()
        if not cleaned:
            return "Tidak ada ucapan yang terdeteksi.", True
        return cleaned, True
    except Exception:
        return STT_FALLBACK_MESSAGE, False


def predict_emotion(
    model: WavLMSERModel,
    processor: Any,
    waveform: torch.Tensor,
    device: torch.device | str,
) -> dict[str, Any]:
    """Jalankan inferensi emosi pada waveform 1D yang sudah dipreprocess."""
    device = torch.device(device)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask)

    probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    predicted_id = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_id])

    prob_df = pd.DataFrame(
        {
            "Emosi": [ID2LABEL[i] for i in range(len(probabilities))],
            "Probabilitas": probabilities,
            "Persentase (%)": probabilities * 100,
        }
    ).sort_values("Probabilitas", ascending=False)

    return {
        "predicted_label": ID2LABEL[predicted_id],
        "predicted_id": predicted_id,
        "confidence": confidence,
        "probabilities": probabilities,
        "logits": logits.squeeze(0).cpu().numpy(),
        "probabilities_df": prob_df,
    }
