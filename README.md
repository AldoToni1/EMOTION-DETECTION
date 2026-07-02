# EMOTION-DETECT

Proyek **Speech Emotion Recognition (SER)** berbasis WavLM + aplikasi Streamlit dengan **Whisper STT**.

## Struktur

```
EMOTION-DETECT/
├── requirements.txt          # Dependensi project
├── ser-streamlit-app/        # Aplikasi inference (Streamlit)
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│   └── models/
│       └── ser_wavlm_v7_best.pt
└── README.md
```

## Instalasi & menjalankan

```bash
pip install -r requirements.txt
cd ser-streamlit-app
streamlit run app.py
```

Buka: http://localhost:8501

## Model aktif

- **Checkpoint:** `ser-streamlit-app/models/ser_wavlm_v7_best.pt`
- **Backbone:** `microsoft/wavlm-base-plus`
- **Test accuracy (v7):** ~77.5%
- **STT:** Whisper `openai/whisper-small` (via transformers)
