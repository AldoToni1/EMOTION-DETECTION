# Speech Emotion Recognition — Streamlit App

Aplikasi web berbasis **Streamlit** untuk mendeteksi emosi dari file audio menggunakan model **WavLM** (`microsoft/wavlm-base-plus`) yang sudah dilatih sebelumnya.

Aplikasi ini **hanya melakukan inferensi/prediksi** — tidak ada proses training, dataset, atau unduhan data pelatihan.

## Fitur

- Upload file audio `.wav` atau `.mp3`
- Prediksi 6 kelas emosi: netral, senang, sedih, marah, takut, jijik
- Confidence score dan probabilitas per kelas
- Bar chart visualisasi probabilitas
- **Transkrip teks (Speech-to-Text)** dari audio menggunakan Whisper
- Informasi audio: nama file, durasi, sample rate, jumlah kanal

## Speech-to-Text (STT)

- Menggunakan **Whisper** (`openai/whisper-small`) via `transformers`, **tanpa perlu FFmpeg**
  (audio sudah di-decode oleh `torchaudio`/`librosa` lalu disuapkan langsung sebagai waveform).
- Transkrip memakai **audio penuh** (bukan potongan 4 detik yang dipakai untuk analisis emosi).
- Bahasa transkrip default: Indonesia (ubah `WHISPER_LANGUAGE` di `app.py`).
- Untuk CPU yang lambat, ganti `WHISPER_MODEL` ke `"openai/whisper-base"` agar lebih ringan
  (akurasi transkrip sedikit menurun). Set `ENABLE_STT = False` untuk menonaktifkan STT.
- Unduhan model Whisper terjadi sekali di awal (butuh internet), setelah itu berjalan lokal.

## Instalasi

```bash
cd ser-streamlit-app
pip install -r requirements.txt
```

> **Catatan:** Unduhan pertama kali akan memuat backbone WavLM dari HuggingFace (`microsoft/wavlm-base-plus`). Pastikan koneksi internet tersedia.

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser (biasanya `http://localhost:8501`).

## Struktur Folder

```
ser-streamlit-app/
│
├── app.py                  # Aplikasi Streamlit utama
├── model.py                # Arsitektur model & load checkpoint
├── utils.py                # Preprocessing audio & inferensi
├── requirements.txt        # Dependensi Python
├── README.md
│
├── models/
│   └── ser_wavlm_v7_best.pt   # Checkpoint model terlatih (v7)
│
└── assets/
    └── sample_audio/       # (Opsional) contoh file audio uji
```

## Mengganti File Model

1. Letakkan checkpoint baru di folder `models/`.
2. Ubah variabel `MODEL_PATH` di `app.py`:

```python
MODEL_PATH = BASE_DIR / "models" / "nama_checkpoint_baru.pt"
```

Checkpoint yang didukung:
- State dict langsung
- Dictionary dengan key `model_state_dict`
- Dictionary dengan key `model_state`

## Spesifikasi Inferensi

| Parameter | Nilai |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Kanal | Mono |
| Durasi maksimum | 8 detik (dipotong jika lebih panjang) |
| Backbone | `microsoft/wavlm-base-plus` |
| Jumlah kelas | 6 |

## Label Emosi

| ID | Label |
|----|-------|
| 0  | netral |
| 1  | senang |
| 2  | sedih |
| 3  | marah |
| 4  | takut |
| 5  | jijik |

## Catatan Teknis

- Aplikasi berjalan di **CPU** maupun **GPU (CUDA)**. CPU lebih lambat namun tetap fungsional.
- Semua path menggunakan **relative path** sehingga dapat dijalankan secara lokal.
- Model checkpoint: `models/ser_wavlm_v7_best.pt` (ubah `MODEL_PATH` di `app.py` jika perlu).
