
# SER WavLM Model Package

Model        : microsoft/wavlm-base-plus
Best epoch   : 15
Best val acc : 73.98%
Test acc     : 74.42%
Labels       : netral, senang, sedih, marah, takut, jijik

## File utama
- `ser_wavlm_best.pt` : checkpoint model terbaik
- `config_wavlm.json` : konfigurasi training dan label
- `feature_extractor/` : konfigurasi HuggingFace feature extractor
- `metadata_split_wavlm.csv` : split train/val/test
- `evaluasi_test_wavlm.csv` : hasil prediksi test set
