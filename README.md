Flask TFT API scaffold

- Endpoints
  - GET /health
  - GET /versions
  - GET / (UI)
  - POST /predict

- UI
  - templates/index.html, static/app.js
- Model loading
  - Loads checkpoints from `Source Code Model/models/version_*/checkpoints`
- Notes
  - Inference currently placeholder. Replace with real TimeSeriesDataSet inference when ready.

## Struktur folder

- `Dataset/` — dataset utama dan contoh CSV
- `Deskripsi Parameter dari Teknisi/` — dokumentasi library dan versi
- `Hasil Labeling Data/` — ruang untuk hasil pelabelan data
- `Hasil Validasi Teknisi (SME)/` — hasil pengujian/validasi
- `Source Code Model/` — notebook, script training, model, dan log eksperimen
- `Source Code Flask/` — source code aplikasi Flask, UI, dan pengujian

Source aplikasi Flask berada di `Source Code Flask/`.
