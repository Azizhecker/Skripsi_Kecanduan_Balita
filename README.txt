PAKET REVISI WEBSITE - SKRIPSI KECANDUAN BALITA

Isi utama:
- app.py
- train.py
- templates/
- static/css/style.css
- tabulasi_data_terbaru(1).xlsx
- model_random_forest_150_110_40.pkl
- Hasil_RF_150_Data_110_40_Analisis_Lokasi.xlsx
- hasil_model_150_110_40.json

KONFIGURASI PENELITIAN
- 150 responden
- 62 dekat kota
- 88 jauh dari kota
- 110 data training
- 40 data testing
- 100 Decision Tree
- Random State 42
- Fitur: Skor_X1, Skor_X2, Skor_X3, Skor_X4

CARA PASANG DI PROJECT VS CODE
1. Backup folder project lama.
2. Salin app.py ke folder project.
3. Salin train.py ke folder project.
4. Salin folder templates dan static ke folder project, replace file template/CSS yang lama.
5. Salin model_random_forest_150_110_40.pkl dan tabulasi_data_terbaru(1).xlsx ke folder project.
6. Aktifkan venv:
   .\.venv\Scripts\activate
7. Install dependency bila diperlukan:
   pip install flask pandas numpy openpyxl joblib matplotlib scikit-learn==1.3.1
8. Jalankan training:
   python train.py
9. Jalankan website:
   python app.py
10. Buka:
   http://127.0.0.1:5000/dashboard

HALAMAN PENTING
- /dashboard
- /analisis-lokasi
- /tabulasi
- /skor-target
- /excel
- /diagnosa
- /feature-importance
- /decision-tree?tree=1 sampai tree=100
- /login

CATATAN
- Jangan mengganti nama model_random_forest_150_110_40.pkl jika app.py masih memakai nama tersebut.
- LABEL pada Excel penelitian dipertahankan. Kode tidak membuat LABEL baru jika LABEL sudah tersedia.
- Kategori TINGGI ditampilkan sebagai kategori target tinggi pada analisis lokasi.
