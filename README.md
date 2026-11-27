# Tugas Besar IF5150 Project 2: Implementasi Algoritma Machine Learning

Klasifikasi status mahasiswa (Dropout, Enrolled, Graduate) dengan algoritma yang dibuat sendiri memakai numpy dan dibandingkan dengan scikit-learn. Kelompok 3.

## Struktur folder

```
.
├── README.md
├── doc/                      laporan (PDF)
└── src/
    ├── datasets/             train.csv, test.csv, sample_submission.csv, deskripsi variabel
    ├── preprocessing/        pipeline.py (build_dataset, tanpa kebocoran data)
    ├── models/               algoritma buatan sendiri (tanpa library ML)
    ├── eval_scratch/         skrip evaluasi model buatan sendiri
    ├── eval_library/         skrip evaluasi model scikit-learn sebagai pembanding
    ├── notebooks/            eksperimen: 01 preprocessing, 02 ANN, 03 Ensemble
    └── results/              model tersimpan (saved_models) dan file submission
```

## Instalasi

```
python -m pip install numpy pandas scikit-learn matplotlib
```

scikit-learn hanya dipakai untuk metrik evaluasi, pembagian data, dan model pembanding di `eval_library`. Algoritma di `models` tidak memakai library ML.

## Cara menjalankan

Semua perintah dijalankan dari dalam folder `src`:

```
cd src
python -m eval_scratch.main_decision_tree_learning
python -m eval_scratch.main_random_forest
python -m eval_scratch.main_knn
python -m eval_scratch.main_logistic_regression
python -m eval_scratch.main_ann
python -m eval_scratch.main_ensemble
python -m eval_library.ann_bawaan
python -m eval_library.ensemble_bawaan
python -m eval_library.dtl_bawaan
python -m eval_library.rf_bawaan
python -m eval_library.knn_bawaan
python -m eval_library.logreg_bawaan
```

Catatan:
- `main_ensemble` memuat lima model yang sudah disimpan di `results/saved_models` (ANN, Logistic Regression, Random Forest, Decision Tree, KNN), jadi jalankan skrip modelnya lebih dulu bila ingin melatih ulang. File ANN (`ann.npz`) dihasilkan oleh `eval_scratch.main_ann` (atau `notebooks/02_ann_model.ipynb`).
- Random Forest membutuhkan waktu beberapa menit karena ada pencarian hyperparameter.
- Skrip menyimpan model ke `results/saved_models` dan prediksi test ke `results/submission_*.csv`.

## Notebook

Notebook dibuat untuk Google Colab. Unggah berkas yang diminta pada Cell 1 (dataset, `pipeline.py`, dan file model), lalu jalankan sel berurutan.

## Pra-pemrosesan

`preprocessing/pipeline.py` menyediakan `build_dataset()`: pembersihan nilai, pembagian latih dan validasi (80/20, stratified), imputasi, pemotongan outlier (IQR), one-hot dan standardisasi, seleksi 20 fitur (mutual information), serta oversampling pada data latih. Semua statistik dipelajari dari data latih saja.

## Pembagian tugas

| Anggota | Tugas |
|---|---|
| Chatelia Dyah Prameswari | ANN, Ensemble, outlier, seleksi fitur, penanganan imbalance, skema validasi, analisis ANN dan Ensemble |
| Ishak | Decision Tree, Random Forest, normalisasi, one-hot, ordinal, analisis Decision Tree dan Random Forest |
| Eka Ardilah Febriyanti | KNN, Logistic Regression, missing values, standardisasi, tokenisasi, analisis KNN dan Logistic Regression |
