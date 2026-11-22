# Project 2 IF5150 Inteligensi Artifisial Lanjut (Kelompok 3)

Implementasi algoritma Machine Learning dari nol (hanya `numpy`) dan perbandingannya dengan `scikit-learn` pada dataset performa akademik mahasiswa. Tugas klasifikasi multikelas pada kolom `Target` dengan tiga kelas: Dropout, Enrolled, dan Graduate.

## Anggota

| Nama | NIM |
|---|---|
| Chatelia Dyah Prameswari | 23526004 |
| Ishak | 23526025 |
| Eka Ardilah Febriyanti | 23526038 |

## Dataset

Dataset terdiri dari `train.csv` (3.096 baris), `test.csv` (1.328 baris), dan `sample_submission.csv`, dengan 36 fitur demografis, sosio-ekonomi, dan akademik. Deskripsi variabel tersedia pada `variables_description.csv` dan `dataset_description.pdf`. Kelas target tidak seimbang: Graduate 49,9%, Dropout 32,1%, dan Enrolled 18,0%.

## Struktur repositori

```
README.md
requirements.txt
src/
  config.py                 konfigurasi bersama (seed, pemetaan label, pengelompokan kolom)
  preprocessing.py          kelas dan fungsi preprocessing
  notebooks/
    01_preprocessing_pipeline.ipynb
artifacts/                  hasil pipeline (split, fitur terpilih, data praproses)
doc/                        laporan akhir (PDF) dan gambar pendukung
```

## Instalasi

```
pip install -r requirements.txt
```

## Menjalankan

Notebook pipeline preprocessing:

```
jupyter notebook src/notebooks/01_preprocessing_pipeline.ipynb
```

Pemakaian dari Python:

```python
from src.preprocessing import prepare_data, load_preprocessed

data = prepare_data()                       # split, clipping, encoding, seleksi fitur
X_train, y_train = data.X_train_resampled, data.y_train_resampled
X_val, y_val = data.X_val.values, data.y_val.values

arrays = load_preprocessed("artifacts/preprocessed_v1.npz")   # hasil yang sudah tersimpan
```

## Pipeline preprocessing

Seluruh langkah di-`fit` hanya pada data train, dan resampling hanya diterapkan pada data train.

1. Pembersihan nama kolom dan audit kualitas data (missing value, duplikat, nilai tidak valid).
2. Split stratified 80/20 (`random_state=42`) menjadi train dan validasi. `test.csv` tidak digunakan pada tahap ini.
3. Outlier clipping berbasis IQR (pengali 3). Kolom nilai rata-rata dipotong dengan batas dari nilai tidak nol, dan kolom dengan IQR nol tidak dipotong.
4. Encoding one-hot untuk kolom nominal dan standardisasi untuk kolom numerik dan ordinal.
5. Seleksi fitur bertahap: dukungan minimum, mutual information, penyaringan redundansi (korelasi 0,95), dan pemangkasan ke 20 fitur.
6. Random oversampling pada data train agar ketiga kelas berjumlah sama.

## Pembagian tugas

**Task 1: Data Preprocessing**

| Anggota | Komponen |
|---|---|
| Chatelia | Outlier detection, feature selection |
| Ishak | Data normalization, one-hot encoding, ordinal encoding |
| Eka | Missing value handling, data standardization, text tokenization |

**Task 2: Modeling (from scratch dan scikit-learn)**

| Anggota | Algoritma |
|---|---|
| Chatelia | Artificial Neural Network (ANN), Ensemble (gabungan lima metode) |
| Ishak | Decision Tree Learning (DTL), Random Forest |
| Eka | K-Nearest Neighbors (KNN), Logistic Regression |

Dataset ini tidak memiliki kolom teks, sehingga tahap text tokenization tidak berlaku.
