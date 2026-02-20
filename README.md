# Sistem Autentikasi Wajah Menggunakan Metode LBPH

Aplikasi sistem keamanan berbasis pengenalan wajah menggunakan metode **Local Binary Pattern Histogram (LBPH)** dengan deteksi wajah menggunakan **Haar Cascade Classifier** dari OpenCV.

## Fitur Utama

-  Pengambilan Dataset Wajah (dengan countdown & kontrol mulai)
-  Pelatihan Model LBPH
-  Autentikasi Wajah (Akses Diterima / Ditolak)
-  Evaluasi Sistem (Confusion Matrix, Accuracy, Precision, Recall, F1-Score)
-  Export Log Akses ke Excel
-  Database SQLite untuk penyimpanan data mahasiswa & log akses

---

##  Teknologi yang Digunakan

- Python 3.x
- Streamlit
- OpenCV
- LBPH Face Recognizer
- Haar Cascade Classifier
- SQLite
- Pandas
- OpenPyXL

---

##  Struktur Project
project/
 app.py
 haarcascade_frontalface_default.xml

## Cara Menjalankan Aplikasi

### Install Dependency
pip install streamlit opencv-contrib-python numpy pillow pandas openpyxl


## Cara Penggunaan
1. Ambil Dataset
Isi NIM, Nama, Kelas
Tekan tombol simpan
Tekan Q untuk mulai pengambilan foto
Sistem mengambil 20 gambar wajah

2️. Latih Model
Klik "Latih Model"
Sistem akan membuat file trained_model.yml

3️. Autentikasi
Scan wajah melalui kamera
Sistem menampilkan:
AKSES DITERIMA
AKSES DITOLAK

4️. Evaluasi
Melihat log akses
Confusion Matrix
Accuracy
Precision
Recall
F1-Score
Download log dalam format Excel

## Metode Evaluasi
Evaluasi sistem menggunakan Confusion Matrix:
True Positive (TP)
False Positive (FP)
True Negative (TN)
False Negative (FN)

Rumus evaluasi:
Accuracy = (TP + TN) / Total Data
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)


## Sumber Model Deteksi Wajah
Haar Cascade Classifier diperoleh dari repository resmi OpenCV:
https://github.com/opencv/opencv/tree/master/data/haarcascades

## Tujuan Pengembangan
Aplikasi ini dikembangkan sebagai implementasi sistem autentikasi berbasis pengenalan wajah untuk meningkatkan keamanan akses menggunakan metode LBPH.


## Author
Muhammad Alfikrah
Politeknik Negeri Samarinda
Tahun 2026
