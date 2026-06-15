# Klasifikasi dan Segmentasi Penyakit Daun Cabai (Smart Farming)

Proyek ini adalah tugas akhir mata kuliah **Data Mining** dengan fokus bidang **Smart Farming**. Sistem ini mengintegrasikan dua pendekatan machine learning untuk membantu petani mendeteksi penyakit tanaman cabai secara dini dan mengukur tingkat kerusakan daun secara kuantitatif:
1. **Supervised Learning**: **Custom Convolutional Neural Network (CNN)** kustom untuk mengklasifikasikan citra daun cabai ke dalam 4 kategori kondisi.
2. **Unsupervised Learning**: **K-Means Clustering** pada ruang warna **HSV** untuk segmentasi lesi penyakit (*necrotic tissue*) dan mengestimasi persentase area kerusakan daun.

---

## 📌 Tautan Repositori Hugging Face

Karena keterbatasan kapasitas penyimpanan GitHub, file dataset besar dan file biner model diunggah ke Hugging Face:
* 🤗 **Dataset**: [edgararya/chili_disease.v12i.multiclass (Hugging Face Datasets)](https://huggingface.co/datasets/edgararya/chili_disease.v12i.multiclass)
* 🤗 **Model Weights (`.pth`)**: [edgararya/chili_model (Hugging Face Models)](https://huggingface.co/models/edgararya/chili_model)

---

## 🛠️ Struktur Proyek

```text
Tugas_Akhir/
├── latihan/
│   └── Daun-cabai-bolong-bolong.jpg  # Gambar contoh untuk pengujian
├── .gitignore                        # Konfigurasi file yang diabaikan Git
├── predict.py                        # Script inferensi mandiri (klasifikasi + segmentasi)
├── tugas_akhir.ipynb                 # Jupyter Notebook utama (pelatihan model & evaluasi)
├── laporan_ilmiah.md                 # Template laporan ilmiah berformat IMRAD
└── README.md                         # Dokumentasi proyek (file ini)
```

---

## 🔬 Metodologi dan Model

### 1. Klasifikasi Penyakit Daun (Supervised - Custom CNN)
Model dirancang dari nol (*from scratch*) dengan arsitektur PyTorch sebagai berikut:
* **Feature Extractor**: 4x Blok Convolutional Layers (Conv2D -> BatchNorm -> ReLU -> MaxPool2D) dengan peningkatan filter secara bertahap (32, 64, 128, 256).
* **Classifier**: Fully Connected Layers dengan regulasi Dropout (0.5 dan 0.3) untuk mencegah *overfitting*, menghasilkan output berupa skor probabilitas untuk 4 kelas:
  1. **Anthracnose** (Antraknosa / patek)
  2. **Healthy** (Daun sehat)
  3. **Leaf Curl** (Daun keriting/virus kuning)
  4. **Leaf Spot** (Bercak daun)

### 2. Segmentasi Kerusakan Daun (Unsupervised - K-Means Clustering)
Untuk mengukur tingkat keparahan infeksi daun, algoritme **K-Means Clustering** diimplementasikan dengan tahapan:
1. Mengonversi citra daun ke ruang warna **HSV** untuk memisahkan saluran warna secara optimal.
2. Melakukan *masking* daun hijau untuk membersihkan latar belakang (*background*).
3. Mengelompokkan piksel area daun menjadi **$K=3$ cluster warna** menggunakan K-Means.
4. Mengidentifikasi cluster bercak penyakit berdasarkan nilai komponen warna cokelat/kuning.
5. Menghitung tingkat keparahan infeksi menggunakan rumus:
   $$\text{Tingkat Kerusakan (\%)} = \frac{\text{Jumlah Piksel Cluster Penyakit}}{\text{Total Piksel Daun}} \times 100\%$$

---

## 🚀 Panduan Instalasi dan Penggunaan

### 1. Prasyarat (Prerequisites)
Pastikan Anda telah menginstal Python 3.8 ke atas, kemudian instal modul yang diperlukan:

```bash
pip install torch torchvision numpy opencv-python pillow huggingface_hub
```

### 2. Unduh Model Weights dari Hugging Face
Unduh bobot model (`chili_model.pth`) dari Hugging Face ke direktori utama proyek Anda menggunakan Hugging Face CLI:

```bash
huggingface-cli download edgararya/chili_model chili_model.pth --local-dir .
```

*Atau unduh secara langsung melalui tautan browser di repositori model Hugging Face dan simpan dengan nama `chili_model.pth`.*

### 3. Menjalankan Prediksi (Inferensi)
Gunakan file `predict.py` untuk menguji gambar baru. Masukkan gambar daun cabai yang ingin dideteksi dan sistem akan mengeluarkan output klasifikasi serta persentase kerusakan dalam format JSON.

**Contoh Perintah:**
```bash
python predict.py --image latihan/Daun-cabai-bolong-bolong.jpg --model chili_model.pth
```

**Contoh Output JSON:**
```json
{
    "status": "success",
    "image_path": "latihan/Daun-cabai-bolong-bolong.jpg",
    "prediction": "Leaf Spot",
    "confidence": 0.9421,
    "damage_severity_percent": 18.75,
    "class_probabilities": {
        "Anthracnose": 0.0210,
        "Healthy": 0.0012,
        "Leaf Curl": 0.0357,
        "Leaf Spot": 0.9421
    }
}
```

---

## 📊 Hasil dan Evaluasi
* Model dilatih menggunakan optimizer **Adam** selama 15 epoch.
* Evaluasi performa klasifikasi divisualisasikan menggunakan **Confusion Matrix** dan **Kurva Belajar** (Loss/Akurasi) yang dapat dieksekusi secara instan pada notebook `tugas_akhir.ipynb`.
* K-Means Clustering berhasil melakukan kuantifikasi area lesi secara adaptif tanpa membutuhkan label anotasi per-piksel.
