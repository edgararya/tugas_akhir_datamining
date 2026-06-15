# Klasifikasi dan Segmentasi Penyakit Daun Cabai (Smart Farming)

Proyek ini bertujuan untuk mengintegrasikan dua pendekatan data mining untuk deteksi penyakit pada daun tanaman cabai:
1. **Supervised Learning**: **Custom Convolutional Neural Network (CNN)** untuk mengklasifikasikan kondisi daun cabai ke dalam 4 kategori (Anthracnose, Leaf Curl, Leaf Spot, Healthy).
2. **Unsupervised Learning**: **K-Means Clustering** pada ruang warna **HSV** untuk segmentasi lesi penyakit dan mengukur persentase area kerusakan daun.

---

## 📌 Tautan Repositori Hugging Face

* 🤗 **Dataset**: [edgararya/chili_disease.v12i.multiclass (Hugging Face Datasets)](https://huggingface.co/datasets/edgararya/chili_disease.v12i.multiclass)
* 🤗 **Model Weights (`.pth`)**: [edgararya/chili_model (Hugging Face Models)](https://huggingface.co/models/edgararya/chili_model)

---

## 🛠️ Dependensi

Untuk menjalankan proyek ini, instal pustaka Python berikut:

```bash
pip install torch torchvision numpy opencv-python pillow huggingface_hub
```

---

## 🔬 Metodologi dan Model

### 1. Klasifikasi Penyakit Daun (Supervised - Custom CNN)
Model dirancang dari nol (*from scratch*) menggunakan kerangka kerja PyTorch dengan arsitektur:
* **Feature Extractor**: 4x Blok Convolutional Layers (Conv2D -> BatchNorm -> ReLU -> MaxPool2D) dengan filter bertahap (32, 64, 128, 256).
* **Classifier**: Fully Connected Layers dengan regulasi Dropout (0.5 dan 0.3) untuk mencegah *overfitting*, menghasilkan output berupa skor probabilitas untuk 4 kelas:
  1. **Anthracnose** (Antraknosa / patek)
  2. **Healthy** (Daun sehat)
  3. **Leaf Curl** (Daun keriting/virus kuning)
  4. **Leaf Spot** (Bercak daun)

### 2. Segmentasi Kerusakan Daun (Unsupervised - K-Means Clustering)
Untuk mengukur tingkat keparahan infeksi daun, algoritme **K-Means Clustering** diterapkan pada citra daun:
1. Mengonversi citra daun ke ruang warna **HSV** untuk meminimalkan pengaruh pencahayaan.
2. Melakukan *masking* daun hijau untuk memisahkan objek daun dari latar belakang (*background*).
3. Mengelompokkan piksel area daun menjadi **$K=3$ cluster warna** menggunakan K-Means.
4. Mengidentifikasi cluster bercak penyakit berdasarkan nilai komponen warna cokelat/kuning.
5. Menghitung persentase tingkat keparahan kerusakan daun dengan rumus:
   $$\text{Tingkat Kerusakan (\%)} = \frac{\text{Jumlah Piksel Cluster Penyakit}}{\text{Total Piksel Daun}} \times 100\%$$
