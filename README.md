### 1. **Persiapan Dataset**

Pada tahap pertama, kita mempersiapkan dataset yang akan digunakan untuk pelatihan model. Dataset yang digunakan adalah **DoS UDP Flood** dan **DoS ICMP Flood**, yang masing-masing berisi data serangan denial of service pada jaringan.

```python
import pandas as pd
dataset1 = pd.read_csv('DoS UDP Flood.csv')
dataset2 = pd.read_csv('DoS ICMP Flood.csv')
datasets = pd.concat([dataset1, dataset2], ignore_index=True)
```

**Penjelasan**:

* Dataset pertama dan kedua dibaca menggunakan `pd.read_csv()` dan kemudian digabungkan menjadi satu dataset besar dengan `pd.concat()`.
* Parameter `ignore_index=True` digunakan untuk memastikan bahwa indeks data yang digabungkan tetap berurutan.

---

### 2. **Menentukan Fitur dan Target**

Setelah dataset digabungkan, kita memilih fitur yang akan digunakan untuk melatih model serta target variabel yang akan diprediksi. Dalam hal ini, fitur dipilih dari kolom 7 hingga 75, dan target variabel adalah nama serangan pada kolom `Attack Name`.

```python
x = datasets.iloc[:,7:76]  # Fitur
y = datasets['Attack Name']  # Target
```

**Penjelasan**:

* `x` berisi fitur yang akan digunakan untuk melatih model. Kolom yang dipilih dari indeks 7 hingga 75 mencakup informasi teknis mengenai trafik jaringan.
* `y` berisi target yang merupakan jenis serangan yang terjadi pada jaringan (Attack Name).

---

### 3. **Pembagian Data Latih dan Uji**

Setelah fitur dan target dipersiapkan, data dibagi menjadi dua bagian: data latih (training) dan data uji (testing). Pembagian dilakukan dengan proporsi 80% data untuk pelatihan dan 20% untuk pengujian.

```python
from sklearn.model_selection import train_test_split
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

**Penjelasan**:

* `train_test_split()` digunakan untuk membagi data secara acak menjadi dua bagian, di mana 80% digunakan untuk melatih model (`x_train` dan `y_train`), dan 20% sisanya digunakan untuk menguji model (`X_test` dan `y_test`).

---

### 4. **Training Model (dengan Penjelasan Rinci)**

Pada tahap ini, beberapa model klasifikasi dilatih menggunakan data latih, dan hasil prediksi diuji terhadap data uji. Kita menggunakan empat model: Decision Tree, Random Forest, KNN, dan SVM.

```python
# Import library yang dibutuhkan
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

**Penjelasan**:

* Kita mengimpor model klasifikasi yang akan digunakan dari `scikit-learn`.
* `accuracy_score` digunakan untuk mengukur seberapa akurat prediksi model terhadap data uji.

```python
# Mendefinisikan model-model yang akan dilatih
models = {
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC()
}
```

**Penjelasan**:

* Kita membuat dictionary `models` yang berisi nama model sebagai key dan objek model sebagai value.
* `DecisionTreeClassifier` menggunakan metode entropy untuk menghitung informasi dan pemilihan split acak.
* `RandomForestClassifier` menggunakan 100 pohon keputusan.
* `KNeighborsClassifier` mencari 5 tetangga terdekat.
* `SVC()` adalah Support Vector Machine dengan parameter default.

```python
# Menyimpan hasil akurasi tiap model
accuracies = {}
```

**Penjelasan**:

* Dictionary kosong ini akan diisi dengan hasil akurasi dari setiap model setelah pelatihan.

```python
# Melatih dan mengevaluasi setiap model
for name, model in models.items():
    model.fit(x_train, y_train)              # Melatih model
    y_pred = model.predict(X_test)           # Menggunakan model untuk prediksi data uji
    acc = accuracy_score(y_test, y_pred)     # Menghitung akurasi prediksi
    accuracies[name] = acc                   # Menyimpan akurasi ke dictionary
    print(f"{name} Accuracy: {acc * 100:.2f}%")  # Menampilkan akurasi
```

**Penjelasan**:

* `fit()` digunakan untuk melatih model dengan data latih (`x_train`, `y_train`).
* `predict()` menghasilkan prediksi label berdasarkan data uji (`X_test`).
* `accuracy_score()` membandingkan label prediksi dengan label asli (`y_test`).
* Akurasi disimpan dalam dictionary `accuracies` dan dicetak dalam bentuk persentase.

---

### 5. **Visualisasi Perbandingan Akurasi Model**

Setelah melatih beberapa model, kita membandingkan akurasi masing-masing model dengan membuat diagram batang yang menggambarkan akurasi tiap model.

```python
import seaborn as sns
import matplotlib.pyplot as plt

model_names = ['Decision Tree', 'Random Forest', 'KNN', 'SVM']
accuracies = [75.98, 81.82, 72.82, 60.77]

df_accuracy = pd.DataFrame({
    'Model': model_names,
    'Accuracy (%)': accuracies
})

plt.figure(figsize=(10, 6))
sns.barplot(data=df_accuracy, x='Model', y='Accuracy (%)', hue='Model', palette='viridis', legend=False)

plt.ylim(0, 100)
plt.title('Perbandingan Akurasi Model')
plt.ylabel('Akurasi (%)')
plt.xlabel('Model Klasifikasi')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

**Penjelasan**:

* Dengan menggunakan `seaborn` dan `matplotlib`, kita membuat barplot untuk memvisualisasikan akurasi yang diperoleh oleh masing-masing model.
* Diagram ini memudahkan untuk membandingkan akurasi tiap model secara visual.

---

### 6. **Visualisasi Confusion Matrix untuk Random Forest**

Setelah model dipilih berdasarkan akurasi tertinggi (Random Forest), kita melakukan visualisasi **Confusion Matrix** untuk model ini. Confusion Matrix menunjukkan bagaimana model membuat prediksi terhadap setiap kelas.

```python
from sklearn.metrics import confusion_matrix

# Prediksi ulang khusus untuk Random Forest agar y_pred sesuai
best_model = models['Random Forest']
y_pred = best_model.predict(X_test)

# Membuat label unik dari target
labels = y.unique()

# Membuat confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

# Visualisasi heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Prediksi')
plt.ylabel('Fakta')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.show()
```

**Penjelasan**:

* Kita mengambil kembali model Random Forest dari dictionary dan melakukan prediksi ulang.
* `confusion_matrix()` digunakan untuk membuat matriks berdasarkan label sebenarnya dan prediksi.
* Hasilnya divisualisasikan dengan heatmap menggunakan `seaborn`.
* Matriks ini memberikan informasi jumlah prediksi benar dan salah untuk tiap kelas.

---
