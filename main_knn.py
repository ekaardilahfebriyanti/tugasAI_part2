import pandas as pd
from preprocessing_pipeline.preprocessing import Preprocessing
from models.knn import KNN

prep = Preprocessing("src/train.csv")
student_clean = prep.cleaning_student_dataset()
X, y = prep.sparating_dataset(student_clean)
X_train, X_test, y_train, y_test = prep.splitting_dataset(X, y)

# menyeleksi fitur berdasarkan korelasi dg threshold = 0.15
X_train_sel, X_test_sel, corr_scores = prep.select_features_by_correlation(
    X_train, X_test, y_train, threshold=0.15
)

# menstandarisasi hanya pada fitur yg sudah terpilih
X_train_scaled, X_test_scaled = prep.scaling_dataset(X_train_sel, X_test_sel)

final_modelknn_eu = KNN(k=5, metric="euclidean")
# menyimpan fitur data latih yg sudah distandarisasi (X_train_scaled) beserta label targetnya (y_train) ke dalam memori model Euclidean sbg acuan pencocokan.
final_modelknn_eu.fit(X_train_scaled, y_train)

final_modelknn_man = KNN(k=5, metric="manhattan")
final_modelknn_man.fit(X_train_scaled, y_train)

# membaca file test.csv, membersihkan nilai yg tidak valid, mengambil kolom fitur terpilih, dan menyamakan skalanya berdasarkan parameter data latih.
student_ids, X_submission_scaled = prep.prepare_test_data(
    '/content/drive/My Drive/project2AILanjut/test.csv'
)

hasil_prediksi_knn_eu = final_modelknn_eu.predict(X_submission_scaled)
hasil_prediksi_knn_man = final_modelknn_man.predict(X_submission_scaled)

# menggabungkan Student_ID dan hasil prediksi ke dlm DataFrame
df_hasil = pd.DataFrame({
    "Student_ID": student_ids,
    "Target_kNN_Euclidean": hasil_prediksi_knn_eu,
    "Target_kNN_Manhattan": hasil_prediksi_knn_man,
})

print("\n5 hasil prediksi pada test.csv:")
print(df_hasil.head())

# jika ingin menyimpan ke file CSV
# df_hasil.to_csv('/content/drive/My Drive/project2AILanjut/hasil_prediksi.csv', index=False)
# print("\nFile hasil_prediksi.csv berhasil disimpan!")