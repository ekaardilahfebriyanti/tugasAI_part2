import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Preprocessing:
    def __init__(self, data_train="src/train.csv"):
        self.student = pd.read_csv(data_train)
        self.selected_features = None
        self.scaler = None
        self.impute_values = {}

    def _clean_dataframe(self, df, is_train=True):
        data = df.copy()

        # merapikan nama kolom dan mengubah teks kosong/aneh menjadi NaN
        data.columns = data.columns.str.strip()
        placeholder_kosong = ["?", "-", "na", "n/a", "null", "none", ""]

        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].str.strip()
            is_invalid = data[col].str.lower().isin(placeholder_kosong)
            data.loc[is_invalid, col] = np.nan

        # menghapus duplikat hanya pd data train
        if is_train:
            data = data.drop_duplicates().reset_index(drop=True)

        # 3. memvalidasi domain nilai akademik
        data.loc[data["Application order"] < 1, "Application order"] = np.nan
        data["Application order"] = (
            data["Application order"]
            .fillna(data["Application order"].mode()[0])
            .astype(int)
        )

        for col in ["Previous qualification (grade)", "Admission grade"]:
            data.loc[(data[col] < 0) | (data[col] > 200), col] = np.nan

        for sem in ["1st sem", "2nd sem"]:
            grade_col = f"Curricular units {sem} (grade)"
            enr_col = f"Curricular units {sem} (enrolled)"
            app_col = f"Curricular units {sem} (approved)"

            data.loc[(data[grade_col] < 0) | (data[grade_col] > 20), grade_col] = np.nan
            data.loc[data[app_col] > data[enr_col], app_col] = np.nan

        # 4. mengisi missing values menggunakan nilai median/modus dr data train
        for col in data.columns:
            if col == "Target":
                continue

            if is_train:
                if pd.api.types.is_numeric_dtype(data[col]):
                    self.impute_values[col] = data[col].median()
                else:
                    self.impute_values[col] = data[col].mode()[0]

            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(self.impute_values[col])

        return data

    def cleaning_student_dataset(self):
        student = self.student.copy()
        print(f"Ukuran awal: {student.shape}")

        if "Student_ID" in student.columns:
            student = student.drop(columns=["Student_ID"])

        student = self._clean_dataframe(student, is_train=True)

        print(f"Ukuran akhir siap analisis: {student.shape}")
        print(f"Total missing values akhir: {student.isnull().sum().sum()}")
        return student

    def sparating_dataset(self, studentcl):
        X = studentcl.drop(columns=["Target"])
        y = studentcl["Target"]
        return X, y

    def splitting_dataset(self, X, y):
        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def select_features_by_correlation(self, X_train, X_test, y_train, threshold=0.05):
        y_encoded = pd.Series(
            LabelEncoder().fit_transform(y_train),
            index=X_train.index,
            name="Target"
        )

        df_train = pd.concat([X_train, y_encoded], axis=1)
        corr_target = df_train.corr(numeric_only=True)["Target"].drop("Target").abs().fillna(0)

        self.selected_features = corr_target[corr_target >= threshold].index.tolist()
        if len(self.selected_features) == 0:
            raise ValueError("Tidak ada fitur yang lolos threshold. Coba untuk turunkan nilai threshold.")

        return X_train[self.selected_features], X_test[self.selected_features], corr_target

    def scaling_dataset(self, X_train, X_test):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def prepare_test_data(self, data_test="src/test.csv"):
        df_test = pd.read_csv(data_test)
        print(f"\nUkuran file test.csv: {df_test.shape}")

        # menyimpan Student_ID jika ada, trs hapus dari fitur
        if "Student_ID" in df_test.columns:
            student_ids = df_test["Student_ID"].copy()
            df_test = df_test.drop(columns=["Student_ID"])
        else:
            student_ids = pd.Series(range(len(df_test)), name="Student_ID")

        # membersihkan data test tanpa menghapus baris
        df_test_clean = self._clean_dataframe(df_test, is_train=False)

        # memilih fitur dan lakukan scaling sesuai parameter dari data train
        X_test_final = df_test_clean[self.selected_features]
        X_test_final_scaled = self.scaler.transform(X_test_final)

        return student_ids, X_test_final_scaled