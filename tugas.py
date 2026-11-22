import numpy as np
import pandas as pd

student = pd.read_csv('src/train.csv')
# student.info()

# Data Pre-Proccesing
def clean_student_dataset(student):
    print(f"Ukuran awal: {student.shape}")

    #  Standarisasi nama kolom dan nilai teks
    # Menghapus karakter tab tersembunyi
    student.columns = student.columns.str.strip()
    kolom_teks = student.select_dtypes(include=['object', 'string']).columns
    student[kolom_teks] = student[kolom_teks].apply(lambda col: col.str.strip())

    # Mengonversi placeholder non-standar menjadi np.nan
    student = student.replace(
        to_replace=r'(?i)^(?:\?|-|na|n/a|null|none|)$',
        value=np.nan,
        regex=True
    )

    # Menghapus baris duplikat
    student = student.drop_duplicates(keep='first')
    kolom_fitur = [col for col in student.columns if col != 'Student_ID']
    student = student.drop_duplicates(subset=kolom_fitur, keep='first')

    # Memvalidasi domain & Menangani invalid data
    # 1 Melakukan koreksi Application order = 0 (di luar rentang valid 1-9) menggunakan modus
    student.loc[student['Application order'] < 1, 'Application order'] = np.nan
    student['Application order'] = (
        student['Application order']
        .fillna(student['Application order'].mode()[0])
        .astype(int)
    )

    # 2. Memvalidasi rentang nilai akademik (skala 0-200 & 0-20) dan konsistensi SKS
    for col in ['Previous qualification (grade)', 'Admission grade']:
        student.loc[(student[col] < 0) | (student[col] > 200), col] = np.nan

    for sem in ['1st sem', '2nd sem']:
        grade_col = f'Curricular units {sem} (grade)'
        enr_col = f'Curricular units {sem} (enrolled)'
        app_col = f'Curricular units {sem} (approved)'

        student.loc[(student[grade_col] < 0) | (student[grade_col] > 20), grade_col] = np.nan
        student.loc[student[app_col] > student[enr_col], app_col] = np.nan

    # Menangani missing values (jika muncul dari hasil validasi domain)
    for col in student.select_dtypes(include=[np.number]).columns:
        if student[col].isnull().sum() > 0:
            student[col] = student[col].fillna(student[col].median())

    for col in student.select_dtypes(include=['object', 'string']).columns:
        if student[col].isnull().sum() > 0:
            student[col] = student[col].fillna(student[col].mode()[0])

    # Menghapus kolom identifier non-informatif sebelum tahap analisis/pemodelan (ID tidak menentukan keputusan model)
    if 'Student_ID' in student.columns:
        student = student.drop(columns=['Student_ID'])

    print(f"Ukuran akhir siap analisis: {student.shape}")
    print(f"Total missing values akhir: {student.isnull().sum().sum()}")
    return student


student_clean = clean_student_dataset(student)