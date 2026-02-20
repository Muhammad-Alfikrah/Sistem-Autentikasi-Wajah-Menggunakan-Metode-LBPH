import streamlit as st
import cv2
import os
import numpy as np
import pickle
import sqlite3
from PIL import Image
from datetime import datetime
import pandas as pd
import io
import time

st.set_page_config(page_title="Sistem Autentikasi Wajah", layout="centered")
st.title("Sistem Autentikasi Wajah Menggunakan LBPH")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# DATABASE
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS mahasiswa (
    nim TEXT PRIMARY KEY,
    nama TEXT,
    kelas TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS log_akses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nim TEXT,
    nama TEXT,
    kelas TEXT,
    waktu TEXT,
    status TEXT
)
""")

conn.commit()

dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

menu = st.sidebar.selectbox(
    "Menu Utama", ["Ambil Dataset", "Latih Model", "Autentikasi", "Admin & Evaluasi"]
)

# 1. AMBIL DATASET
if menu == "Ambil Dataset":
    st.subheader("Ambil Dataset Wajah")

    nim = st.text_input("NIM")
    nama = st.text_input("Nama")
    kelas = st.text_input("Kelas")

    if st.button("Simpan Data & Ambil Foto"):

        if nim == "" or nama == "" or kelas == "":
            st.error("Semua field wajib diisi!")
            st.stop()

        cursor.execute(
            """
            INSERT OR REPLACE INTO mahasiswa (nim, nama, kelas)
            VALUES (?, ?, ?)
            """,
            (nim, nama, kelas),
        )
        conn.commit()

        cap = cv2.VideoCapture(0)
        folder_path = os.path.join(dataset_dir, nama)
        os.makedirs(folder_path, exist_ok=True)

        count = 1
        countdown = 3
        start_time = None
        ready = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if not ready:
                cv2.putText(
                    frame,
                    "Tekan Q untuk mulai ambil foto",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if start_time is None:
                        start_time = time.time()

                    elapsed = int(time.time() - start_time)
                    remaining = countdown - elapsed

                    if remaining > 0:
                        cv2.putText(
                            frame,
                            f"Ambil Foto dalam {remaining}",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2,
                        )
                    else:
                        face = gray[y:y+h, x:x+w]
                        face = cv2.resize(face, (200, 200))
                        cv2.imwrite(f"{folder_path}/{count}.png", face)
                        count += 1
                        start_time = None

                    if count > 20:
                        break

            cv2.imshow("Ambil Dataset", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                ready = True

            if count > 20:
                break

        cap.release()
        cv2.destroyAllWindows()

        st.success("Dataset berhasil disimpan!")

# 2. LATIH MODEL
elif menu == "Latih Model":
    st.subheader("Latih Model LBPH")

    if st.button("Latih Sekarang"):

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        label_ids = {}
        x_train = []
        y_labels = []
        current_id = 0

        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith("png"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root)

                    if label not in label_ids:
                        label_ids[label] = current_id
                        current_id += 1

                    id_ = label_ids[label]
                    image = Image.open(path).convert("L")
                    image_np = np.array(image, "uint8")

                    faces = face_cascade.detectMultiScale(image_np, 1.1, 5)

                    for x, y, w, h in faces:
                        roi = image_np[y : y + h, x : x + w]
                        x_train.append(roi)
                        y_labels.append(id_)

        with open("labels.pickle", "wb") as f:
            pickle.dump(label_ids, f)

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trained_model.yml")

        st.success("Model berhasil dilatih!")

# 3. AUTENTIKASI
elif menu == "Autentikasi":
    st.subheader("Autentikasi Wajah")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trained_model.yml")

    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    img = st.camera_input("Scan Wajah")

    if img is not None:
        img_pil = Image.open(img).convert("L")
        img_np = np.array(img_pil)

        faces = face_cascade.detectMultiScale(img_np, 1.1, 5)

        if len(faces) == 0:
            st.error("Wajah tidak terdeteksi")
            st.stop()

        for x, y, w, h in faces:
            roi = img_np[y : y + h, x : x + w]
            id_, conf = recognizer.predict(roi)

            waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if conf < 60:
                nama = labels[id_]

                cursor.execute(
                    """
                    SELECT nim, kelas FROM mahasiswa WHERE nama=?
                """,
                    (nama,),
                )
                data = cursor.fetchone()
                nim_db, kelas_db = data if data else ("-", "-")

                cursor.execute(
                    """
                    INSERT INTO log_akses (nim, nama, kelas, waktu, status)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (nim_db, nama, kelas_db, waktu, "DITERIMA"),
                )
                conn.commit()

                st.success(f"AKSES DITERIMA - {nama}")

            else:
                cursor.execute(
                    """
                    INSERT INTO log_akses (nim, nama, kelas, waktu, status)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    ("-", "Unknown", "-", waktu, "DITOLAK"),
                )
                conn.commit()

                st.error("AKSES DITOLAK")

# 4. ADMIN & EVALUASI
elif menu == "Admin & Evaluasi":
    st.subheader("Evaluasi Sistem")

    cursor.execute("SELECT * FROM log_akses")
    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=["ID", "NIM", "Nama", "Kelas", "Waktu", "Status"])
    st.dataframe(df)

    # Hitung Confusion Matrix
    TP = len(df[(df["Status"] == "DITERIMA") & (df["Nama"] != "Unknown")])
    FP = len(df[(df["Status"] == "DITERIMA") & (df["Nama"] == "Unknown")])
    TN = len(df[(df["Status"] == "DITOLAK") & (df["Nama"] == "Unknown")])
    FN = len(df[(df["Status"] == "DITOLAK") & (df["Nama"] != "Unknown")])

    total = TP + TN + FP + FN

    if total > 0:
        accuracy = (TP + TN) / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        st.subheader("Confusion Matrix")
        st.write(f"TP: {TP}")
        st.write(f"FP: {FP}")
        st.write(f"TN: {TN}")
        st.write(f"FN: {FN}")

        st.subheader("Metode Evaluasi")
        st.write(f"Akurasi: {accuracy*100:.2f}%")
        st.write(f"Precision: {precision*100:.2f}%")
        st.write(f"Recall: {recall*100:.2f}%")
        st.write(f"F1-Score: {f1*100:.2f}%")

    # Export Excel
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    st.download_button(
        label="Download Log Excel",
        data=buffer,
        file_name="log_akses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
