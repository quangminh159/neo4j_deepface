import datetime
import os
import time
import cv2
import pandas as pd
from mtcnn import MTCNN
from neo4j import GraphDatabase

# Kết nối Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456789"

class FaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = MTCNN()
        self.db = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.model_path = os.path.join("FRAS", "TrainingImageLabel", "Trainer.yml")
        self.student_csv_path = os.path.join("FRAS", "StudentDetails", "StudentDetails.csv")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Lỗi: File '{self.model_path}' không tồn tại!")

        if not os.path.exists(self.student_csv_path):
            raise FileNotFoundError(f"❌ Lỗi: File '{self.student_csv_path}' không tồn tại!")

        self.recognizer.read(self.model_path)
        self.df = pd.read_csv(self.student_csv_path)

    def recognize_attendance(self):
        """Nhận diện khuôn mặt và lưu điểm danh vào Neo4j"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        col_names = ['Id', 'Name', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col_names)

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        if not cam.isOpened():
            print("❌ Lỗi: Không thể mở camera!")
            return {"status": "error", "message": "Không thể mở camera!"}

        print("📸 Đang nhận diện khuôn mặt... Nhấn 'q' để thoát.")

        while True:
            ret, im = cam.read()
            if not ret:
                print("❌ Lỗi: Không thể đọc dữ liệu từ camera!")
                break

            img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(img_rgb)

            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(im, (x, y), (x + w, y + h), (10, 159, 255), 2)

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                Id, conf = self.recognizer.predict(gray[y:y+h, x:x+w])

                if conf < 100:
                    name_row = self.df.loc[self.df['Id'] == Id, 'Name']
                    name = name_row.values[0] if not name_row.empty else "Unknown"
                    confstr = f"  {round(100 - conf)}%"
                    tt = f"{Id} - {name}"
                else:
                    Id, name, tt, confstr = "Unknown", "Unknown", "Unknown", "  0%"

                if (100 - conf) > 65:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    attendance.loc[len(attendance)] = {'Id': Id, 'Name': name, 'Date': date, 'Time': timeStamp}

                status = "[Pass]" if (100 - conf) > 65 else "[Fail]"
                cv2.putText(im, f"{tt} {status}", (x + 5, y - 5), font, 1, (255, 255, 255), 2)

            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')

            cv2.imshow('Attendance', im)

            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

        if attendance.empty:
            print("❌ Không có ai được điểm danh!")
            return {"status": "error", "message": "Không có ai được điểm danh!"}

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")

        attendance_folder = os.path.join("FRAS", "Attendance")
        os.makedirs(attendance_folder, exist_ok=True)

        fileName = os.path.join(attendance_folder, f"Attendance_{date}_{Hour}-{Minute}-{Second}.csv")
        attendance.to_csv(fileName, index=False)
        print(f"✅ Điểm danh thành công! File lưu tại '{fileName}'")
        with self.db.session() as session:
            for _, row in attendance.iterrows():
                session.run(
                    "MERGE (s:Student {id: $id}) "
                    "ON MATCH SET s.name = $name "
                    "MERGE (a:Attendance {date: $date, time: $time}) "
                    "MERGE (s)-[:ATTENDED]->(a)",
                    id=row["Id"], name=row["Name"], date=row["Date"], time=row["Time"]
                )

        return {"status": "success", "message": f"Điểm danh thành công! File lưu tại {fileName}"}

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.recognize_attendance()
