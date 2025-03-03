import os
import time
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from neo4j import GraphDatabase
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456789"

class FaceTrainer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = MTCNN()
        self.training_path = os.path.join("FRAS", "TrainingImage")
        self.output_folder = os.path.join("FRAS", "TrainingImageLabel")
        os.makedirs(self.output_folder, exist_ok=True)
        self.db = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def get_images_and_labels(self):
        """Lấy ảnh và nhãn từ thư mục TrainingImage"""
        if not os.path.exists(self.training_path):
            print(f"❌ Lỗi: Thư mục '{self.training_path}' không tồn tại!")
            return [], []

        imagePaths = [os.path.join(self.training_path, f) for f in os.listdir(self.training_path)]
        faces, Ids = [], []

        for imagePath in imagePaths:
            try:
                pilImage = Image.open(imagePath).convert('RGB')
                imageNp = np.array(pilImage, 'uint8')
                results = self.detector.detect_faces(imageNp)
                for result in results:
                    x, y, w, h = result['box']
                    face = imageNp[y:y+h, x:x+w]
                    faces.append(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY))
                    Id = int(os.path.split(imagePath)[-1].split(".")[1])
                    Ids.append(Id)
            except Exception as e:
                print(f"⚠ Lỗi khi xử lý ảnh '{imagePath}': {e}")

        return faces, Ids

    def train_images(self):
        """Huấn luyện mô hình nhận diện khuôn mặt và lưu vào Neo4j"""
        faces, Ids = self.get_images_and_labels()

        if len(faces) == 0 or len(Ids) == 0:
            print("❌ Không có dữ liệu hình ảnh để huấn luyện!")
            return {"status": "error", "message": "Không có dữ liệu hình ảnh để huấn luyện!"}

        print("✅ Bắt đầu huấn luyện mô hình...")
        self.recognizer.train(faces, np.array(Ids))

        model_path = os.path.join(self.output_folder, "Trainer.yml")
        self.recognizer.save(model_path)
        print(f"✅ Huấn luyện hoàn tất! Model đã lưu tại '{model_path}'")
        with self.db.session() as session:
            session.run(
                "MERGE (t:Training {model: $model, num_faces: $count})",
                model=model_path, count=len(faces)
            )

        return {"status": "success", "message": f"Model đã lưu tại {model_path}"}

if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.train_images()
