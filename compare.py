from deepface import DeepFace
from neo4j import GraphDatabase
import base64
import os
import uuid
import time
import cv2
import numpy as np
import json
import shutil
from mtcnn import MTCNN
class FaceRecognitionApp:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass):
        self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.UPLOAD_FOLDER = 'face_database'
        self.MODELS_FOLDER = 'face_models'
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(self.MODELS_FOLDER, exist_ok=True)
        self._initialize_neo4j_schema()
       
    def close(self):
        """Đóng kết nối đến Neo4j"""
        self._driver.close()
    
    def _initialize_neo4j_schema(self):
        """Khởi tạo cấu trúc dữ liệu và ràng buộc trong Neo4j"""
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")

            session.run("CREATE CONSTRAINT image_path_unique IF NOT EXISTS FOR (i:FaceImage) REQUIRE i.path IS UNIQUE")
            
            session.run("CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.name)")
    
    def _decode_base64_image(self, image_base64):
        """Giải mã ảnh base64 thành mảng numpy"""
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
   
    def save_face_multi(self, id, name, image_base64, num_captures=30):
        """Lưu nhiều ảnh khuôn mặt từ một ảnh gốc với các biến thể nhỏ và thông tin người dùng"""
        
        try:
            img = self._decode_base64_image(image_base64)
            if img is None:
                raise ValueError("Không thể giải mã ảnh")
            
            # face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # faces = face_detector.detectMultiScale(gray, 1.3, 5)
            detector = MTCNN()
            faces = detector.detect_faces(img)
            
            if len(faces) == 0:
                raise ValueError("Không phát hiện khuôn mặt trong ảnh")
            # (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            best_face = max(faces, key=lambda f: f['confidence'])
            x, y, w, h = best_face['box']
            x = max(0, x - 10)
            y = max(0, y - 10)
            w = min(w + 20, img.shape[1] - x)
            h = min(h + 20, img.shape[0] - y)
            
            face_img = img[y:y+h, x:x+w]

            image_paths = []
            
            for i in range(num_captures):
                unique_filename = f"{name}_{uuid.uuid4()}.jpg"
                file_path = os.path.join(self.UPLOAD_FOLDER, unique_filename)
                variant = face_img.copy()

                brightness = np.random.uniform(0.9, 1.1)
                variant = cv2.convertScaleAbs(variant, alpha=brightness, beta=0)
                
                contrast = np.random.uniform(0.9, 1.1)
                variant = cv2.convertScaleAbs(variant, alpha=contrast, beta=0)
                
                angle = np.random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((variant.shape[1]//2, variant.shape[0]//2), angle, 1)
                variant = cv2.warpAffine(variant, M, (variant.shape[1], variant.shape[0]))
                

                scale = np.random.uniform(0.95, 1.05)
                if scale != 1.0:
                    new_width = int(variant.shape[1] * scale)
                    new_height = int(variant.shape[0] * scale)
                    variant = cv2.resize(variant, (new_width, new_height))

                    if scale > 1.0:
                    
                        start_x = (new_width - face_img.shape[1]) // 2
                        start_y = (new_height - face_img.shape[0]) // 2
                        variant = variant[start_y:start_y+face_img.shape[0], start_x:start_x+face_img.shape[1]]
                    else:
                        
                        pad_x = (face_img.shape[1] - new_width) // 2
                        pad_y = (face_img.shape[0] - new_height) // 2
                        variant = cv2.copyMakeBorder(variant, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT)
                        variant = cv2.resize(variant, (face_img.shape[1], face_img.shape[0]))
                
               
                cv2.imwrite(file_path, variant)
                image_paths.append(file_path)
            
          
            with self._driver.session() as session:
                session.run(
                    "MERGE (p:Person {id: $id}) "
                    "SET p.name = $name, "
                    "    p.created_at = $created_at, "
                    "    p.updated_at = $updated_at",
                    id=id,
                    name=name,
                    created_at=time.time(),
                    updated_at=time.time()
                )
                for idx, image_path in enumerate(image_paths):
                    session.run(
                        "MATCH (p:Person {id: $id}) "
                        "MERGE (i:FaceImage {path: $path}) "
                        "SET i.index = $idx, "
                        "    i.created_at = $created_at "
                        "MERGE (p)-[:HAS_FACE]->(i)",
                        id=id,
                        path=image_path,
                        idx=idx,
                        created_at=time.time()
                    )
            self._train_face_model(id, name, image_paths)
            
            return image_paths
       
        except Exception as e:
            print(f"Lỗi khi lưu ảnh đa biến thể: {e}")
            raise
    
    def _train_face_model(self, id, name, image_paths):
        """Huấn luyện mô hình nhận diện cho người dùng"""
        try:
            user_model_folder = os.path.join(self.MODELS_FOLDER, id)
            if os.path.exists(user_model_folder):
                shutil.rmtree(user_model_folder)  
            os.makedirs(user_model_folder, exist_ok=True)
            model_info = {
                'id': id,
                'name': name,
                'image_paths': image_paths,
                'created_at': time.time()
            }
            with open(os.path.join(user_model_folder, 'model_info.json'), 'w') as f:
                json.dump(model_info, f)

            for i, image_path in enumerate(image_paths):
                try:

                    embedding = DeepFace.represent(img_path=image_path, model_name="VGG-Face")

                    embedding_path = os.path.join(user_model_folder, f'embedding_{i}.npy')
                    np.save(embedding_path, np.array(embedding))

                    with self._driver.session() as session:
                        session.run(
                            "MATCH (p:Person {id: $id})-[:HAS_FACE]->(i:FaceImage {path: $path}) "
                            "SET i.embedding_path = $embedding_path",
                            id=id,
                            path=image_path,
                            embedding_path=embedding_path
                        )
                    
                except Exception as e:
                    print(f"Không thể trích xuất đặc trưng cho ảnh {image_path}: {e}")
            
            print(f"Đã huấn luyện mô hình cho {name} với {len(image_paths)} ảnh")
            return True
            
        except Exception as e:
            print(f"Lỗi khi huấn luyện mô hình: {e}")
            return False
   
    def recognize_face(self, image_base64):
        """Nhận diện khuôn mặt so với cơ sở dữ liệu"""
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        temp_path = os.path.join(self.UPLOAD_FOLDER, temp_filename)
       
        try:
            img = self._decode_base64_image(image_base64)
            cv2.imwrite(temp_path, img)

            with self._driver.session() as session:
                result = session.run(
                    "MATCH (p:Person)-[:HAS_FACE]->(i:FaceImage) "
                    "RETURN p.id AS id, p.name AS name, i.path AS image_path"
                )
                known_faces = list(result)
 
            input_embedding = DeepFace.represent(img_path=temp_path, model_name="VGG-Face")
            
            results = []
            verified_persons = set() 
            
            for face in known_faces:
                id = face['id']
                name = face['name']
                image_path = face['image_path']
                if id in verified_persons:
                    continue
                
                try:
                    verification = DeepFace.verify(
                        img1_path=temp_path,
                        img2_path=image_path,
                        model_name="VGG-Face",
                        distance_metric="cosine"
                    )
                    
                    if verification['verified']:
                        results.append({
                            'id': id,
                            'name': name,
                            'similarity': 1 - verification['distance'],
                            'image_path': image_path
                        })
                        verified_persons.add(id) 
                except Exception as e:
                    print(f"Lỗi khi so sánh: {e}")
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results
       
        except Exception as e:
            print(f"Lỗi nhận diện: {e}")
            raise
       
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def get_person_list(self):
        """Lấy danh sách tên người dùng"""
        try:
            with self._driver.session() as session:
                result = session.run(
                    "MATCH (p:Person) RETURN p.id AS id, p.name AS name"
                )
                return [{"id": record['id'], "name": record['name']} for record in result]
        except Exception as e:
            print(f"Lỗi khi lấy danh sách người dùng: {e}")
            raise
    
    def get_persons_with_image_count(self):
        """Lấy danh sách người dùng cùng số lượng ảnh"""
        try:
            with self._driver.session() as session:
                result = session.run(
                    "MATCH (p:Person)-[:HAS_FACE]->(i:FaceImage) "
                    "WITH p, COUNT(i) AS image_count "
                    "RETURN p.id AS id, p.name AS name, image_count"
                )
                return [{"id": record['id'], "name": record['name'], "image_count": record['image_count']} for record in result]
        except Exception as e:
            print(f"Lỗi khi lấy danh sách người dùng: {e}")
            raise
    
    def delete_person(self, id):
        """Xóa người dùng và ảnh của họ"""
        try:
            with self._driver.session() as session:
                image_result = session.run(
                    "MATCH (p:Person {id: $id})-[:HAS_FACE]->(i:FaceImage) "
                    "RETURN i.path AS path",
                    id=id
                )
                image_paths = [record['path'] for record in image_result]
                user_model_folder = os.path.join(self.MODELS_FOLDER, id)
                if os.path.exists(user_model_folder):
                    shutil.rmtree(user_model_folder)
                for path in image_paths:
                    if os.path.exists(path):
                        os.remove(path)
                session.run(
                    "MATCH (p:Person {id: $id})-[r:HAS_FACE]->(i:FaceImage) "
                    "DELETE r, i, p",
                    id=id
                )
                
            return True
        except Exception as e:
            print(f"Lỗi khi xóa người dùng: {e}")
            raise
            
    def visualize_person_faces(self, id):
        """Tạo câu truy vấn Cypher để hiển thị người và khuôn mặt của họ trong Neo4j Browser"""
        cypher_query = """
        MATCH (p:Person {id: $id})-[:HAS_FACE]->(i:FaceImage)
        RETURN p, i
        """
        print(f"Để hiển thị người dùng {id} và khuôn mặt của họ trong Neo4j Browser, chạy câu truy vấn sau:")
        print(cypher_query.replace("$id", f"'{id}'"))
        
        return cypher_query