import cv2
import os
from mtcnn import MTCNN
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456789"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def save_to_neo4j(student_id, name, images):
    with driver.session() as session:
        session.run(
            """
            MERGE (s:Student {id: $id})
            SET s.name = $name, s.image_count = $count
            """,
            id=student_id, name=name, count=len(images)
        )
        for img_path in images:
            session.run(
                """
                MATCH (s:Student {id: $id})
                MERGE (img:Image {path: $path})
                MERGE (s)-[:HAS_IMAGE]->(img)
                """,
                id=student_id, path=img_path
            )

def takeImages(student_id, name):
    if not student_id.isdigit():
        return "⚠️ Error: ID must be a number!"

    if not name.isalpha():
        return "⚠️ Error: Name must be alphabetic!"

    cam = cv2.VideoCapture(0)
    detector = MTCNN()
    sampleNum = 0
    save_dir = os.path.join("FRAS", "TrainingImage")

    os.makedirs(save_dir, exist_ok=True)
    saved_images = []

    while True:
        ret, img = cam.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        for face in faces:
            x, y, w, h = face['box']
            sampleNum += 1
            face_img = img[y:y+h, x:x+w]
            img_path = os.path.join(save_dir, f"{name}.{student_id}.{sampleNum}.jpg")
            cv2.imwrite(img_path, face_img)
            saved_images.append(img_path)

            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
            cv2.imshow('Frame', img)

        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()

    save_to_neo4j(student_id, name, saved_images)

    return f"✅ {sampleNum} images saved for ID: {student_id}, Name: {name}, stored in Neo4j!"