from flask import Flask, request, jsonify, render_template
import json
import os
from compare import FaceRecognitionApp

app = Flask(__name__)
face_app = FaceRecognitionApp(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_pass="123456789"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        id = data.get('id')
        name = data.get('name')
        image = data.get('image')
       
        if not id or not name or not image:
            return jsonify({"status": "error", "message": "Thiếu thông tin"})
       
        image_paths = face_app.save_face_multi(id, name, image, num_captures=30)
       
        return jsonify({
            "status": "success",
            "message": f"Đã đăng ký khuôn mặt cho {name} với 30 ảnh",
            "image_count": len(image_paths)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image = data.get('image')
       
        if not image:
            return jsonify({"status": "error", "message": "Thiếu ảnh"})
       
        matches = face_app.recognize_face(image)
        sorted_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
        best_matches_by_id = {}
        for match in sorted_matches:
            if match['id'] not in best_matches_by_id or match['similarity'] > best_matches_by_id[match['id']]['similarity']:
                best_matches_by_id[match['id']] = match

        best_matches = list(best_matches_by_id.values())
        best_matches = sorted(best_matches, key=lambda x: x['similarity'], reverse=True)[:3]
       
        return jsonify({
            "status": "success",
            "matches": best_matches
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/list', methods=['GET'])
def list_persons():
    try:
        persons = face_app.get_person_list()
        return jsonify({
            "status": "success",
            "persons": persons
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/persons', methods=['GET'])
def get_persons():
    try:
        persons = face_app.get_persons_with_image_count()
        return jsonify({
            "status": "success",
            "persons": persons
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/delete', methods=['POST'])
def delete_person():
    try:
        data = request.get_json()
        person_id = data.get('id')
        name = data.get('name')

        if not person_id and not name:
            return jsonify({"status": "error", "message": "Thiếu ID hoặc tên người dùng"})

        if not person_id:
            persons = face_app.get_person_list()
            matched_person = next((p for p in persons if p['name'] == name), None)
            if matched_person:
                person_id = matched_person['id']
            else:
                return jsonify({"status": "error", "message": f"Không tìm thấy người dùng với tên {name}"})

        if face_app.delete_person(person_id):
            return jsonify({"status": "success", "message": f"Đã xóa người dùng {person_id}"})
        else:
            return jsonify({"status": "error", "message": "Không thể xóa người dùng"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    try:
        os.makedirs('static', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
       
        app.run(host="0.0.0.0", port=5001, debug=True)
    finally:
        face_app.close()