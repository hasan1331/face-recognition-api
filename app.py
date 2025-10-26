from flask import Flask, request, jsonify
import face_recognition
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load wajah yang dikenal
known_image = face_recognition.load_image_file("known_faces/hasan.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 → image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        unknown_image = np.array(img)

        unknown_encodings = face_recognition.face_encodings(unknown_image)
        if not unknown_encodings:
            return jsonify({"error": "No face detected"}), 400

        # Bandingkan wajah
        results = face_recognition.compare_faces([known_encoding], unknown_encodings[0])
        return jsonify({"match": bool(results[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "Face Recognition API is running 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
