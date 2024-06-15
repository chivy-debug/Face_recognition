from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import os

# Load mô hình FaceNet
facenet_model = load_model('path_to_facenet_model.h5')

# Load ảnh và chuyển đổi thành vectơ nhúng (embedding)
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Load ảnh và phát hiện khuôn mặt
def load_image(filename):
    pixels = np.asarray(Image.open(filename))
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    return pixels, results

# Thư mục chứa ảnh khuôn mặt
faces_dir = 'path_to_faces_directory/'

# Duyệt qua tất cả các file trong thư mục và nhận dạng khuôn mặt
for filename in os.listdir(faces_dir):
    path = os.path.join(faces_dir, filename)
    pixels, faces = load_image(path)

    for face in faces:
        x, y, width, height = face['box']
        x1, y1 = abs(x), abs(y)
        x2, y2 = x1 + width, y1 + height

        # Trích xuất khuôn mặt từ ảnh
        face_pixels = pixels[y1:y2, x1:x2]

        # Chuyển đổi khuôn mặt thành vectơ nhúng (embedding) bằng FaceNet
        face_embedding = get_embedding(facenet_model, face_pixels)

        # In ra tên file và vectơ nhúng tương ứng
        print(f"File: {filename}, Embedding: {face_embedding}")
