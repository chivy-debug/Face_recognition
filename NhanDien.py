import cv2
import face_recognition
import os
import numpy as np

# Load mô hình nhận diện khuôn mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đường dẫn đến thư mục chứa ảnh đã biết
known_images_folder = 'picture'

# Lấy danh sách tên các file trong thư mục
known_image_files = [f for f in os.listdir(known_images_folder) if os.path.isfile(os.path.join(known_images_folder, f))]

# Load và mã hóa ảnh đã biết
known_encodings = []
known_names = []
for image_file in known_image_files:
    image_path = os.path.join(known_images_folder, image_file)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(img_rgb)
    if not face_encodings:
        print(f"Không tìm thấy khuôn mặt trong ảnh {image_file}.")
        continue
    for face_encoding in face_encodings:
        known_encodings.append(face_encoding)
        known_names.append(os.path.splitext(image_file)[0])


# Khởi động webcam
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    # Chuyển đổi frame sang ảnh đen trắng (gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt trong frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Mã hóa khuôn mặt từ webcam
        roi_color = frame[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        face_encodings_webcam = face_recognition.face_encodings(roi_rgb)

        # Nếu có kết quả khớp, chọn tên của người đã biết
        if face_encodings_webcam:
            for face_encoding_webcam in face_encodings_webcam:
                matches = face_recognition.compare_faces(known_encodings, face_encoding_webcam)
                name = "Unknown"

                # Nếu có kết quả khớp, chọn tên của người đã biết
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                # Hiển thị tên lên frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị frame kết quả
    cv2.imshow('Face Recognition', frame)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
