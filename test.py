import cv2
import face_recognition
from mtcnn.mtcnn import MTCNN
import os

# Đường dẫn đến thư mục chứa các ảnh khuôn mặt
folder_path = "path/to/folder"

# Danh sách các ảnh trong thư mục
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Danh sách các khuôn mặt đã biết và mã hóa của chúng
known_faces = []
known_encodings = []

# Load và mã hóa các khuôn mặt từ các ảnh trong thư mục
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    face_image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(face_image)

    if len(face_encoding) > 0:
        known_faces.append(os.path.splitext(image_file)[0])
        known_encodings.append(face_encoding[0])

# Khởi tạo MTCNN
mtcnn = MTCNN()

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()

    # Sử dụng MTCNN để nhận dạng khuôn mặt
    faces = mtcnn.detect_faces(frame)

    for face in faces:
        # Lấy tọa độ của khuôn mặt
        x, y, w, h = face['box']

        # Cắt ảnh khuôn mặt từ frame
        face_image = frame[y:y + h, x:x + w]

        # Sử dụng face_recognition để mã hóa khuôn mặt và so sánh
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) > 0:
            matches = face_recognition.compare_faces(known_encodings, face_encoding[0])

            # Xác định kết quả so sánh
            name = "Unknown"
            for i, match in enumerate(matches):
                if match:
                    name = known_faces[i]
                    break

            # Hiển thị kết quả
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị frame kết quả
    cv2.imshow("Face Recognition", frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
