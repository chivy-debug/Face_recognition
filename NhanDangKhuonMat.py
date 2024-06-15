import cv2
import face_recognition
import os
import numpy as np


# step 1 load ảnh từ kho ảnh nhận dạng


path="picture"
images = []
classNames = []
myList =os.listdir(path)
print(myList)
for cl in myList:
    print(cl)
    curImg = cv2.imread(f"{path}/{cl}") # picture/Chi Vy.jpg
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # splitext sẽ tách path ra thành 2 phần, phần trước đuôi mở rộng và phần mở rộng
print(len(images))
print(classNames)

#step encoding

def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #BGR được chuyển đổi sang RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = Mahoa(images)
print("ma hoa thanh cong")
print(len(encodeListKnow))

#khởi dộng webcam
cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while(True):
    ret, frame= cap.read()
    framS = cv2.resize(frame,(0,0),None,fx=1,fy=1)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    # xác định vị trí khuôn mặt trên cam và encode hình ảnh trên cam
    facecurFrame = face_recognition.face_locations(framS) # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame): # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại theo cặp
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis) #đẩy về index của faceDis nhỏ nhất


        if faceDis[matchIndex] <0.50 :
            name = classNames[matchIndex].upper()
        else:
            name = "Unknow"

        #print tên lên frame
        y1, x2, y2, x1 = faceLoc
        cv2.rectangle(frame,(x1,y1), (x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('Nhan dien khuon mat', frame)
    if cv2.waitKey(1) == ord("0"):  # độ trễ 1/1000s , nếu bấm 0 sẽ thoát
        break
cap.release()  # giải phóng camera
cv2.destroyAllWindows()  # thoát tất cả các cửa sổ

