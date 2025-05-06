import math
import threading
import cv2
import numpy as np
import os
import Preprocess
import mysql.connector
import tkinter as tk
import re
import time
from tkinter import filedialog

# Kết nối cơ sở dữ liệu MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  
    database="bien_so_xe"
)
cursor = conn.cursor()

ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

Min_char_area = 0.015
Max_char_area = 0.06

Min_char = 0.01
Max_char = 0.09

Min_ratio_char = 0.25
Max_ratio_char = 0.7

max_size_plate = 18000
min_size_plate = 5000

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

tongframe = 0
biensotimthay = 0

last_save_time = 0
last_saved_plate = ""


# Tải mô hình KNN
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))   # Chuyển đổi numpy array thành 1 chiều, cần thiết để huấn luyện KNN
kNearest = cv2.ml.KNearest_create()  # Khởi tạo đối tượng KNN
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

# Đọc video
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])

if not video_path:
    print("No file selected. Exiting...")
    exit()

cap = cv2.VideoCapture(video_path)


while (cap.isOpened()):

    # Xử lý ảnh
    ret, img = cap.read()
    if not ret or img is None:
        print("Không thể đọc frame từ video hoặc video đã kết thúc.")
        break
    tongframe = tongframe + 1
    
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)   
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel,iterations=1)  # Giãn nở ảnh

    # Lọc biển số xe
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 đường viền lớn nhất
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tính chu vi
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)   # Xấp xỉ các cạnh của đường viền
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)
    if screenCnt is None:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:
        n = 1
        for screenCnt in screenCnt:

            # Tìm góc nghiêng của biển số xe
            (x1, y1) = screenCnt[0, 0]
            (x2, y2) = screenCnt[1, 0]
            (x3, y3) = screenCnt[2, 0]
            (x4, y4) = screenCnt[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            sorted_array = array.sort(reverse=True, key=lambda x: x[1])
            (x1, y1) = array[0]
            (x2, y2) = array[1]

            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = math.atan(doi / ke) * (180.0 / math.pi)
            #################################################

             # Mặt nạ khu vực biển số
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

             # Cắt vùng biển số
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx + 1, topy:bottomy + 1]
            imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

             # Xử lý biển số xeng
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Nhận diện ký tự
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width
            # print ("roiarea",roiarea)
            for ind, cnt in enumerate(cont):
                area = cv2.contourArea(cnt)
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind

            # Nhận diện ký tự trên biển số xe
            if len(char_x) in range(7, 10):
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

                char_x = sorted(char_x)
                strFinalString = ""
                first_line = ""
                second_line = ""

                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    imgROI = thre_mor[y:y + h, x:x + w]

                    imgROIResized = cv2.resize(imgROI,
                                               (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize
                    npaROIResized = imgROIResized.reshape(
                        (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # đưa hình ảnh về mảng 1 chiều
                    # Chuyển ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
                    npaROIResized = np.float32(npaROIResized)  # chuyển mảng về dạng float
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # call KNN function find_nearest; neigh_resp là hàng xóm
                    strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of the character
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

                    if (y < height / 3):   
                        first_line = first_line + strCurrentChar
                    else:
                        second_line = second_line + strCurrentChar

                strFinalString = first_line + second_line
                print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
                cv2.putText(img, strFinalString, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                n = n + 1
                biensotimthay = biensotimthay + 1

                cv2.imshow("a", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                
                #-----------------------------------------------------
                # Kiểm tra nếu biển số hợp lệ (không rỗng và có định dạng đúng)
                if len(first_line) > 0 and len(second_line) > 0:
                    license_plate = first_line + "-" + second_line
                    first_two_digits = first_line[:2]  # Kiểm tra 2 ký tự đầu có phải số không
                    last_five_digits = second_line.replace(".", "")[-5:]  # Kiểm tra 5 ký tự cuối
                    current_time = time.time()
                    
                    if first_two_digits.isdigit() and last_five_digits.isdigit():
                        if license_plate != last_saved_plate or (current_time - last_save_time >= 1):
                            last_save_time = current_time
                            last_saved_plate = license_plate
                            _, img_encoded = cv2.imencode('.jpg', roi)
                            img_blob = img_encoded.tobytes()

                            sql = "INSERT INTO so_xedb (bien_so, hinh_anh) VALUES (%s, %s)"
                            cursor.execute(sql, (license_plate, img_blob))
                            conn.commit()
                            print(f"✔ Biển số hợp lệ: {license_plate} → Đã lưu vào MySQL")
                        else:
                            print(f"⚠ Biển số {license_plate} trùng với lần trước, không lưu lại trong 1 giây.")
                    else:
                        print(f"❌ Biển số {license_plate} không hợp lệ (Hai ký tự đầu và năm ký tự cuối phải là số), không lưu vào CSDL.")



    def save_images_from_mysql():
        print("🔄 Đang tải ảnh từ MySQL...")

        result_folder = "result"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # 🔹 Lấy danh sách ảnh từ MySQL
        cursor.execute("SELECT id, hinh_anh FROM so_xedb ORDER BY id ASC")
        records = cursor.fetchall()  # 🔥 Đọc hết dữ liệu trước khi chạy truy vấn mới

        if records:
            for record in records:
                img_id = record[0]  
                img_blob = record[1]  

                # Chuyển BLOB thành ảnh
                img_array = np.frombuffer(img_blob, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    file_path = os.path.join(result_folder, f"{img_id}.jpg")
                    cv2.imwrite(file_path, img)
                    print(f"✔ Ảnh ID {img_id} đã được lưu vào {file_path}")
                else:
                    print(f"❌ Lỗi giải mã ảnh ID {img_id}")
        else:
            print("❌ Không tìm thấy ảnh trong cơ sở dữ liệu!")

        # 🔹 Đóng cursor để tránh lỗi `Unread result found`
        cursor.close()


    imgcopy = cv2.resize(img, None, fx=1.2, fy=1.2)
    cv2.imshow('Biển số xe', imgcopy)
    print("Biển số tìm thấy", biensotimthay)
    print("Tổng frame", tongframe)
    print("Tỷ lệ tìm thấy biển số:", 100 * biensotimthay / 368, "%")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#video.py