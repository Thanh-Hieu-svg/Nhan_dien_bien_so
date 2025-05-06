import math
import cv2
import numpy as np
import Preprocess
import tkinter as tk
from tkinter import filedialog
import mysql.connector
import os
import re

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
n = 1
Min_char = 0.01
Max_char = 0.09
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.webp")])
if not file_path:
    print("No file selected. Exiting...")
    exit()
    
# Đọc ảnh và resize về kích thước chuẩn
img = cv2.imread(file_path)
img = cv2.resize(img, dsize=(1200, 1000))

######## Tải mô hình KNN ######################
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

################ Xử lý ảnh (chuyển ảnh xám và nhị phân)#################
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
canny_image = cv2.Canny(imgThreshplate, 250, 255)
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

###### Vẽ contour và lọc biển số xe #############

contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  

screenCnt = []
for c in contours:
    peri = cv2.arcLength(c, True)  # Tính chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h
    # cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    # cv2.putText(img, str(ratio), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    if (len(approx) == 4):
        screenCnt.append(approx)

        cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

        ############## Tìm góc của biển số xe #####################
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


        ############ Cắt biển số xe và xoay lại góc đúng ################

        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        # cv2.imshow("new_image",new_image)

        # Cropping
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        #################### Tiền xử lý và phân tách ký tự ####################
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow(str(n + 20), thre_mor) # kết quả nhận diện của open cv
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số

        ##################### Lọc ký tự #################
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h
            # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
            # cv2.putText(roi, str(ratiochar), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind

                # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

        ############ Nhận diện ký tự ##########################
        char_x = sorted(char_x)
        strFinalString = ""
        first_line = ""
        second_line = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            imgROI = thre_mor[y:y + h, x:x + w]  # Cắt ảnh ký tự

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # Resize ảnh
            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

            npaROIResized = np.float32(npaROIResized)
            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # Gọi KNN để nhận diện ký tự
            strCurrentChar = str(chr(int(npaResults[0][0])))  # Chuyển kết quả sang ASCII
            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

            if (y < height / 3):  # Xác định biển số 1 hay 2 dòng   
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # cv2.putText(img, first_line + "-" + second_line ,(topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        n = n + 1


        # Chuyển ảnh biển số thành dạng BLOB
        _, img_encoded = cv2.imencode('.jpg', roi)
        img_blob = img_encoded.tobytes()
        
        def process_license_plate(first_line, second_line, roi, cursor, conn):
            license_plate = None 
            # Kết hợp hai dòng thành một nếu cần
            if len(first_line) > 0 and len(second_line) > 0:
                license_plate = first_line + "-" + second_line
            elif len(first_line) > 0:
                license_plate = first_line
            else:
                license_plate = None

            if license_plate:
                # Xóa các ký tự không cần thiết
                license_plate_clean = re.sub(r"[.\-]", "", license_plate)

                # Kiểm tra độ dài hợp lệ (8 hoặc 9 ký tự)
                if len(license_plate_clean) in [8, 9]:
                    first_two_digits = license_plate_clean[:2]
                    third_char = license_plate_clean[2]
                    last_five_digits = license_plate_clean[-5:]

                    # Điều kiện kiểm tra biển số
                    if first_two_digits.isdigit() and third_char.isalpha() and last_five_digits.isdigit():
                        valid_plate = True
                    else:
                        valid_plate = False
                else:
                    valid_plate = False
            else:
                valid_plate = False

            # Nếu biển số hợp lệ, lưu vào MySQL
            if valid_plate:
                _, img_encoded = cv2.imencode('.jpg', roi)
                img_blob = img_encoded.tobytes()

                sql = "INSERT INTO so_xedb (bien_so, hinh_anh) VALUES (%s, %s)"
                cursor.execute(sql, (license_plate, img_blob))
                conn.commit()
                print(f"✔ Biển số hợp lệ: {license_plate} → Đã lưu vào MySQL")
            else:
                print(f"❌ Biển số không hợp lệ ({license_plate}), không lưu vào CSDL.")

        # Gọi hàm với các tham số
        process_license_plate(first_line, second_line, roi, cursor, conn)



        
# Tạo thư mục 'result' nếu chưa có
result_folder = "result"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 🔹 Lấy danh sách ảnh từ MySQL
cursor.execute("SELECT id, hinh_anh FROM so_xedb ORDER BY id ASC")
records = cursor.fetchall()

if records:
    for record in records:
        img_id = record[0] 
        img_blob = record[1] 

        # Chuyển BLOB thành ảnh
        img_array = np.frombuffer(img_blob, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is not None:
            file_path = os.path.join(result_folder, f"{img_id}.jpg")
            # Lưu ảnh vào thư mục result
            cv2.imwrite(file_path, img)
            print(f"✔ Ảnh ID {img_id} đã được lưu vào {file_path}")
        else:
            print(f"❌ Lỗi giải mã ảnh ID {img_id}")
else:
    print("❌ Không tìm thấy ảnh trong cơ sở dữ liệu!")

img = cv2.resize(img, None, fx=0.5, fy=0.5)
# cv2.imshow('Ket qua', img)
cv2.waitKey(0)
# Đóng kết nối MySQL
cursor.close()
conn.close()
