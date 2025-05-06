import math
import cv2
import numpy as np
import Preprocess
import mysql.connector
import os
import threading

# Kết nối cơ sở dữ liệu MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Thay bằng mật khẩu MySQL nếu có
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

# Load KNN model
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

# **Chuyển sang nhận diện từ camera thay vì video**
cap = cv2.VideoCapture(0)  # Dùng camera mặc định

if not cap.isOpened():
    print("Không thể mở camera. Kiểm tra kết nối!")
    exit()

while True:
    ret, img = cap.read()
    if not ret or img is None:
        print("Không thể đọc frame từ camera.")
        break
    tongframe += 1

    # Tiền xử lý ảnh
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

    # Tìm contour của biển số
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        if len(approx) == 4 and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)

    if not screenCnt:
        print("Không phát hiện biển số.")
    else:
        n = 1
        for screenCnt in screenCnt:
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = [screenCnt[i, 0] for i in range(4)]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            array.sort(reverse=True, key=lambda x: x[1])

            doi = abs(array[0][1] - array[1][1])
            ke = abs(array[0][0] - array[1][0])
            angle = math.atan(doi / ke) * (180.0 / math.pi)

            # Tạo mask và crop biển số
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
            (x, y) = np.where(mask == 255)
            (topx, topy), (bottomx, bottomy) = (np.min(x), np.min(y)), (np.max(x), np.max(y))

            roi = img[topx:bottomx + 1, topy:bottomy + 1]
            imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle if x1 < x2 else angle, 1.0)
            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

            roi = cv2.resize(roi, None, fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, None, fx=3, fy=3)

            # Xử lý ký tự biển số
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width

            for ind, cnt in enumerate(cont):
                area = cv2.contourArea(cnt)
                (x, y, w, h) = cv2.boundingRect(cnt)
                ratiochar = w / h
                if Min_char * roiarea < area < Max_char * roiarea and 0.25 < ratiochar < 0.7:
                    char_x.append(x)
                    char_x_ind[x] = ind

            if 7 <= len(char_x) <= 10:
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

                char_x.sort()
                # strFinalString = first_line = second_line = ""
                first_line = second_line = ""
                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    imgROI = thre_mor[y:y + h, x:x + w]
                    imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                    npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)).astype(np.float32)

                    _, npaResults, _, _ = kNearest.findNearest(npaROIResized, k=3)
                    strCurrentChar = str(chr(int(npaResults[0][0])))
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

                    if y < height / 3:
                        first_line += strCurrentChar
                    else:
                        second_line += strCurrentChar

                strFinalString = first_line + second_line
                #----
                
                #-----
                print(f"\n Biển số {n}: {first_line} - {second_line}\n")
                cv2.putText(img, strFinalString, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                n += 1
                biensotimthay += 1

                cv2.imshow("License Plate", roi)
                # ------------------- 🌟 Kiểm tra và lưu vào CSDL 🌟 -------------------
                if len(first_line) > 0 and len(second_line) > 0:
                    license_plate = first_line + "-" + second_line

                    first_two_digits = first_line[:2]  # Kiểm tra 2 ký tự đầu có phải số không
                    last_five_digits = second_line.replace(".", "")[-5:]  # Kiểm tra 5 ký tự cuối

                    if first_two_digits.isdigit() and last_five_digits.isdigit():
                        _, img_encoded = cv2.imencode('.jpg', roi)
                        img_blob = img_encoded.tobytes()

                        sql = "INSERT INTO so_xedb (bien_so, hinh_anh) VALUES (%s, %s)"
                        cursor.execute(sql, (license_plate, img_blob))
                        conn.commit()
                        print(f"✔ Biển số hợp lệ: {license_plate} → Đã lưu vào MySQL")
                    else:
                        print(f"❌ Biển số {license_plate} không hợp lệ (Hai ký tự đầu và năm ký tự cuối phải là số), không lưu vào CSDL.")
                else:
                    print("❌ Không phát hiện được biển số hợp lệ, không lưu vào CSDL.")
        
    # note: Không dùng kết nối MySQL chính trong luồng phụ    
    # chú ý: Xém bay màu con laptop        
    def save_images_from_db():
        print("🔹 Bắt đầu luồng lưu ảnh từ MySQL...")

        # Kết nối CSDL mới trong luồng phụ
        conn_thread = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Thay bằng mật khẩu MySQL nếu có
            database="bien_so_xe"
        )
        cursor_thread = conn_thread.cursor()

        # Tạo thư mục 'result' nếu chưa có
        result_folder = "result"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        # Lấy danh sách ảnh từ MySQL
        cursor_thread.execute("SELECT id, hinh_anh FROM so_xedb ORDER BY id ASC")
        records = cursor_thread.fetchall()

        if records:
            for record in records:
                img_id = record[0]  # Lấy ID ảnh từ MySQL
                img_blob = record[1]  # Lấy dữ liệu ảnh (BLOB)

                # Chuyển BLOB thành ảnh
                img_array = np.frombuffer(img_blob, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    file_path = os.path.join(result_folder, f"{img_id}.jpg")

                    # Kiểm tra nếu file đã tồn tại
                    if not os.path.exists(file_path):
                        cv2.imwrite(file_path, img)
                        print(f"✔ Ảnh ID {img_id} đã được lưu vào {file_path}")
                    else:
                        print(f"⚠ Ảnh ID {img_id} đã tồn tại, bỏ qua...")
                else:
                    print(f"❌ Lỗi giải mã ảnh ID {img_id}")

        else:
            print("❌ Không tìm thấy ảnh trong cơ sở dữ liệu!")

        # Đóng kết nối MySQL trong luồng phụ
        cursor_thread.close()
        conn_thread.close()
        print("🔹 Đã hoàn thành lưu ảnh từ MySQL.")

    # Chạy luồng riêng để lưu ảnh từ MySQL
    thread = threading.Thread(target=save_images_from_db)
    thread.start()

                

    scale = 1.0  # Hệ số phóng to (tăng lên nếu cần)
    frame_resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    cv2.imshow("amera", frame_resized)
    print(f"Biển số tìm thấy: {biensotimthay} / Tổng frame: {tongframe}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

