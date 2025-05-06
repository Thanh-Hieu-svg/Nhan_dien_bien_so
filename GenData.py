import numpy as np
import cv2
import sys

# Các biến cấp module ##########################################################################
MIN_CONTOUR_AREA = 40  # Diện tích tối thiểu của contour để được coi là ký tự hợp lệ

RESIZED_IMAGE_WIDTH = 20  # Chiều rộng ảnh sau khi resize
RESIZED_IMAGE_HEIGHT = 30  # Chiều cao ảnh sau khi resize

###################################################################################################
def main():
    imgTrainingNumbers = cv2.imread("training_chars.png")  # Đọc ảnh chứa các ký tự mẫu
    imgTrainingNumbers = cv2.resize(imgTrainingNumbers, (800, 400))  # Điều chỉnh kích thước tùy ý  
    
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang ảnh xám
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)  # Làm mờ ảnh để giảm nhiễu

    # Chuyển đổi ảnh từ ảnh xám sang ảnh đen trắng sử dụng adaptive threshold
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)

    cv2.imshow("imgThresh", imgThresh)  # Hiển thị ảnh sau khi threshold

    imgThreshCopy = imgThresh.copy()  # Tạo một bản sao của ảnh threshold

    # Tìm các đường viền trong ảnh
    npaContours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Khởi tạo mảng rỗng để lưu ảnh sau khi làm phẳng
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []  # Danh sách lưu trữ các nhãn của ký tự

    # Các ký tự hợp lệ (sử dụng mã ASCII)
    intValidChars = [ord(c) for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

    for npaContour in npaContours:  # Duyệt qua từng đường viền tìm được
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:  # Kiểm tra diện tích contour
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # Xác định hình chữ nhật bao quanh ký tự

            # Vẽ hình chữ nhật xung quanh ký tự
            cv2.rectangle(imgTrainingNumbers, (intX, intY), (intX+intW, intY+intH), (0, 0, 255), 2)

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]  # Cắt vùng chứa ký tự
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # Resize về kích thước chuẩn

            cv2.imshow("imgROI", imgROI)  # Hiển thị ảnh vùng ký tự
            cv2.imshow("imgROIResized", imgROIResized)  # Hiển thị ảnh ký tự đã resize
            cv2.imshow("training_numbers.png", imgTrainingNumbers)  # Hiển thị ảnh chứa ký tự với các ô đánh dấu

            intChar = cv2.waitKey(0)  # Chờ người dùng nhập ký tự tương ứng

            if intChar == 27:  # Nếu nhấn ESC thì thoát chương trình
                sys.exit()
            elif intChar in intValidChars:  # Nếu ký tự nằm trong danh sách ký tự hợp lệ
                intClassifications.append(intChar)  # Thêm ký tự vào danh sách nhãn
                
                # Làm phẳng ảnh ký tự để lưu trữ
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    # Chuyển danh sách nhãn thành mảng numpy
    fltClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))  # Định dạng lại để lưu trữ

    print("\n\nQuá trình huấn luyện hoàn tất!!\n")

    np.savetxt("classifications.txt", npaClassifications)  # Lưu nhãn vào file
    np.savetxt("flattened_images.txt", npaFlattenedImages)  # Lưu ảnh đã làm phẳng vào file

    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ hiển thị

if __name__ == "__main__":
    main()
