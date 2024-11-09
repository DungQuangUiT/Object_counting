from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, pairwise_distances
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from skimage.feature import hog
import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle

# Dự đoán xe từ các vùng bounding box trong video
def detect_car_in_frame(image):
    features, hog_image = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    prediction = knn.predict([features])
    probabilities = knn.predict_proba([features])
    return prediction, probabilities

def sliding_window(image, step_size, window_size):
    # Trả về các bounding box di chuyển qua khung hình
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def correlation_distance(x, y):
    # cal mean vector
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # - mean
    x_centered = x - mean_x
    y_centered = y - mean_y
    
    # multiply
    numerator = np.sum(x_centered * y_centered)
    
    # denominator
    denominator = np.sqrt(np.sum(x_centered ** 2)) * np.sqrt(np.sum(y_centered ** 2))
    
    return 1 - (numerator / (denominator + 1e-10))  # use epsilon avoid 0

# Non-Maximum Suppression (NMS) để loại bỏ các bounding box bị overlap
def non_max_suppression(boxes, scores, threshold=0.5):
    if len(boxes) == 0:
        return []
    
    # Lấy các tọa độ của các bounding box
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Tính diện tích của các bounding box và sắp xếp theo score
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []  # Danh sách các box giữ lại
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Tính diện tích giao (intersection)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        # Tính diện tích hợp (union)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Chọn các box có tỷ lệ overlap nhỏ hơn threshold
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

# Load model
with open('SVM.pkl', 'rb') as f:
    knn = pickle.load(f)

# Khởi tạo video
cap = cv2.VideoCapture('hm11.jpg')

#cc = 26000
output_folder = 'dataset'

# Các kích thước sliding window khác nhau
#hm3 = [(180, 450), (160, 400), (230, 650)]     # scale = 1
#hm4 = [(170, 450), (110, 400), (230, 650)]     # scale = 1
#hm5 = [(170, 450), (150, 340), (230, 650)]     # scale = 1.8
#hm6 = [(200, 470), (270, 500), (240, 560)]    # scale = 1
#hm7 = [(150, 380), (170, 450), (140, 410)]    # scale = 1
#hm8 = [(240, 500), (180, 460), (220, 370)]    # scale = 1
#hm9 = [(190, 530), (150, 480), (340, 730)]    # scale = 1
#hm10 = [(55, 150), (60, 210), (110, 230)]    # scale = 1
#hm11 = [(130, 320), (150, 320), (200, 370)]    # scale = 1
window_sizes = [(130, 320), (150, 320), (200, 370)]  # Có thể thay đổi thêm các kích thước khác

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes = []  # Danh sách các bounding box (x1, y1, x2, y2)
    scores = []  # Danh sách xác suất tương ứng với mỗi bounding box

    # Áp dụng sliding window với nhiều kích thước
    for window_size in window_sizes:
        for (x, y, window) in sliding_window(gray, step_size=8, window_size=window_size):
            #cc +=1
            if window.shape[1] != window_size[0] or window.shape[0] != window_size[1]:
                continue  # Bỏ qua nếu kích thước không khớp với window

            # Resize cửa sổ để chuẩn hóa kích thước đầu vào
            window_resized = cv2.resize(window, (88, 254))
            prediction, probabilities = detect_car_in_frame(window_resized)

            if prediction == 'human' and probabilities[0][0] > 0.96:
                print(prediction, probabilities)
                boxes.append([x, y, x + window_size[0], y + window_size[1]])
                scores.append(probabilities[0][0])
                #output_image_path = os.path.join(output_folder, f'{cc}.jpg')
                #cv2.imwrite(output_image_path, frame[y:y + window_size[1], x:x + window_size[0]])

    # Áp dụng non-maximum suppression
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = non_max_suppression(boxes, scores, threshold=0.3)  # Thay đổi threshold nếu cần

        # Vẽ các bounding box sau khi áp dụng NMS
        for i in indices:
            (x1, y1, x2, y2) = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Hiển thị khung hình
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite("hm11_detect.jpg", frame)
    cv2.imshow('Detected pipe', frame)
    break
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()