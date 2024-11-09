from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, pairwise_distances
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from skimage.feature import hog
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle

# Dự đoán xe từ các vùng bounding box trong video
def detect_car_in_frame(image):
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    prediction = knn.predict([features])
    probabilities = knn.predict_proba([features])
    return prediction, probabilities

def sliding_window(image, step_size, window_size):
    # Trả về các bounding box di chuyển qua khung hình
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def correlation_distance(x, y):
    # Tính vector trung bình
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Trừ đi giá trị trung bình
    x_centered = x - mean_x
    y_centered = y - mean_y
    
    # Nhân
    numerator = np.sum(x_centered * y_centered)
    
    # Mẫu số
    denominator = np.sqrt(np.sum(x_centered ** 2)) * np.sqrt(np.sum(y_centered ** 2))
    
    return 1 - (numerator / (denominator + 1e-10))  # sử dụng epsilon tránh chia cho 0

# Non-Maximum Suppression (NMS) để loại bỏ các bounding box bị overlap
def non_max_suppression(boxes, scores, threshold=0.5):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep

# Tải model
with open('LR.pkl', 'rb') as f:
    knn = pickle.load(f)

cap = cv2.VideoCapture('video.mp4')

# Khởi tạo biến
first_frame_saved = False
boxes = []
scores = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Lưu frame đầu tiên
    if not first_frame_saved:
        cv2.imwrite("first_frame.png", frame)
        first_frame_saved = True
        #break 
    frame = cv2.resize(frame, (0, 0), fx=1.3, fy=1.3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Áp dụng sliding window
    for (x, y, window) in sliding_window(gray, step_size=8, window_size=(80, 36)):
        if window.shape[1] != 80 or window.shape[0] != 36:
            continue
        prediction, probabilities = detect_car_in_frame(window)
        if prediction == 'car' and probabilities[0][0] > 0.9:
            print(prediction, probabilities)
            boxes.append([x, y, x + 80, y + 36])
            scores.append(probabilities[0][0])

    # Áp dụng non-maximum suppression
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = non_max_suppression(boxes, scores, threshold=0.1)
        for i in indices:
            (x1, y1, x2, y2) = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("LR_0.9.png", frame)
    cv2.imshow('Detected Cars', frame)
    break
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Reset boxes và scores cho khung hình tiếp theo
    boxes = []
    scores = []

cap.release()
cv2.destroyAllWindows()
