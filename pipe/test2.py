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


#load model
with open('LR.pkl', 'rb') as f:
    knn = pickle.load(f)
#cc = 3000
cap = cv2.VideoCapture('pipe1.jpg')
#output_folder = 'dataset'

boxes = []  # Danh sách các bounding box (x1, y1, x2, y2)
scores = []  # Danh sách xác suất tương ứng với mỗi bounding box
########################################################################################################
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break


    #frame = cv2.resize(frame, (0, 0), fx=1.3, fy=1.3)

    c = 0
    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Áp dụng sliding window
    for (x, y, window) in sliding_window(gray, step_size=8, window_size=(40, 44)):
        c += 1
        #cc +=1
        #print(c)
        if window.shape[1] != 40 or window.shape[0] != 44:
            continue  # Bỏ qua nếu kích thước không khớp với window


        window = cv2.resize(window, (80, 36))
        prediction, probabilities = detect_car_in_frame(window)

        if prediction == 'count' and probabilities[0][0] > 0.9:
            print(prediction, probabilities)
            # Thêm bounding box và xác suất vào danh sách
            temp = frame
            boxes.append([x, y, x + 40, y + 44])

            # Lưu ảnh phát hiện xe vào thư mục
            #output_image_path = os.path.join(output_folder, f'{cc}.png')
            #cv2.imwrite(output_image_path, frame[y:y + 32, x:x + 80])

            scores.append(probabilities[0][0])

    # Áp dụng non-maximum suppression
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = non_max_suppression(boxes, scores, threshold=0.2)  # Thay đổi threshold nếu cần

        # Vẽ các bounding box sau khi áp dụng NMS
        for i in indices:
            (x1, y1, x2, y2) = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)



    # Hiển thị khung hình
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite("check2.png", frame)
    cv2.imshow('Detected pipe', frame)
    break
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Reset boxes và scores cho khung hình tiếp theo
    boxes = []
    scores = []

cap.release()
cv2.destroyAllWindows()