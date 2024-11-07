import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle
from skimage.feature import hog


# Dự đoán xe từ các vùng bounding box trong video
def detect_car_in_frame(image):
    features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    prediction = knn.predict([features])
    probabilities = knn.predict_proba([features])
    return prediction, probabilities

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

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

# Load model
with open('LR.pkl', 'rb') as f:
    knn = pickle.load(f)

cap = cv2.VideoCapture('video.mp4')

# Thiết lập VideoWriter để ghi đầu ra video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Hoặc 'XVID' tùy vào định dạng bạn muốn
out = cv2.VideoWriter('output_with_boxes.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

boxes = []
scores = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=1.3, fy=1.3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, window) in sliding_window(gray, step_size=8, window_size=(80, 36)):
        if window.shape[1] != 80 or window.shape[0] != 36:
            continue

        prediction, probabilities = detect_car_in_frame(window)
        if prediction == 'car' and probabilities[0][0] > 0.9:
            boxes.append([x, y, x + 80, y + 36])
            scores.append(probabilities[0][0])

    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = non_max_suppression(boxes, scores, threshold=0.1)

        for i in indices:
            (x1, y1, x2, y2) = boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Ghi khung hình vào video đầu ra
    out.write(frame)

    boxes = []
    scores = []

cap.release()
out.release()
cv2.destroyAllWindows()
