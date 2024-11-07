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
import xml.etree.ElementTree as ET

# Hàm mới để đọc bounding boxes từ file XML
def read_xml_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('.//object'):
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            boxes.append([xmin, ymin, xmax, ymax])
    
    return np.array(boxes)

# Hàm mới để tính IoU giữa hai bounding box
def calculate_iou(box1, box2):
    # Tính toạ độ giao điểm
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Tính diện tích phần giao
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Tính diện tích từng box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Tính diện tích hợp
    union = box1_area + box2_area - intersection
    
    # Tính IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def detect_car_in_frame(image):
    features, hog_image = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    prediction = knn.predict([features])
    probabilities = knn.predict_proba([features])
    return prediction, probabilities

def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def correlation_distance(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x_centered = x - mean_x
    y_centered = y - mean_y
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered ** 2)) * np.sqrt(np.sum(y_centered ** 2))
    return 1 - (numerator / (denominator + 1e-10))

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
with open('SVM.pkl', 'rb') as f:
    knn = pickle.load(f)

# Đọc ground truth từ file XML
xml_path = 'test123.xml'  # Thay đổi đường dẫn tới file XML của bạn
ground_truth_boxes = read_xml_annotations(xml_path)

# Khởi tạo video
cap = cv2.VideoCapture('hm9.jpg')
window_sizes = [(160, 440), (160, 400), (230, 550)]

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.9, fy=0.9)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes = []
    scores = []

    for window_size in window_sizes:
        for (x, y, window) in sliding_window(gray, step_size=8, window_size=window_size):
            if window.shape[1] != window_size[0] or window.shape[0] != window_size[1]:
                continue

            window_resized = cv2.resize(window, (88, 254))
            prediction, probabilities = detect_car_in_frame(window_resized)

            if prediction == 'human' and probabilities[0][0] > 0.95:
                print(prediction, probabilities)
                boxes.append([x, y, x + window_size[0], y + window_size[1]])
                scores.append(probabilities[0][0])

    # Áp dụng non-maximum suppression
    if len(boxes) > 0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        indices = non_max_suppression(boxes, scores, threshold=0.3)
        
        # Tính IoU cho mỗi predicted box với ground truth
        for i in indices:
            pred_box = boxes[i]
            # Vẽ predicted box màu xanh lá
            cv2.rectangle(frame, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 2)
            
            # Tính và hiển thị IoU với tất cả ground truth boxes
            for gt_box in ground_truth_boxes:
                iou = calculate_iou(pred_box, gt_box)
                # Vẽ ground truth box màu đỏ
                cv2.rectangle(frame, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 0, 255), 2)
                # Hiển thị giá trị IoU
                cv2.putText(frame, f'IoU: {iou:.2f}', 
                          (pred_box[0], pred_box[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite("check5.png", frame)
    cv2.imshow('Detected pipe', frame)
    break
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()