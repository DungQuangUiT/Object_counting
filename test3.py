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


def sliding_window(image, step_size, window_size):
    # Trả về các bounding box di chuyển qua khung hình
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])



cc = 1045
cap = cv2.VideoCapture('t.png')
output_folder = 'dataset'

boxes = []  # Danh sách các bounding box (x1, y1, x2, y2)
scores = []  # Danh sách xác suất tương ứng với mỗi bounding box
########################################################################################################
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break


    #frame = cv2.resize(frame, (0, 0), fx=1.3, fy=1.3)
    # Chuyển đổi sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Áp dụng sliding window
    for (x, y, window) in sliding_window(gray, step_size=16, window_size=(40, 44)):
        cc +=1
        #print(c)
        if window.shape[1] != 40 or window.shape[0] != 44:
            continue  # Bỏ qua nếu kích thước không khớp với window

            # Lưu ảnh phát hiện xe vào thư mục
        output_image_path = os.path.join(output_folder, f'{cc}.jpg')
        cv2.imwrite(output_image_path, frame[y:y + 44, x:x + 40])



    # Hiển thị khung hình
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite("check.png", frame)
    cv2.imshow('Detected Cars', frame)
    break
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Reset boxes và scores cho khung hình tiếp theo
    boxes = []
    scores = []

cap.release()
cv2.destroyAllWindows()