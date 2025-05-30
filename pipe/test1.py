from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, pairwise_distances, classification_report
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

# read image
def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    #image = cv2.imread("8050.png")
    return image

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


def empty(a):
    pass

def preProcessing(img):
    cv2.namedWindow("Settings")
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    imgPre = cv2.Canny(imgPre, 90/255, 95/255)
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre

###################################################################################
DATA_PATH = 'dataset'
# find number of train images
number_of_train_image_count = 0
color_list = os.listdir(DATA_PATH)
for index, label in enumerate(color_list):
    path = os.path.join(DATA_PATH, label)
    image_list = os.listdir(os.path.join(path))
    for image_name in image_list:
        number_of_train_image_count += 1

X = []
y = []
color_list = os.listdir(DATA_PATH)
with tqdm(total=number_of_train_image_count) as pbar:
    for index, label in enumerate(color_list):
        path = os.path.join(DATA_PATH, label)
        image_list = os.listdir(os.path.join(path))
        for image_name in image_list:
            image = read_image(os.path.join(path, image_name))
            image = cv2.resize(image, (80, 36))
            #histogram_features = normalized_color_histogram(image)
            #moment_features = color_moment(image)
            #dcd_features = dominant_color_descriptor(image)
            #ccv_features = color_coherence_vector(image)

            print(' ', label, image_name)
            features, hog_image = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
            #imgPre = preProcessing(image)  #
            X.append(features)
            y.append(label)
            pbar.update(1)
            print(' ', label, image_name)
#normalize_moment_feature(X, 0, image.shape[2])
#X = np.nan_to_num(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# TRAIN #################################################################################################
#KNN
# print("#####################################\nKNN classifier\n")
# knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1, metric=correlation_distance))
# knn.fit(X_train, y_train)

# prediction = knn.predict(X_test)
# # print(f"Accuracy: {accuracy_score(y_test, prediction)}")
# # print(f"Precision: {precision_score(y_test, prediction, labels=['count', 'uncount'], average='macro', zero_division=0)}")
# # print(f"Recall: {recall_score(y_test, prediction, labels=['count', 'uncount'], average='micro', zero_division=0)}")
# # print("Confusion matrix:")
# # print(confusion_matrix(y_test, prediction))
# print(classification_report(y_test, prediction, labels=['count', 'uncount']))

#with open('KNN.pkl', 'wb') as f:
#    pickle.dump(knn, f)

import matplotlib.pyplot as plt
if False:
    # EVALUATION #################################################################################################
    accuracy_list = []

    # Kiểm tra độ chính xác cho từng giá trị của k từ 1 đến 20
    for k in range(1, 21):
        knn = make_pipeline(StandardScaler(), 
                            KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1, metric=correlation_distance))
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)
        accuracy = accuracy_score(y_test, prediction)
        accuracy_list.append(accuracy)
        print(f"k={k}, Accuracy: {accuracy}")

    # Tìm giá trị k và độ chính xác cao nhất
    best_k = np.argmax(accuracy_list) + 1  # Giá trị k tốt nhất (do index bắt đầu từ 0)
    best_accuracy = accuracy_list[best_k - 1]

    # Vẽ biểu đồ độ chính xác theo k
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), accuracy_list, marker='o', linestyle='-', color='b', label='Accuracy')

    # Đánh dấu vị trí có độ chính xác cao nhất
    plt.scatter(best_k, best_accuracy, color='red', label=f'Best k={best_k}, Acc={best_accuracy:.4f}')
    plt.text(best_k, best_accuracy, f'{best_accuracy:.4f}', fontsize=10, color='red', ha='center', va='bottom')

    # Trang trí biểu đồ
    plt.title('Accuracy vs Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 21))
    plt.grid(True)
    plt.legend()
    plt.show()

#############################################################################################
# LogisticRegression
print("#####################################\nLogistic Regression classifier\n")
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(n_jobs=-1, max_iter = 10000)
lr.fit(X_train, y_train)

prediction = lr.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, prediction)}")
# print(f"Precision: {precision_score(y_test, prediction, labels=['count', 'uncount'], average='macro', zero_division=0)}")
# print(f"Recall: {recall_score(y_test, prediction, labels=['count', 'uncount'], average='micro', zero_division=0)}")
# print("Confusion matrix:")
# print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction, labels=['count', 'uncount']))

#with open('LR.pkl', 'wb') as f:
#    pickle.dump(lr, f)

#############################################################################################
# SVM
print("#####################################\nSVM classifier\n")
from sklearn import svm

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter

svm = svm.SVC(gamma=0.1, C=100, probability=True)
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, prediction)}")
# print(f"Precision: {precision_score(y_test, prediction, labels=['count', 'uncount'], average='macro', zero_division=0)}")
# print(f"Recall: {recall_score(y_test, prediction, labels=['count', 'uncount'], average='micro', zero_division=0)}")
# print("Confusion matrix:")
# print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction, labels=['count', 'uncount']))

#with open('SVM.pkl', 'wb') as f:
#    pickle.dump(svm, f)

if True:
    from sklearn import svm
    # EVALUATION #################################################################################################

    # 1. Đánh giá độ chính xác theo gamma (với C cố định)
    gamma_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]  # Khoảng giá trị cho gamma
    accuracy_gamma = []

    for gamma in gamma_values:
        svm_model = svm.SVC(gamma=gamma, C=100, probability=True)
        svm_model.fit(X_train, y_train)
        prediction = svm_model.predict(X_test)
        acc = accuracy_score(y_test, prediction)
        accuracy_gamma.append(acc)
        print(f"Gamma={gamma}, Accuracy: {acc}")

    # Vẽ biểu đồ độ chính xác theo gamma
    plt.figure(figsize=(10, 6))
    plt.plot(gamma_values, accuracy_gamma, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.xscale('log')  # Log scale để dễ quan sát khi gamma có giá trị chênh lệch lớn
    plt.title('Accuracy vs Gamma (C=100)')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.scatter(0.1, accuracy_gamma[gamma_values.index(0.1)], color='red', label=f'Gamma=0.1, Acc={accuracy_gamma[gamma_values.index(0.1)]:.4f}')
    plt.text(0.1, accuracy_gamma[gamma_values.index(0.1)], f'{accuracy_gamma[gamma_values.index(0.1)]:.4f}', fontsize=10, color='red', ha='center', va='bottom')
    plt.show()

    # 2. Đánh giá độ chính xác theo C (với gamma cố định)
    C_values = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]  # Khoảng giá trị cho C
    accuracy_C = []

    for C in C_values:
        svm_model = svm.SVC(gamma=0.1, C=C, probability=True)
        svm_model.fit(X_train, y_train)
        prediction = svm_model.predict(X_test)
        acc = accuracy_score(y_test, prediction)
        accuracy_C.append(acc)
        print(f"C={C}, Accuracy: {acc}")

    # Vẽ biểu đồ độ chính xác theo C
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, accuracy_C, marker='o', linestyle='-', color='g', label='Accuracy')
    plt.xscale('log')  # Log scale để dễ quan sát khi C có giá trị chênh lệch lớn
    plt.title('Accuracy vs C (Gamma=0.1)')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.scatter(100, accuracy_C[C_values.index(100)], color='red', label=f'C=100, Acc={accuracy_C[C_values.index(100)]:.4f}')
    plt.text(100, accuracy_C[C_values.index(100)], f'{accuracy_C[C_values.index(100)]:.4f}', fontsize=10, color='red', ha='center', va='bottom')
    plt.show()

# #############################################################################################
# # naive bayes
# print("#####################################\nNaive bayes\n")
# from sklearn.naive_bayes import GaussianNB

# nb = GaussianNB()
# nb.fit(X_train, y_train)
# prediction = nb.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, prediction)}")
# print(f"Precision: {precision_score(y_test, prediction, labels=['count', 'uncount'], average='macro', zero_division=0)}")
# print(f"Recall: {recall_score(y_test, prediction, labels=['count', 'uncount'], average='micro', zero_division=0)}")
# print("Confusion matrix:")
# print(confusion_matrix(y_test, prediction))

# #############################################################################################
# # random forest
# print("#####################################\nRandom Forest classifier\n")
# from sklearn.ensemble import RandomForestClassifier

# nb = RandomForestClassifier(n_jobs=-1, max_depth=20, max_features=30)
# nb.fit(X_train, y_train)
# prediction = nb.predict(X_test)
# print(f"Accuracy: {accuracy_score(y_test, prediction)}")
# print(f"Precision: {precision_score(y_test, prediction, labels=['count', 'uncount'], average='macro', zero_division=0)}")
# print(f"Recall: {recall_score(y_test, prediction, labels=['count', 'uncount'], average='micro', zero_division=0)}")
# print("Confusion matrix:")
# print(confusion_matrix(y_test, prediction))