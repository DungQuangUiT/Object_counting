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
    cv2.resizeWindow("Settings", 640, 240)
    cv2.createTrackbar("Threshold1", "Settings", 90, 255, empty)
    cv2.createTrackbar("Threshold2", "Settings", 95, 255, empty)
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    thresh1 = cv2.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2)
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
            features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
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
print("###########################\nKNN classifier\n")
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11, weights='distance', n_jobs=-1, metric=correlation_distance))
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
print(f"Precision: {precision_score(y_test, prediction, labels=['car', 'no_car'], average='macro', zero_division=0)}")
print(f"Recall: {recall_score(y_test, prediction, labels=['car', 'no_car'], average='micro', zero_division=0)}")
print("Confusion matrix:")
print(confusion_matrix(y_test, prediction))

#with open('KNN.pkl', 'wb') as f:
#    pickle.dump(knn, f)

#############################################################################################
# LogisticRegression
print("###########################\nLogistic Regression classifier\n")
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(n_jobs=-1, max_iter = 10000)
lr.fit(X_train, y_train)

prediction = lr.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
print(f"Precision: {precision_score(y_test, prediction, labels=['car', 'no_car'], average='macro', zero_division=0)}")
print(f"Recall: {recall_score(y_test, prediction, labels=['car', 'no_car'], average='micro', zero_division=0)}")
print("Confusion matrix:")
print(confusion_matrix(y_test, prediction))

#with open('LR.pkl', 'wb') as f:
#    pickle.dump(lr, f)

#############################################################################################
# SVM
print("###########################\nSVM classifier\n")
from sklearn import svm

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter

svm = svm.SVC(gamma=0.1, C=10, probability=True)
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
print(f"Precision: {precision_score(y_test, prediction, labels=['car', 'no_car'], average='macro', zero_division=0)}")
print(f"Recall: {recall_score(y_test, prediction, labels=['car', 'no_car'], average='micro', zero_division=0)}")
print("Confusion matrix:")
print(confusion_matrix(y_test, prediction))

#with open('SVM.pkl', 'wb') as f:
#    pickle.dump(svm, f)

#############################################################################################
# naive bayes
print("###########################\nNaive bayes\n")
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
prediction = nb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
print(f"Precision: {precision_score(y_test, prediction, labels=['car', 'no_car'], average='macro', zero_division=0)}")
print(f"Recall: {recall_score(y_test, prediction, labels=['car', 'no_car'], average='micro', zero_division=0)}")
print("Confusion matrix:")
print(confusion_matrix(y_test, prediction))

#############################################################################################
# random forest
print("###########################\nRandom Forest classifier\n")
from sklearn.ensemble import RandomForestClassifier

nb = RandomForestClassifier(n_jobs=-1, max_depth=20, max_features=30)
nb.fit(X_train, y_train)
prediction = nb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
print(f"Precision: {precision_score(y_test, prediction, labels=['car', 'no_car'], average='macro', zero_division=0)}")
print(f"Recall: {recall_score(y_test, prediction, labels=['car', 'no_car'], average='micro', zero_division=0)}")
print("Confusion matrix:")
print(confusion_matrix(y_test, prediction))