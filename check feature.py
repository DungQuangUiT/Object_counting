import cv2
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder
from skimage.feature import hog


totalMoney = 0

myColorFinder = ColorFinder(False)
# Custom Orange Color
hsvVals = {'hmin': 0, 'smin': 0, 'vmin': 145, 'hmax': 63, 'smax': 91, 'vmax': 255}



def empty(a):
    pass


cv2.namedWindow("Settings")
cv2.resizeWindow("Settings", 640, 240)
cv2.createTrackbar("Threshold1", "Settings", 162, 255, empty)
cv2.createTrackbar("Threshold2", "Settings", 55, 255, empty)


def preProcessing(img):
    imgPre = cv2.GaussianBlur(img, (5, 5), 3)
    thresh1 = cv2.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv2.getTrackbarPos("Threshold2", "Settings")
    imgPre = cv2.Canny(imgPre, thresh1, thresh2)
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv2.dilate(imgPre, kernel, iterations=1)
    imgPre = cv2.morphologyEx(imgPre, cv2.MORPH_CLOSE, kernel)

    return imgPre


def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    return image

img = cv2.imread("pipe1.jpg")


image = read_image("pipe1.jpg")

while True:
    imgPre = preProcessing(image)
    #imgContours, conFound = cvzone.findContours(imgPre, imgPre, minArea=20)


    #image = cv2.resize(img, (80, 36))
    features, hog_image = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

    cv2.imshow("Image", hog_image)
    # cv2.imshow("imgColor", imgColor)
    cv2.waitKey(1)