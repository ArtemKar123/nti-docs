import cv2
import sys
import numpy as np


def nothing(x):
    pass


cv2.namedWindow('image')
# Load in image
# image = cv2.imread('im54.png')
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)  # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 179, 179, nothing)
cv2.createTrackbar('SMax', 'image', 255, 255, nothing)
cv2.createTrackbar('VMax', 'image', 255, 255, nothing)

# Set default value for MAX HSV trackbars.
# cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = 0
hMax = 179
sMax = vMax = 255

phMin = psMin = pvMin = phMax = psMax = pvMax = 0

image = cv2.imread('im1.jpg')
output = image

cv2.namedWindow('image_out', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image_out', 248*2, 700)
cv2.namedWindow('image_out1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image_out1', 248*2, 700)
wait_time = 33
main_contour = None
while (1):
    # create trackbars for color change
    #image = cv2.imread('im54.png')
    output = image.copy()

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')

    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output1 = cv2.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
    if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
            hMin, sMin, vMin, hMax, sMax, vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv2.inRange(hsv, lower, upper)  # применяем цветовой фильтр
    contours0, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours0 = sorted(contours0, key=lambda _x: cv2.contourArea(_x))
    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        break
       # cnt = contours0[-1]
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольникpp
        #  x, y, w, h = cv2.boundingRect(cnt)
        #  if w*h > 1000:
        #      main_contour = x, y, w, h
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        cv2.drawContours(output, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник
    # break
    # output = output[min(box[1][1], box[2][1]):max(box[3][1]), box[]]
    #if main_contour is not None:
    #    x, y, w, h = main_contour
    #    output1 = output1[y:y + h, x:x + w]
    # Display output image
    #cv2.imshow('image', output)
    # Display output image
    #cv2.imshow('image_out', output)
    cv2.imshow('image_out1', output1)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
