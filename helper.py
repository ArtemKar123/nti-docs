import time
import numpy as np
import argparse
import imutils
import cv2
from os import listdir
from os.path import isfile, join

img = cv2.imread('13.png')

img = cv2.resize(img, (1095, 1435), interpolation=cv2.INTER_LINEAR_EXACT)
print(img.shape)
mouseX, mouseY = 0, 0


def find_contour(im, accuracy=1):
    a = im
    MAX = 245
    rows, cols = a.shape[0], a.shape[1]
    topI = 0
    botI = rows
    leftJ = 0
    rightJ = cols
    for i in range(int(rows / accuracy), 0, -1):
        mean = a[(i - 1) * accuracy:i * accuracy, int(cols / 10):int(9 * cols / 10)].mean()
        # prev_mean = a[i*accuracy:i*accuracy+accuracy, int(cols/10):int(9*cols/10)].mean()

        if mean < MAX:  # and prev_mean > 254:
            botI = i
            break
    #    print(topI, botI, leftJ, rightJ)
    half = int(accuracy / 2)
    a = a[topI * accuracy - half:botI * accuracy - 5, leftJ * accuracy - half:rightJ * accuracy + half]
    return a


def find_offset(im, accuracy=1):
    a = im
    MAX = 254
    rows, cols = a.shape[0], a.shape[1]
    topI = leftJ = 0
    for i in range(int(rows / accuracy)):
        mean = a[i * accuracy:i * accuracy + accuracy, 500:575].mean()
        # prev_mean = a[i*accuracy-accuracy:i*accuracy, int(cols/10):int(9*cols/10)].mean()
        if mean < MAX:  # and prev_mean > 254:
            topI = i
            break
    for j in range(int(cols / accuracy)):
        mean = a[710:775, j * accuracy:j * accuracy + accuracy].mean()
        # prev_mean = a[int(rows/10):int(9*rows/10), j*accuracy-accuracy:j*accuracy].mean()
        if mean < MAX:  # and prev_mean > 254:
            leftJ = j
            break
    #    print(topI, botI, leftJ, rightJ)
    return topI, leftJ


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        mouseX, mouseY = x, y
        print(mouseX, mouseY)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', draw_circle)

while (0):
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX, mouseY)

top_offset, left_offset = find_offset(img)
top_offset -= 2
left_offset -= 1
print(top_offset, left_offset)
zones = [[(23, 27), (1072, 49)], [(133, 62), (186, 85)], [(214, 62), (280, 83)], [(356, 65), (395, 84)],
         [(409, 60), (594, 82)], [(619, 60), (1077, 82)], [(329, 125), (1077, 148)],
         [(655, 230), (1073, 252)], [(3, 288), (1077, 314)],
         [(1, 485), (1087, 507)], [(376, 551), (438, 573)], [(465, 550), (588, 572)], [(662, 550), (682, 572)],
         [(697, 550), (768, 572)],
         [(790, 545), (1055, 569)], [(237, 620), (1088, 639)], [(3, 1404), (378, 1428)], [(402, 1404), (544, 1428)],
         [(849, 1404), (887, 1428)], [(905, 1404), (987, 1428)], [(1024, 1404), (1055, 1428)]]
for zone in zones:
    zone = ((zone[0][0] + left_offset, zone[0][1] + top_offset), (zone[1][0] + left_offset, zone[1][1] + top_offset))
    slc = img[zone[0][1]:zone[1][1], zone[0][0]:zone[1][0]]
    slc = cv2.cvtColor(slc, cv2.COLOR_BGR2GRAY)
    slc = find_contour(slc)
    print('mean', np.mean(slc), np.median(slc))
    # rect = ((zone[0][0], zone[0][1]), (zone[1][0], zone[1][1]), 0)
    # im = img[zone[0][1]:zone[1][1], zone[0][0]:zone[1][0]]
    # output = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    output = cv2.rectangle(img, zone[0], zone[1], (0, 0, 255), 2)
    cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
    cv2.namedWindow('slice')
    cv2.imshow('contour', output)
    cv2.imshow('slice', slc)
    cv2.waitKey()
# print(im.shape)
# PARAM = 75
# for i in range(im.shape[0] // PARAM):
#     cv2.imshow('image', im[i * PARAM:i * PARAM + PARAM, :])
#     cv2.waitKey()
