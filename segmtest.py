import cv2
import numpy as np

img = cv2.imread('header0.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
img_erode = cv2.erode(thresh, np.ones((1, 1), np.uint8), iterations=1)

# Get contours
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

output = img.copy()

for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    if hierarchy[0][idx][3] == 0:
        if w < 1:
            print(w)
            prev = contours[idx + 1]
            (x1, y1, w1, h1) = cv2.boundingRect(prev)
            cv2.rectangle(output, (x1, y1), (x1 + w1, y1 + h1), (70, 0, 0), 5)
            #cv2.rectangle(output, (x // 2 + x1 // 2, y // 2 + y1 // 2), (x1 + w + x, y1 + y + h), (70, 0, 0), 1)
        else:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

cv2.imshow("Input", img)
cv2.imshow("Enlarged", img_erode)
cv2.imshow("Output", output)
cv2.waitKey(0)
