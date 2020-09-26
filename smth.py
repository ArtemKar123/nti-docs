import cv2
import numpy as np

image_file = "header0.png"
img = cv2.imread(image_file)
output = img.copy()
img = 255 - img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(gray, kernel, iterations=2)
dilation = 255 - dilation
# Get contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


for idx, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    if hierarchy[0][idx][3] == 0:
        cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

cv2.imshow("Input", img)
cv2.imshow("Enlarged", dilation)
cv2.imshow("Output", output)
cv2.waitKey(0)
