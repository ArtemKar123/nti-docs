import time
import numpy as np
import argparse
import imutils
import cv2
from os import listdir
from os.path import isfile, join

visualize = False
template = cv2.imread('18.png')
template2 = cv2.imread('14.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
# template = 255 - template
template = cv2.Canny(template, 50, 200)
template2 = cv2.Canny(template2, 50, 200)
(tH, tW) = template2.shape[:2]
# (tH1, tW1) = template.shape[:2]
# cv2.imshow("Template", template)
# cv2.imshow("Templat2", template2)
# cv2.waitKey()
files = [f for f in listdir('Headers/') if isfile(join('Headers/', f))]
names = ['header8.png']
# names = ['header0.png']
# names = ['8.png', '10.png', '9.png', '1.png', '2.png']
doc_names = ['до 18', 'до 14']
templates = [template, template2]
cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Visualize', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Final', 720, 1280)
times = []
for name in files:
    print(name)
    start = time.time()
    image = cv2.imread(f'Headers/{name}')
    # print(image.mean())
    # ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = found1 = found2 = None
    skipFirst = False
    idx = -1
    # loop over the scales of the image
    for scale in np.linspace(0.8, 1.2, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        edged = cv2.Canny(resized, 130, 200)
        for i in range(len(templates)):
            if i == 0 and skipFirst:
                continue
            template = templates[i]
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if visualize:
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                if i == 0:
                    found1 = found
                else:
                    found2 = found
                idx = i
    if found is None:
        pass  # print('wrong')
    else:
        (_, maxLoc, r) = found
        print(found[0])
        # print(time.time()-start)
        if found[0] < 2500000:
            pass  # print('wrong')
        else:
            print(doc_names[idx])
        times.append(time.time() - start)
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # # draw a bounding box around the detected result and display the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.imshow("Final", orig)
        cv2.waitKey(0)
print(np.mean(times))
