import time
import numpy as np
import argparse
import imutils
import cv2
from os import listdir
from os.path import isfile, join


class HeaderMatcher:
    def __init__(self):
        self.visualize = False
        self.MAX = 2400000
        template = cv2.imread('14ref.png')
        template2 = cv2.imread('18ref.png')
        template3 = cv2.imread('headerRef3.png')
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
        template3 = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)
        # template = 255 - template
        template = cv2.Canny(template, 50, 200)
        template2 = cv2.Canny(template2, 50, 200)
        template3 = cv2.Canny(template3, 50, 200)
        (self.tH, self.tW) = template2.shape[:2]
        self.templates = [template, template2, template3]
        self.found_1 = self.found_2 = self.found_3 = 0

    def classify(self, image):
        # orig = image.copy()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = None
        idx = -1
        skip_long = True
        # loop over the scales of the image
        if image.shape[1] == 0 or image.shape[0] == 0:
            return -1
        for scale in np.linspace(0.8, 1.2, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(image, width=int(image.shape[1] * scale))
            r = image.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < self.tH or resized.shape[1] < self.tW:
                break
            edged = cv2.Canny(resized, 130, 200)
            for i in range(len(self.templates)):
                # if i == 0 and skip_first:
                #                    continue
                template = self.templates[i]
                try:
                    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
                    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                    if found is None or maxVal > found[0]:
                        found = (maxVal, maxLoc, r)
                        idx = i
                        if i == 0:
                            self.found_1 = found[0]
                        elif i == 1:
                            self.found_2 = found[0]
                        else:
                            self.found_3 = found[0]
                except Exception as e:
                    pass
        if found is None:
            return -1
        else:
            # print(self.found_1, self.found_2, self.found_3)
            (_, maxLoc, r) = found
            # print(found[0])
            if found[0] < self.MAX:
                return -1
            else:
                if idx == 1:
                    f1 = self.found_1
                    if f1 > self.MAX:
                        return 0
                elif idx == 2:
                    f1 = self.found_1
                    f2 = self.found_2
                    if f1 > self.MAX:
                        if f1 > f2:
                            return 0
                        elif f2 > f1:
                            return 1
                    else:
                        return 2
                return idx
# hm = HeaderMatcher()
# id = hm.classify(cv2.imread('Headers/header12.png'))
# print(id)
