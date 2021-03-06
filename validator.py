import time
import numpy as np
import argparse
import imutils
import cv2
from os import listdir
from os.path import isfile, join


# print(img.shape)
# mouseX, mouseY = 0, 0


class Validator:
    def __init__(self):
        self.zones14 = [[(24, 44), (1070, 74)], [(134, 95), (185, 125)], [(215, 91), (280, 125)],
                        [(356, 98), (394, 124)],
                        [(411, 93), (474, 124)], [(482, 91), (591, 124)], [(621, 90), (1072, 125)],
                        [(329, 168), (1075, 190)],
                        [(3, 280), (706, 313)], [(4, 422), (1082, 440)], [(377, 485), (438, 520)],
                        [(465, 485), (588, 520)],
                        [(661, 485), (682, 520)], [(699, 485), (767, 520)], [(790, 485), (1052, 520)],
                        [(234, 530), (875, 560)],
                        [(7, 1395), (380, 1424)], [(405, 1395), (537, 1424)], [(849, 1395), (883, 1424)],
                        [(905, 1395), (986, 1424)], [(1031, 1395), (1056, 1424)], ]
        self.zones18 = [[(25, 25), (1076, 52)], [(139, 90), (192, 114)], [(222, 90), (284, 114)],
                        [(361, 90), (396, 114)],
                        [(414, 90), (477, 114)], [(482, 90), (596, 114)], [(626, 90), (1073, 114)],
                        [(328, 143), (1076, 175)],
                        [(235, 196), (1086, 219)], [(27, 324), (1077, 349)], [(131, 386), (185, 408)],
                        [(215, 386), (278, 408)],
                        [(355, 386), (394, 408)], [(412, 386), (472, 408)], [(480, 386), (592, 408)],
                        [(617, 386), (1074, 408)],
                        [(327, 443), (1075, 468)], [(410, 532), (473, 554)], [(498, 532), (577, 554)],
                        [(654, 532), (672, 554)],
                        [(691, 532), (752, 554)], [(777, 532), (1077, 554)], [(38, 572), (1076, 594)],
                        [(1, 1351), (242, 1382)],
                        [(265, 1351), (407, 1382)], [(673, 1351), (713, 1382)], [(731, 1351), (809, 1382)],
                        [(850, 1351), (879, 1382)], [(2, 1396), (241, 1425)], [(266, 1396), (408, 1425)],
                        [(807, 1396), (847, 1425)],
                        [(863, 1396), (944, 1425)], [(985, 1396), (1014, 1425)]]

        # self.zones_sogl = [[(4, 69), (885, 88)], [(4, 109), (1087, 124)], [(71, 129), (917, 144)],
        #                    [(4, 148), (247, 165)], [(5, 169), (876, 185)], [(3, 208), (1085, 222)],
        #                    [(797, 1360), (913, 1394)],
        #                    [(974, 1360), (1085, 1394)], ]
        self.zones_sogl = [[(4, 74), (885, 88)], [(4, 114), (1087, 128)], [(71, 133), (917, 147)],
                           [(4, 154), (247, 167)], [(3, 174), (876, 189)], [(3, 212), (1085, 227)],
                           [(797, 1382), (913, 1427)],
                           [(975, 1382), (1081, 1427)], ]
        self.zones = [self.zones14, self.zones18, self.zones_sogl]
        self.optional_zones = [[8], [22], [4]]
        pass

    def find_contour(self, im, accuracy=1):
        a = im
        MAX = 245
        rows, cols = a.shape[0], a.shape[1]
        topI = 0
        botI = rows
        leftJ = 0
        rightJ = cols
        for i in range(int(rows / accuracy), 0, -1):
            zone = a[(i - 1) * accuracy:i * accuracy, int(cols / 10):int(9 * cols / 10)]
            mean = zone.mean()
            # prev_mean = a[i*accuracy:i*accuracy+accuracy, int(cols/10):int(9*cols/10)].mean()

            if mean < MAX:  # and prev_mean > 254:
                botI = i
                break
        #    print(topI, botI, leftJ, rightJ)
        half = int(accuracy / 2)
        a = a[topI * accuracy - half:botI * accuracy - self.top_offset // 2,
            leftJ * accuracy - half:rightJ * accuracy - self.left_offset]
        return a

    def no_lines(self, im):

        gray = cv2.bitwise_not(im)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        horizontal = np.copy(bw)
        cols = horizontal.shape[1]
        horizontal_size = cols // 50
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        final = 255 - (bw - horizontal)
        return final

    def find_offset(self, im, accuracy=1):
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

    def validate(self, image, code):
        image = cv2.resize(image, (1095, 1435), interpolation=cv2.INTER_LINEAR_EXACT)
        image = self.no_lines(image)
        self.top_offset, self.left_offset = self.find_offset(image)
        self.top_offset -= 2
        self.left_offset -= 1
        # print(top_offset, left_offset)
        # start = time.time()
        c = 0
        zones = self.zones[code]
        is_valid = True
        spec_for18 = 0
        valid_zones = []
        for zone in zones:
            # print(c)
            zone = (
                (zone[0][0] + self.left_offset, zone[0][1] + self.top_offset),
                (zone[1][0] + self.left_offset, zone[1][1] + self.top_offset))
            slc = image[zone[0][1]:zone[1][1], zone[0][0]:zone[1][0]]
            # slc = cv2.cvtColor(slc, cv2.COLOR_BGR2GRAY)
            slc = self.find_contour(slc)

            if slc.shape[0] <= 0 or slc.shape[1] <= 0:
                continue
            mean = np.mean(slc)
            # print(mean)
            if mean > 250:
                if code == 1:
                    if c in range(17, 23):
                        valid_zones.append(0)
                    else:
                        return False
                else:
                    return False
                    pass
            else:
                valid_zones.append(1)
            if c == 22 and code == 1:
                mi = np.min(valid_zones[17:22])
                # print(mi, valid_zones[22])
                if mi == 1 or (mi == 0 and valid_zones[22] == 1):
                    pass
                else:
                    return False
                # print('mean', np.mean(slc), np.median(slc))
            # rect = ((zone[0][0], zone[0][1]), (zone[1][0], zone[1][1]), 0)
            # im = img[zone[0][1]:zone[1][1], zone[0][0]:zone[1][0]]
            # print(c)
            c += 1
            # output = cv2.rectangle(image, zone[0], zone[1], (0, 0, 255), 2)
        # cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('slice')
        # cv2.imshow('contour', output)
        # cv2.imshow('slice', slc)
        # cv2.waitKey()
        # # print(time.time() - start)
        return is_valid

# img = cv2.imread('1.png')
# img = cv2.resize(img, (1095, 1435), interpolation=cv2.INTER_LINEAR_EXACT)
#
#
# vald = Validator()
# print(vald.validate(img, 0))
#
# # print(im.shape)
# # PARAM = 75
# # for i in range(im.shape[0] // PARAM):
# #     cv2.imshow('image', im[i * PARAM:i * PARAM + PARAM, :])
# #     cv2.waitKey()
#
# #
# def draw_circle(event, x, y, flags, param):
#     global mouseX, mouseY
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
#         mouseX, mouseY = x, y
#         print(mouseX, mouseY)
#
#
# #
# #
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.setMouseCallback('image', draw_circle)
# #
# while (1):
#     cv2.imshow('image', img)
#     k = cv2.waitKey(20) & 0xFF
#     if k == 27:
#         break
#     elif k == ord('a'):
#         print(mouseX, mouseY)
#
