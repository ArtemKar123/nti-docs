import os
from matcher import HeaderMatcher
import cv2
import numpy as np
import time
import argparse
from validator import Validator
from os import listdir
from os.path import isfile, join
from pdf2image import convert_from_path


def find_contour(im, sec, accuracy=10, mode=1):
    a = im.copy()
    top_found = False
    bot_found = False
    left_found = False
    right_found = False
    MAX = 245
    rows, cols = a.shape[0], a.shape[1]
    topI = 0
    botI = rows
    leftJ = 0
    rightJ = cols
    if mode != 2:
        prev_mean = 0
        for i in range(int(rows / accuracy) // 2):
            mean = a[i * accuracy:i * accuracy + accuracy, :].mean()
            # prev_mean = a[i*accuracy-accuracy:i*accuracy, int(cols/10):int(9*cols/10)].mean()

            if mean < MAX:
                topI = i
                break
        prev_mean = 0
        if mode != 3:
            for i in range(int(rows / accuracy), int(rows / accuracy) // 2, -1):
                mean = a[(i - 1) * accuracy:i * accuracy, :].mean()
                # prev_mean = a[i*accuracy:i*accuracy+accuracy, int(cols/10):int(9*cols/10)].mean()

                if mean < MAX:  # and prev_mean > 254:
                    botI = i
                    break
        # prev_mean = mean

    prev_mean = 0
    # print(a.shape)
    for j in range(int(cols / accuracy) // 2):
        mean = a[:, j * accuracy:j * accuracy + accuracy].mean()
        if mean < MAX:
            leftJ = j
            break

    prev_mean = 0
    for j in range(int(cols / accuracy), int(cols / accuracy) // 2, -1):
        mean = a[:, (j - 1) * accuracy:j * accuracy].mean()
        # prev_mean = a[int(rows/10):int(9*rows/10), j*accuracy:j*accuracy+accuracy].mean()
        if mean < MAX:
            rightJ = j
            break

    #    print(topI, botI, leftJ, rightJ)
    half = int(accuracy / 2)
    a = a[topI * accuracy - half:botI * accuracy + half, leftJ * accuracy - half:rightJ * accuracy + half]
    if mode != 2:
        sec = sec[topI * accuracy - half:botI * accuracy + half, leftJ * accuracy - half:rightJ * accuracy + half]
    return a, sec


def get_max_y(box, mode=0):
    m = 0
    mi = 10000
    for b in box:
        y = b[1]
        if y > m:
            m = y
        if y < mi:
            mi = y
    if mode == 0 and (m - mi > 50 or m - mi < 0):
        return -1
    else:
        return m


def get_width(box):
    m = 0
    mi = 10000
    for b in box:
        x = b[0]
        if b[0] > m:
            m = x
        if b[0] < mi:
            mi = x
    return m - mi


def find_lines(im):
    im = cv2.bitwise_not(im)
    horizontal = np.copy(im)
    out = horizontal.copy()
    # out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: get_max_y(cv2.boxPoints(cv2.minAreaRect(x))), reverse=True)[:]
    # cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        w = get_width(box)  # 255 259
        # print(w)
        coef = w / cols
        # print(coef)
        if 0.099 < coef < 0.14:
            # print(coef)
            # print(w / cols)
            # out = cv2.drawContours(out, [box], 0, (0, 0, 255), 3)
            # cv2.imshow('out', out)
            # cv2.waitKey()
            return get_max_y(box, mode=1)
            # print(y)
        # break
    return 1000000


def process_image(name):
    if name[-4:] == '.pdf':
        pages = convert_from_path(name, 200)
        if len(pages) != 1:
            return -1, -1
        img = pages[0]
        img = np.array(img)
    else:
        img = cv2.imread(name)
    # mean = img.mean()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if med < 250:
    # print(img.mean())
    secondary = img.copy()

    dilated_img = cv2.dilate(img, np.ones((4, 4), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    ret3, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, img = cv2.threshold(img, img.mean() - 20, 255, cv2.THRESH_BINARY)
    # img, _s = find_contour(img, img)
    # cv2.imwrite('pre0.png', img)
    orig = img.copy()
    img = 255 - img
    kernel = np.ones((10, 10), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=5)
    # dilation = 255 - dilation
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    angles = []
    # print(len(contours))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]
    # if angle > -89 and angle < 89:
    if angle < -45:
        angle = (90 + angle)

    # print(angle, angles)
    rows, cols = orig.shape[0], orig.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(orig, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    secondary = cv2.warpAffine(secondary, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    img = dst.copy()
    # output = img.copy()

    img = 255 - img
    kernel = np.ones((10, 10), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=5)
    # dilation = 255 - dilation
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #    print(len(contours))
    contours = sorted(contours, key=lambda x: cv2.contourArea(np.int0(cv2.boxPoints(cv2.minAreaRect(x)))),
                      reverse=True)[:]
    # na = cv2.contourArea(np.int0(cv2.boxPoints(cv2.minAreaRect(contours[1]))))
    # if na > 1000000 or na < 100000:
    #        contours = contours[1:]
    minX = minY = 100000
    maxX = maxY = 0
    # cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    counted = 0
    # total_area = 0

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(cnt)
        box_area = cv2.contourArea(box)
        w = get_width(box)
        # per = cv2.arcLength(cnt, True)
        cnt_coef = box_area / area
        # print(cnt_coef, box_area, box_area/area, box_area/cnt_coef)
        if area < 50000:
            break
        # print(w, w/dst.shape[1], dst.shape)
        if w < dst.shape[1] * 0.5 or cnt_coef > 2:
            continue
        # total_area += area
        counted += 1
        for b in box:
            x = b[0]
            y = b[1]
            if x < minX: minX = x
            if y < minY: minY = y
            if x > maxX: maxX = x
            if y > maxY: maxY = y

        # orig = orig[:, cx - w // 2: cx + w // 2]
        # secondary = secondary[:, cx - w // 2: cx + w // 2]
        # output = cv2.drawContours(output, [cnt], 0, (0, 0, 255), 3)
        # cv2.imshow('out', output)
        # cv2.waitKey()
        # cv2.imwrite('out.png', output)
        # print(2)
    # print(total_area)
    # print(img.shape, dst.shape)
    if counted >= 1:
        w = int((maxX - minX) * 1.05)
        h = int((maxY - minY) * 1.15)
        shape = dst.shape
        if w > shape[1] or w < shape[1] / 2:
            w = shape[1]
            cx = shape[1] // 2
        else:
            cx = (maxX + minX) // 2
        if h > shape[0] or h < shape[0] / 2:
            h = shape[0]
            cy = shape[0] // 2
        else:
            cy = (maxY + minY) // 2
        # print(w, h, cx, cy)
        newH1 = cy - h // 2
        newH2 = cy + h // 2
        newW1 = cx - w // 2
        newW2 = cx + w // 2
        if newH1 < 0: newH1 = 0
        if newW1 < 0: newW1 = 0
        dst = dst[newH1: newH2, newW1: newW2]
        secondary = secondary[newH1: newH2, newW1: newW2]
        # cv2.imwrite('rotated.png', dst)
    # cv2.imwrite('dilated.png', output)

    # print(dst.shape)
    dst, secondary = find_contour(dst, secondary, mode=3)

    y = find_lines(dst)
    y += 15
    _h = dst.shape[0]
    # print(y, dst.shape)
    if y > _h:
        y = _h
    dst = dst[:y, :]
    secondary = secondary[:y, :]
    # cv2.imwrite('dst.png', dst)
    # cv2.imwrite('croped.png', dst)
    return dst, secondary


# names = ['cat.jpg', 'im54.png', 'yellow.jpg', '1Doc.jpg', 'mirror.png', 'im12.png', 'im13.png',
#          'im1.png', 'test.png', 'real-0.jpg', 'real-1.jpg', 'wrong.png', 'wrong1.png', 'wrong3.png',
#          'real-0.jpg', 'real-1.jpg', 'real-3.jpg']
# names = ['cat.jpg', 'im54.png', 'wrong1.png', 'real-6.jpg', 'real-8.jpg', 'real-9.jpg']
# MAX_HEADER_DIFF = 10000
count = 0

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="Path to image")
args = vars(ap.parse_args())

vd = Validator()
times = []
hm = HeaderMatcher()
doc_names = ['до 14', 'до 18', 'согласие']

dir_path = args['path']
# print(dir_path)
names = [os.path.join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))]
# print(names)
for n in names:
    print(n)
    start = time.time()
    try:
        processed, second = process_image(n)
        if type(processed) is int:
            continue
        if processed.shape[1] > 0 and processed.shape[0] > 0:
            # cv2.imwrite(f'{count}.png', processed)
            header = second[:75, :]
            help_header = processed[:50, :]
            # print('prLen', processed.shape[1])
            help_header, _s = find_contour(help_header, help_header, accuracy=1, mode=2)
            # processed_header, s = find_contour(header, header, accuracy=1, mode=2)
            # cv2.imwrite(f'Headers/header{count}.png', help_header)
            # print(processed_header.shape)
            #       cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #      cv2.imshow('image', header)
            #     cv2.waitKey()
            # croped_header = header
            # croped_header, s = find_contour(header, header, accuracy=1)
            # res = match_header(header)
            # print(res)
            # print(ht.convert(f'Headers/header{count}.png'))
            count += 1
            _id = hm.classify(header, help_header, processed.shape[1])
            # _id = -1
            # width = processed_header.shape[1]
            # if width in range(1750, 1800):
            #    _id = 0
            # elif width in range(1600, 1705):
            #    _id = 1
            # elif width in range(625, 700):
            #    _id = 2
            if _id >= 0:
                print(doc_names[_id], vd.validate(processed, _id))
            else:
                pass
                # print(n, 'no')
            times.append(time.time() - start)
    except Exception as e:
        raise e
        # print(e)
        pass
        # print(n, 'no')
if len(times) > 0:
    print(np.mean(times))
