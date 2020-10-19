from matcher import HeaderMatcher
import cv2
import numpy as np
import time
import argparse
from validator import Validator
from os import listdir
from os.path import isfile, join


def find_contour(im, sec, accuracy=10):
    a = im
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
    for i in range(int(rows / accuracy)):
        mean = a[i * accuracy:i * accuracy + accuracy, int(cols / 10):int(9 * cols / 10)].mean()
        # prev_mean = a[i*accuracy-accuracy:i*accuracy, int(cols/10):int(9*cols/10)].mean()

        if mean < MAX:  # and prev_mean > 254:
            topI = i
            break
    for i in range(int(rows / accuracy), 0, -1):
        mean = a[(i - 1) * accuracy:i * accuracy, int(cols / 10):int(9 * cols / 10)].mean()
        # prev_mean = a[i*accuracy:i*accuracy+accuracy, int(cols/10):int(9*cols/10)].mean()

        if mean < MAX:  # and prev_mean > 254:
            botI = i
            break

    for j in range(int(cols / accuracy)):
        mean = a[int(rows / 10):int(9 * rows / 10), j * accuracy:j * accuracy + accuracy].mean()
        # prev_mean = a[int(rows/10):int(9*rows/10), j*accuracy-accuracy:j*accuracy].mean()
        if mean < MAX:  # and prev_mean > 254:
            leftJ = j
            break
    for j in range(int(cols / accuracy), 0, -1):
        mean = a[int(rows / 10):int(9 * rows / 10), (j - 1) * accuracy:j * accuracy].mean()
        # prev_mean = a[int(rows/10):int(9*rows/10), j*accuracy:j*accuracy+accuracy].mean()
        if mean < MAX:  # and prev_mean > 254:
            rightJ = j
            break

    #    print(topI, botI, leftJ, rightJ)
    half = int(accuracy / 2)
    a = a[topI * accuracy - half:botI * accuracy + half, leftJ * accuracy - half:rightJ * accuracy + half]
    sec = sec[topI * accuracy - half:botI * accuracy + half, leftJ * accuracy - half:rightJ * accuracy + half]
    return a, sec


def process_image(name):
    img = cv2.imread(name)
    # mean = img.mean()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if med < 250:
    # print(img.mean())
    secondary = img.copy()
    ret, img = cv2.threshold(img, img.mean() - 20, 255, cv2.THRESH_BINARY)
    orig = img.copy()
    # output = img.copy()
    img = 255 - img

    kernel = np.ones((10, 10), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=5)
    # dilation = 255 - dilation
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    angles = []
    # print(len(contours))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # box = np.int0(box)
        angle = rect[2]
        # if angle > -89 and angle < 89:
        if angle < -45:
            angle = (90 + angle)
        angles.append(angle)
    #  output = cv2.drawContours(output,[box],0,(0,0,255),2)
    # cv2.imwrite('dilated.png', dilation)
    # cv2.imwrite('out.png', output)
    angles = np.array(angles)
    if angles.mean() < 0:
        angle = angles.min()
    else:
        angle = angles.max()
    # print(angle, angles)
    rows, cols = orig.shape[0], orig.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(orig, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    secondary = cv2.warpAffine(secondary, M, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # cv2.imwrite('rotated.png', dst)
    dst, secondary = find_contour(dst, secondary)
    # cv2.imwrite('croped.png', dst)
    return dst, secondary


def match_header(img):
    ref = cv2.imread('cropedRef.png')
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not ref.shape == img.shape:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    diff_img = cv2.absdiff(img, ref)
    tile_diff = int(np.sum(diff_img) / 255)
    # print('header diff:', tile_diff)
    return tile_diff


# names = ['cat.jpg', 'im54.png', 'yellow.jpg', '1Doc.jpg', 'mirror.png', 'im12.png', 'im13.png',
#          'im1.png', 'test.png', 'real-0.jpg', 'real-1.jpg', 'wrong.png', 'wrong1.png', 'wrong3.png',
#          'real-0.jpg', 'real-1.jpg', 'real-3.jpg']
# names = ['cat.jpg', 'im54.png', 'wrong1.png', 'real-6.jpg', 'real-8.jpg', 'real-9.jpg']
# MAX_HEADER_DIFF = 10000
count = 0

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="Path to image")
ap.add_argument("-m", "--mode", required=False,
                help="Mode:\n0 -- all, used by default; 1 -- 14; 2 -- 14-18; 3 -- soglasie na obrabotku")
args = vars(ap.parse_args())

vd = Validator()
times = []
hm = HeaderMatcher()
doc_names = ['до 14', 'до 18', 'согласие']

dir_path = args['path']
# print(dir_path)
names = [dir_path + f for f in listdir(dir_path) if isfile(join(dir_path, f))]
# print(names)
for n in names:
    start = time.time()
    try:
        processed, second = process_image(n)
        if processed.shape[1] > 0 and processed.shape[0] > 0:
            # cv2.imwrite(f'{count}.png', processed)
            header = second[:75, :]

            #       cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #      cv2.imshow('image', header)
            #     cv2.waitKey()
            # croped_header = header
            # croped_header, s = find_contour(header, header, accuracy=1)
            # res = match_header(header)
            # print(res)
            try:
                pass
                # print(count)
                # cv2.imwrite(f'Headers/header{count}.png', header)
            except Exception as e:
                pass
            # print(ht.convert(f'Headers/header{count}.png'))
            count += 1
            _id = hm.classify(header)
            times.append(time.time() - start)
            if _id >= 0:
                print(n, doc_names[_id])
                print(vd.validate(processed, _id))
            else:
                pass
                # print(n, 'no')
    except Exception as e:
        pass
        # print(n, 'no')
print(np.mean(times))
