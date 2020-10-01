import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import sys


class HeaderToText:
    def __init__(self):
        self.dictionary = ['0', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О',
                           'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю',
                           'Я', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.refLetters = []
        self.files = [f for f in listdir('Dictionary/') if isfile(join('Dictionary/', f))]
        for filename in self.files:
            im = cv2.cvtColor(cv2.imread(f'Dictionary/{filename}'), cv2.COLOR_BGR2GRAY)
            self.refLetters.append([im, filename.replace('.png', '')])

    def extract_letters(self, img, out_size=28):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
        img_erode = cv2.erode(thresh, np.ones((1, 1), np.uint8), iterations=1)
        # Get contours
        contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        output = img.copy()
        letters = []
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
            # hierarchy[i][0]: the index of the next contour of the same level
            # hierarchy[i][1]: the index of the previous contour of the same level
            # hierarchy[i][2]: the index of the first child
            # hierarchy[i][3]: the index of the parent
            if hierarchy[0][idx][3] == 0:
                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
                letter_crop = gray[y:y + h, x:x + w]
                # print(letter_crop.shape)
                # Resize letter canvas to square
                size_max = max(w, h)
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
                if w > h:
                    # Enlarge image top-bottom
                    # ------
                    # ======
                    # ------
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h:
                    # Enlarge image left-right
                    # --||--
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop
                # Resize letter to 28x28 and add letter and its X-coordinate
                letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))
        # Sort array in place by X-coordinate
        letters.sort(key=lambda x: x[0], reverse=False)
        return letters

    def convert(self, name):
        # image = cv2.imread('header8.png')
        image = cv2.imread(name)
        letters = self.extract_letters(image)
        # letter = cv2.imread("testLetter.png")
        # letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
        count = 0
        word = ''
        for l in letters:
            #cv2.imshow("letter", l[0])
            #cv2.waitKey(0)
            # cv2.imwrite(f"letters/letter{count}.png", l[2])
            minimal = sys.maxsize
            best = ''
            for ref in self.refLetters:
                bws = cv2.bitwise_and(255 - l[2], 255 - ref[0])
                bor = cv2.bitwise_or(255 - l[2], 255 - ref[0])
                # cv2.imshow("1", 255 - l[2])
                # cv2.imshow("bws", bws)
                # cv2.imshow("bor", bor)
                # cv2.waitKey(0)
                score = abs(bor.mean() - bws.mean())
                #   print(score)
                # mean = bws.mean()
                if score < minimal:
                    minimal = score
                    best = ref[1]
            best = self.dictionary[int(best)]
            # print(best)
            # break
            if best == 'Ы' and word[-1] == 'Ь':
                word = word[:-1] + best
            else:
                word += best
            # cv2.imshow("dif", bws)
            count += 1
        return word
