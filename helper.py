from os import listdir
from os.path import isfile, join
import cv2

onlyfiles = [f for f in listdir('Dictionary/') if isfile(join('Dictionary/', f))]
for i in range(len(onlyfiles)):
    cv2.imwrite(f'Dictionary/{i+1}.png', cv2.imread(f'Dictionary/{onlyfiles[i]}'))
print(onlyfiles)
