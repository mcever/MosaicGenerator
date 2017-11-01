from ImageProcessor import ImageProcessor
import argparse
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#parser.add_argument("img1", default="")
parser.add_argument("imgs", nargs="+")
args = parser.parse_args()

if len(args.imgs) < 2:
    print("Need at least two images")
    exit()

# print all the arguments in case user input with wildcards
for arg in args.imgs:
    print(arg)

first = True
i = 0
for arg in args.imgs:
    if first:
        result = cv2.imread(arg)
        first = False
        continue
    else:
        IP = ImageProcessor()
        i += 1
        result = IP.Mosaic(result, cv2.imread(arg))
        cv2.imwrite(str(i) + ".jpg", result)
        plt.imshow(result)
        plt.show()
