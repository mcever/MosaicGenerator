from Mosaic import Mosaic
import imutils
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("img1", default="")
parser.add_argument("img2", default="")
args = parser.parse_args()

img1 = cv2.imread(args.img1, 0) # read in images in greyscale
img2 = cv2.imread(args.img2, 0)

cv2.imshow("Img1", img1)
cv2.imshow("Img2", img2)
cv2.waitKey(0)

M = Mosaic()
M.create(img1, img2)
