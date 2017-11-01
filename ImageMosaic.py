from ImageProcessor import ImageProcessor
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("img1", default="")
parser.add_argument("img2", default="")
args = parser.parse_args()

img1 = cv2.imread(args.img1, 0) # read in images in greyscale
img2 = cv2.imread(args.img2, 0)

IP = ImageProcessor()
IP.Mosaic(img2, img1)
