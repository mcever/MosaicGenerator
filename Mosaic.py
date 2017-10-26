import imutils
import cv2

class Mosaic:
  def __init__(self):
      pass

  def create(self, img1, img2):
      # resize to make faster
      img1 = imutils.resize(img1, width=400)
      img2 = imutils.resize(img2, width=400)

      # detect keypoints by SIFT

      # detect keypoints by SURF (maybe)

      # select best keypoints with FLANN (or BruteForce) / establish correspondence for DLT

      # estimate homography with standard DLT

      # estimate homography with normalized DLT

      # estimate homography and select best keypoints by RANSAC

      # warp images and create mosaic

      # -- with standard DLT

      # -- with normalized DLT

      # -- with normalized DLT using keypoints from RANSAC






      #return img
