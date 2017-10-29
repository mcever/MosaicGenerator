import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("img1", default="")
parser.add_argument("img2", default="")
args = parser.parse_args()

img1 = cv2.imread(args.img1, 0)
img2 = cv2.imread(args.img2, 0)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []

for m, n in matches:
    if m.distance < 0.70*n.distance:
        good.append(m)

very_good = []

for m in good:
    if ((kp1[m.queryIdx].size/kp2[m.trainIdx].size > 0.7) and
    (kp1[m.queryIdx].size/kp2[m.trainIdx].size < 1.3)):
        very_good.append(m)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

p1 = []
p2 = []
for m in very_good:
    p1.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
    p2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])

matrixIndex = 0
A = np.zeros((8, 9))

for i in range(0, 4):
    x = p1[i][0]
    y = p1[i][1]

    u = p2[i][0]
    v = p2[i][1]

    A[matrixIndex] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
    A[matrixIndex + 1] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]

    matrixIndex = matrixIndex + 2

    U, s, V = np.linalg.svd(A, full_matrices=True)
    matrix = V[:, 8].reshape(3, 3).transpose()

for i in range(0, 3):
    for j in range(0, 3):
        matrix[i][j] /= matrix[2][2];

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0);

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)

plt.imshow(img3),plt.show()
