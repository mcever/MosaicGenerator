import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('/Users/christianlee/Projects/gonzalez1.png', 0)
img2 = cv2.imread('/Users/christianlee/Projects/gonzalez2.png', 0)


sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []

for m, n in matches:
    if m.distance < 0.50*n.distance:
        good.append(m)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

p1 = []
p2 = []
for m in good:
    p1.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
    p2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])

matrixIndex = 0

A = []

for i in range(0, len(p1)):
    x = p1[i][0]
    y = p1[i][1]

    u = p2[i][0]
    v = p2[i][1]

    A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
    A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

A = np.array(A)
U, S, V = np.linalg.svd(A)
L = V[-1,:] / V[-1,-1]
matrix = L.reshape(3, 3)

print(matrix)

x1 = np.asarray(p1)
m1, s1 = np.mean(x1, 0), np.std(x1)
Tr = np.array([[s1, 0, m1[0]], [0, s1, m1[1]], [0, 0, 1]])
Tr = np.linalg.inv(Tr)
x1 = np.dot(Tr, np.concatenate((x1.T, np.ones((1, x1.shape[0])))))
x1 = x1[0:2, :].T

x2 = np.asarray(p2)
m2, s2 = np.mean(x2, 0), np.std(x2)
Tr = np.array([[s2, 0, m2[0]], [0, s2, m2[1]], [0, 0, 1]])
Tr = np.linalg.inv(Tr)
x2 = np.dot(Tr, np.concatenate((x1.T, np.ones((1, x1.shape[0])))))
x2 = x2[0:2, :].T

A = []
for i in range(0, len(p1)):
    x = x1[i][0]
    y = x1[i][1]

    u = x2[i][0]
    v = x2[i][1]

    A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
    A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

A = np.array(A)
U, S, V = np.linalg.svd(A)
L = V[-1,:] / V[-1,-1]
matrix2 = L.reshape(3, 3)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print(M)
print(matrix2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
img_1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
img4 = cv2.warpPerspective(img_1, matrix, (1000, 1000))

plt.imshow(img4),plt.show()
