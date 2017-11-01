import cv2
import numpy as np
import matplotlib.pyplot as plt

def Stitch(img1, img2):
    # stitch im1 to img2
    img1 = img1
    img2 = img2
    img_2 = img2
    T = np.float32([[1, 0, 1000], [0, 1, 1000]])

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.warpAffine(gray2, T, (3000, 3000))
    img2 = cv2.warpAffine(img2, T, (3000, 3000))

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.50*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(img1, M, (3000, 3000))

    for i in range(1000, 1000+img_2.shape[0]):
        for j in range(1000, 1000+img_2.shape[1]):
            result[i, j] = img_2[i-1000, j-1000]

    return result

def main():
    img1 = cv2.imread('kohn_2.tif')
    img2 = cv2.imread('kohn_3.tif')
    result1 = Stitch(img1, img2)
    img1 = result1
    img2 = cv2.imread('kohn_4.tif')
    result2 = Stitch(img1, img2)
    img1 = result2
    img2 = cv2.imread('kohn_5.tif')
    result3 = Stitch(img1, img2)
    result = np.concatenate((result1, result2, result3), axis = 1)
    plt.imshow(result),plt.show()

if __name__ == "__main__": main()
