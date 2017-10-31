import imutils
import cv2
from pprint import pprint
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pdb

class ImageProcessor:
    def __init__(self):
        pass

    def ratioTest(matches):
        good = []
        for m, n in matches:
            if m.distance < 0.70*n.distance:
                good.append(m)
        return good

    def DLT(matches):
        # break the matches into two lists with corresponding pairs for easier DLT computation
        p1 = [] # p1[0] matches p2[0]
        p2 = []
        for m in good:
            p1.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
            p2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])

        # estimate homography with standard DLT
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

    def RANSAC(matches):
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0);

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        plt.imshow(img3, 'gray'),plt.show()

    def nDLT(matches):
        pass

    def Warp(img1, img2, useDLT=False, usenDLT=False, useRANSAC=True, showMatches=False):
        # resize to make faster
        img1 = imutils.resize(img1, width=400)
        img2 = imutils.resize(img2, width=400)

        # detect keypoints by SIFT
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # select best keypoints with brute force / establish correspondence for DLT
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        if showMatches:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

        if useDLT:
            DLT_H = DLT(matches)
            img3 = cv2.warpPerspective(img1, DLT_H, (400,400))
            plt.imshow(img3)
            plt.show() #title="DLT"
        if usenDLT:
            nDLT_H = nDLT(matches)
            img3 = cv2.warpPerspective(img1, nDLT_H, (400,400))
            plt.imshow(img3)
            plt.show() #title="nDLT"
        if useRANSAC:
            R_H = RANSAC(matches)
            img3 = cv2.warpPerspective(img1, R_H, (400,400))
            plt.imshow(img3)
            plt.show() #title="RANSAC"  




    def createMosaic(self, img1, img2):
        # resize to make faster
        img1 = imutils.resize(img1, width=400)
        img2 = imutils.resize(img2, width=400)

        # detect keypoints by SIFT (SURF later?)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # select best keypoints with brute force / establish correspondence for DLT
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # use ratio test to filter some matches before DLT
        good = ratioTest(matches)

        # estimate homography with normalized DLT

        # estimate homography and select best keypoints by RANSAC

        # warp images and create mosaic

        # -- with standard DLT

        # -- with normalized DLT

        # -- with normalized DLT using keypoints from RANSAC

        #return img

    def FLANN_stuff():
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=100)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        #matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        good = []
        for i,(m,n) in enumerate(matches):
            good.append(m)
            if m.distance < 0.65*n.distance:
                #matchesMask[i]=[1,0]
                good.append(m)

    def brute_force_stuff():
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        #matches = bf.knnMatch(des1,des2, k=2) not working idk
        #matches = bf.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
        #plt.imshow(img3)
        #plt.show()
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)
        plt.imshow(img3),plt.show()
        raw_input("")
