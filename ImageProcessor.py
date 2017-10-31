import imutils
import cv2
from pprint import pprint
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import math

class ImageProcessor:
    def __init__(self):
        pass

    def ratioTest(self, matches):
        good = []
        for m, n in matches:
            if m.distance < 0.50*n.distance:
                good.append(m)
        return good

    def DLT(self, matches, kp1, kp2):
        # break matches to two lists
        p1 = []
        p2 = []
        for m in matches:
            p1.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
            p2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])

        matrixIndex = 0
        A = [] # list of 2x9 matrices for DLT

        for i in range(0, len(p1)): # for each match
            x = p1[i][0] # gather each points' coordinates
            y = p1[i][1]

            u = p2[i][0]
            v = p2[i][1]

            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u]) # create 2x9 matrix
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

        A = np.array(A)
        U, S, V = np.linalg.svd(A) # use SVD to acquire Unitary matrices
        L = V[-1,:] / V[-1,-1] # get homography from unitar matrices
        matrix = L.reshape(3, 3)
        return matrix

    def RANSAC(self, matches, kp1, kp2):
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M

    def nDLT(self, matches, kp1, kp2):
        p1 = []
        p2 = []
        for m in matches:
            p1.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])
            p2.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])

        x1 = np.asarray(p1)
        m1, s1 = np.mean(x1, 0), math.sqrt(2) # set a normalized mean to 0 and std_dev to sqrt(2)
        Tr = np.array([[s1, 0, s1*m1[0]], [0, s1, s1*m1[1]], [0, 0, 1]])
        Tr = np.linalg.inv(Tr)
        x1 = np.dot(Tr, np.concatenate((x1.T, np.ones((1, x1.shape[0])))))
        x1 = x1[0:2, :].T

        x2 = np.asarray(p2)
        m2, s2 = np.mean(x2, 0), math.sqrt(2)
        Tr = np.array([[s2, 0, s2*m2[0]], [0, s2, s2*m2[1]], [0, 0, 1]])
        Tr = np.linalg.inv(Tr)
        x2 = np.dot(Tr, np.concatenate((x2.T, np.ones((1, x2.shape[0])))))
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

        return matrix2

    def Warp(self, img1, img2, useDLT=False, usenDLT=False, useRANSAC=True, showMatches=False):
        # resize to make faster
        #img1 = imutils.resize(img1, width=400)
        #img2 = imutils.resize(img2, width=400)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        # detect keypoints by SIFT
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # select best keypoints with brute force / establish correspondence for DLT
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        matches = self.ratioTest(matches)

        if showMatches:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

        if useDLT:
            DLT_H = self.DLT(matches, kp1, kp2)
            img3 = cv2.warpPerspective(img1, DLT_H, (1000,1000))
            plt.title("DLT")
            plt.imshow(img3)
            plt.show()
        if usenDLT:
            nDLT_H = self.nDLT(matches, kp1, kp2)
            img3 = cv2.warpPerspective(img1, nDLT_H, (1000,1000))
            plt.title("nDLT")
            plt.imshow(img3)
            plt.show() #title="nDLT"
        if useRANSAC:
            R_H = self.RANSAC(matches, kp1, kp2)
            img3 = cv2.warpPerspective(img1, R_H, (1000,1000))
            plt.title("RANSAC")
            plt.imshow(img3)
            plt.show() #title="RANSAC"

    def Mosaic(self, img1, img2):
        # resize to make faster
        #img1 = imutils.resize(img1, width=400)
        #img2 = imutils.resize(img2, width=400)

        # detect keypoints by SIFT
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # select best keypoints with brute force
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        matches = self.ratioTest(matches)

        # use RANSAC to compute homograpy
        H = self.RANSAC(matches, kp1, kp2)
        img3 = cv2.warpPerspective(img1, H, (600,600))
        result = self.mix_and_match(img1, img3)

        # stitch the images together and display

        cv2.imshow("title", result)
        cv2.waitKey(0)


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

    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        #print leftImage[-1,-1]

        #t = time.time()
        black_l = np.where(leftImage == np.array([0,0,0]))
        black_wi = np.where(warpedImage == np.array([0,0,0]))
        #print time.time() - t
        #print black_l[-1]

        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if(np.array_equal(leftImage[j,i],np.array([0,0,0])) and  np.array_equal(warpedImage[j,i],np.array([0,0,0]))):
                        # print "BLACK"
                        # instead of just putting it with black,
                        # take average of all nearby values and avg it.
                        warpedImage[j,i] = [0, 0, 0]
                    else:
                        if(np.array_equal(warpedImage[j,i],[0,0,0])):
                            # print "PIXEL"
                            warpedImage[j,i] = leftImage[j,i]
                        else:
                            if not np.array_equal(leftImage[j,i], [0,0,0]):
                                bw, gw, rw = warpedImage[j,i]
                                bl,gl,rl = leftImage[j,i]
                                # b = (bl+bw)/2
                                # g = (gl+gw)/2
                                # r = (rl+rw)/2
                                warpedImage[j, i] = [bl,gl,rl]
                except:
                    pass
        cv2.imshow("waRPED mix", warpedImage)
        cv2.waitKey()
        return warpedImage
