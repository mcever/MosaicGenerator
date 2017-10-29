import imutils
import cv2
from pprint import pprint
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pdb

class Mosaic:
    def __init__(self):
        pass

    def create(self, img1, img2):
        # resize to make faster
        img1 = imutils.resize(img1, width=400)
        img2 = imutils.resize(img2, width=400)

        # detect keypoints by SIFT (SURF later?)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # select best keypoints with FLANN / establish correspondence for DLT
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

        # estimate homography with standard DLT

        # estimate homography with normalized DLT

        # estimate homography and select best keypoints by RANSAC
        if len(good)>10: # ensure enough matches
            #pdb.set_trace()
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            #pdb.set_trace()
            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        plt.imshow(img3, 'gray'),plt.show()


        # warp images and create mosaic

        # -- with standard DLT

        # -- with normalized DLT

        # -- with normalized DLT using keypoints from RANSAC

        #return img

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
