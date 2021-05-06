# Test SIFT with mask specified in sift.detectAndCompute(img_query, img1_mask)
# The effects of ratio test is also explored

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


fname = os.getcwd() + '/img_gates/'
MIN_MATCH_COUNT = 10

img_query = cv2.imread(fname + 'img_331.png')          # queryImage
img_query = cv2.cvtColor(img_query, cv2.COLOR_RGB2GRAY)
img_train = cv2.imread(fname + 'img_262.png') # trainImage
img_train = cv2.cvtColor(img_train, cv2.COLOR_RGB2GRAY)
img1_mask = cv2.imread(fname + 'mask_331.png')
img1_mask = cv2.cvtColor(img1_mask, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_query, img1_mask)
kp2, des2 = sift.detectAndCompute(img_train, None)
img1_kp = cv2.drawKeypoints(img_query, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv2.drawKeypoints(img_train, kp2, None, color=(0,255,0), flags=0)
plt.imshow(img1_kp),plt.show()
plt.imshow(img2_kp),plt.show()
# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

if len(matches)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h, w, d = img_query.shape
    pts = np.float32([[0, 0], [0, h-1],[w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts,M)
    img_train = cv2.polylines(img_train,[np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor=(0,255,0), # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask, # draw only inliers
                   flags=2)
img3 = cv2.drawMatches(img_query, kp1, img_train, kp2, matches, None, **draw_params)
plt.imshow(img3, 'gray'),plt.show()
