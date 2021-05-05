import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


fname = os.getcwd() + '/img_gates/'

img_query = cv2.imread(fname + 'img_331.png')          # queryImage
# plt.imshow(img_query[...,::-1]),plt.show()
img_query = cv2.cvtColor(img_query, cv2.COLOR_RGB2GRAY)
img_query_corp = img_query[88:340, 60:315]
img_train = cv2.imread(fname + 'img_255.png') # trainImage
img_train = cv2.cvtColor(img_train, cv2.COLOR_RGB2GRAY)
# img1_mask = cv2.imread(fname + 'mask_331.png')
# img1_mask = cv2.cvtColor(img1_mask, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_query_corp, None)
kp2, des2 = sift.detectAndCompute(img_train, None)
img1_kp = cv2.drawKeypoints(img_query_corp, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv2.drawKeypoints(img_train, kp2, None, color=(0,255,0), flags=0)
plt.imshow(img1_kp),plt.show()
plt.imshow(img2_kp),plt.show()
# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# Draw first 10 matches.
img4 = cv2.drawMatchesKnn(img_query_corp, kp1, img_train, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img4)
plt.show()
