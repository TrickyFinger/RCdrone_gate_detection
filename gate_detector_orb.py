import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


fname = os.getcwd() + '/WashingtonOBRace/WashingtonOBRace/'

img1 = cv2.imread(fname + 'img_331.png')          # queryImage
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(fname + 'img_262.png') # trainImage
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#img1_mask = cv2.imread(fname + 'mask_413.png')
#img1_mask = cv2.cvtColor(img1_mask, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
plt.imshow(img1_kp),plt.show()
plt.imshow(img2_kp),plt.show()
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img4 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img4),plt.show()