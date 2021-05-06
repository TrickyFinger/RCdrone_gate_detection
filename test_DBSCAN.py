# Use DBSCAN to find clusters of points

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from sklearn.cluster import DBSCAN
import collections
import time


def white_area_ratio(mask, cx, cy, R):
    roi = mask[cy-R:cy+R, cx-R:cx+R]
    rows, cols = roi.shape
    board = np.zeros((rows, cols), dtype=np.uint8)
    mask_circle = cv2.circle(board, (int(cols/2),int(rows/2)), R, 255, -1)
    img = cv2.bitwise_and(roi, roi, mask=mask_circle)
    area = cv2.countNonZero(img)
    return area/(np.pi*R**2)

k_knnmatch = 3
sample_circle_step = 15
sample_circle_radius = 30
sample_reject_ratio = 3
dir = os.path.join(os.getcwd(), 'img_gates')
target = os.path.join(dir, 'img_60.png')

img_query = cv2.imread('pattern.png')          # queryImage
img_query = cv2.cvtColor(img_query, cv2.COLOR_RGB2GRAY)

img_train = cv2.imread(target) # trainImage
img_train = cv2.cvtColor(img_train, cv2.COLOR_RGB2GRAY)
plt.imshow(img_train, cmap='gray'), plt.show()
mask = cv2.inRange(img_train, 210, 255)
kernel = np.ones((8, 8), np.uint8)
plt.imshow(mask, cmap='gray'), plt.show()
dilation = cv2.dilate(mask, kernel, iterations=1)
plt.imshow(dilation, cmap='gray'), plt.show()
[h, w] = img_train.shape

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_query, None)
kp2, des2 = sift.detectAndCompute(img_train, mask=dilation)
img1_kp = cv2.drawKeypoints(img_query, kp1, None, color=(0,255,0), flags=0)
img2_kp = cv2.drawKeypoints(img_train, kp2, None, color=(0,255,0), flags=0)

# plt.imshow(img2_kp),plt.show()
# create BFMatcher object
bf = cv2.BFMatcher()
# Match descriptors.
matches = bf.knnMatch(des1, des2, k=k_knnmatch)
# Draw first 10 matches.
img_matches = cv2.drawMatchesKnn(img_query, kp1, img_train, kp2, matches, None, flags=2)

pt_match = []
for m in matches:
    for x in m:
        pt_match.append([int(round(kp2[x.trainIdx].pt[0])), int(round(kp2[x.trainIdx].pt[1]))])

pt_match = np.asarray(pt_match)
start = time.time()
cluster_labels = DBSCAN(eps=20, min_samples=5).fit_predict(pt_match)
labels = np.sort(cluster_labels)
# print(labels)
i_2_del = []
for i, num in enumerate(labels):
    if num < 0:
        i_2_del.append(i)
labels = np.delete(labels, i_2_del)
counter=collections.Counter(labels)
# print(labels)
keys = counter.keys()

counter = np.asarray([list(counter.keys()), list(counter.values())])
# print(counter)
# print(counter.shape)
c = counter[:, counter[1].argsort()]
# print(c)
max1_label = c[0, -1]
max2_label = c[0, -2]
max3_label = c[0, -3]
max4_label = c[0, -4]

img_tmp = cv2.imread(target)
img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)

img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2BGR)
for pt, label in zip(pt_match, cluster_labels):
    if label == max1_label:
        img_tmp = cv2.circle(img_tmp, tuple(pt), 3, (255, 0, 0), -1)
    elif label == max2_label:
        img_tmp = cv2.circle(img_tmp, tuple(pt), 3, (255, 0, 0), -1)
    elif label == max3_label:
        img_tmp = cv2.circle(img_tmp, tuple(pt), 3, (255, 0, 0), -1)
    elif label == max4_label:
        img_tmp = cv2.circle(img_tmp, tuple(pt), 3, (255, 0, 0), -1)
    else:
        pass
cv2.imshow('1', img_tmp)
cv2.waitKey(0)

cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []
for pt, label in zip(pt_match, cluster_labels):
    if label == max1_label:
        cluster1.append(pt)
    elif label == max2_label:
        cluster2.append(pt)
    elif label == max3_label:
        cluster3.append(pt)
    elif label == max4_label:
        cluster4.append(pt)
    else:
        pass
cluster1 = np.asarray(cluster1)
avg_pt1 = [int(np.sum(cluster1[:,0])/cluster1.shape[0]), int(np.sum(cluster1[:,1])/cluster1.shape[0])]
cluster2 = np.asarray(cluster2)
avg_pt2 = [int(np.sum(cluster2[:,0])/cluster2.shape[0]), int(np.sum(cluster2[:,1])/cluster2.shape[0])]
cluster3 = np.asarray(cluster3)
avg_pt3 = [int(np.sum(cluster3[:,0])/cluster3.shape[0]), int(np.sum(cluster3[:,1])/cluster3.shape[0])]
cluster4 = np.asarray(cluster4)
avg_pt4 = [int(np.sum(cluster4[:,0])/cluster4.shape[0]), int(np.sum(cluster4[:,1])/cluster4.shape[0])]
print(time.time()-start)
img_tmp = cv2.circle(img_tmp, tuple(avg_pt1), 8, (0, 255, 0), -1)
img_tmp = cv2.circle(img_tmp, tuple(avg_pt2), 8, (0, 255, 0), -1)
img_tmp = cv2.circle(img_tmp, tuple(avg_pt3), 8, (0, 255, 0), -1)
img_tmp = cv2.circle(img_tmp, tuple(avg_pt4), 8, (0, 255, 0), -1)

gate_center = [int((avg_pt1[0]+avg_pt2[0]+avg_pt3[0]+avg_pt4[0]) / 4), int((avg_pt1[1]+avg_pt2[1]+avg_pt3[1]+avg_pt4[1]) / 4)]
img_tmp = cv2.circle(img_tmp, tuple(gate_center), 15, (16, 155, 5), 3)
cv2.imshow('1', img_tmp)
cv2.waitKey(0)

'''
step = sample_circle_step
R = sample_circle_radius
sample_circle = []
for x in range(R, w-R, step):
    for y in range(R, h-R, step):
        war = white_area_ratio(mask, x, y, R)
        if war > 0.9 or war < 0.1:
            pass
        else:
            sample = np.array([x, y, 0])
            for pt in pt_match:
                dist = np.sqrt((x-pt[0])**2+(y-pt[1])**2)
                if dist <= R and (pt[1]-y) >= 0 and (pt[0]-x) >= 0:
                    sample[2] += 1
            if sample[2] != 0:
                sample_circle.append(sample)

sample_circle = np.array(sample_circle)
sample_circle = sample_circle[sample_circle[:, 2].argsort()[::-1]]
max_nu = sample_circle[0, 2]
[rows, cols] = sample_circle.shape
for i in range(0, rows, 1):
    if sample_circle[i, 2] < max_nu/sample_reject_ratio:
        sample_circle = np.delete(sample_circle, np.s_[i:rows], 0)
        break

[rows, cols] = sample_circle.shape

img_tmp = cv2.imread(target)

# Display sample circles
for i in range(0, rows):
    x = int(round(sample_circle[i, 0]))
    y = int(round(sample_circle[i, 1]))
    img_tmp = cv2.circle(img_tmp, (x, y) , R, (0, 0, 255), 1)

sample_pt = np.zeros((rows, 3))
for i in range(0, rows):
    cx = sample_circle[i, 0]
    cy = sample_circle[i, 1]
    x = cx + R*0.5*0.707
    y = cy + R*0.5*0.707
    img_tmp = cv2.circle(img_tmp, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    angle = np.arctan2(y, x)
    sample_pt[i, :] = [x, y, angle]
mid_angle = 0.5 * (np.max(sample_pt[:, 2]) + np.min(sample_pt[:, 2]))
mid_k = np.tan(mid_angle)

mid_x = np.dot((np.dot(mid_k, sample_pt[:, 1])+sample_pt[:, 0]), 1/(mid_k**2+1))
mid_y = np.dot(mid_k, mid_x)

gate_center = [int(round(0.5 * (np.max(mid_x)+np.min(mid_x)))), int(round(0.5 * (np.max(mid_y)+np.min(mid_y))))]
print(gate_center)
img_tmp = cv2.circle(img_tmp, tuple(gate_center), 5, (0, 0, 255), -1)
cv2.imshow('1', img_tmp)
cv2.waitKey(0)
'''


