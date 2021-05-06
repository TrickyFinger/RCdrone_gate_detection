# This file contains functions to be executed in the main.py
# Function 1: gate_detector_sampling.
#   This is the function developed by myself which uses circles to sample all the interested area
#   to perform data clustering task. It is slow and lacking accuracy. So I abandoned it after discovering DBSCAN.
# Function 2: gate_detector_clustering.
#   This function uses DBSCAN to perform data clustering task. It returns the coordinates of four corners.

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import collections
from sklearn.cluster import DBSCAN

# Compute the ratio in a binary image: (white area / total area). Total area is the size of the circle
def white_area_ratio(mask, cx, cy, R):
    roi = mask[cy-R:cy+R, cx-R:cx+R]
    rows, cols = roi.shape
    board = np.zeros((rows, cols), dtype=np.uint8)
    mask_circle = cv2.circle(board, (int(cols/2),int(rows/2)), R, 255, -1)
    img = cv2.bitwise_and(roi, roi, mask=mask_circle)
    area = cv2.countNonZero(img)
    return area/(np.pi*R**2)


def gate_detector_sampling(img_path, k_knnmatch=4, sample_circle_step=20, sample_circle_radius=40, sample_reject_ratio=4):
    img_query = cv2.imread('pattern.png')  # queryImage
    img_query = cv2.cvtColor(img_query, cv2.COLOR_RGB2GRAY)

    img_train = cv2.imread(img_path)  # trainImage
    img_train = cv2.cvtColor(img_train, cv2.COLOR_RGB2GRAY)
    mask = cv2.inRange(img_train, 210, 255)
    [h, w] = img_train.shape

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_query, None)
    kp2, des2 = sift.detectAndCompute(img_train, None)

    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=k_knnmatch)

    # Save the matched keypoints into a list
    pt_match = []
    for m in matches:
        for x in m:
            pt_match.append([int(round(kp2[x.trainIdx].pt[0])), int(round(kp2[x.trainIdx].pt[1]))])

    # Sampling the image with a circle
    step = sample_circle_step
    R = sample_circle_radius
    sample_circle = []
    for x in range(R, w - R, step):
        for y in range(R, h - R, step):
            war = white_area_ratio(mask, x, y, R)
            # refer to the binary image, reject all samples with too much or too less white area.
            if war > 0.9 or war < 0.1:
                pass
            else:
                sample = np.array([x, y, 0])
                for pt in pt_match:
                    dist = np.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
                    # Only consider points lie in the bottom-right part of the circle
                    if dist <= R and (pt[1] - y) >= 0 and (pt[0] - x) >= 0:
                        sample[2] += 1
                if sample[2] != 0:
                    sample_circle.append(sample)

    sample_circle = np.array(sample_circle)
    # max_nu: The maximum number of keypoints inside a sample circle
    sample_circle = sample_circle[sample_circle[:, 2].argsort()[::-1]]
    max_nu = sample_circle[0, 2]
    [rows, cols] = sample_circle.shape
    for i in range(0, rows):
        # Reject those sample circles which contain keypoints less than (max_nu / sample_reject_ratio)
        if sample_circle[i, 2] < max_nu / sample_reject_ratio:
            sample_circle = np.delete(sample_circle, np.s_[i:rows], 0)
            break

    [rows, cols] = sample_circle.shape

    # Display sample circles
    # for i in range(0, rows):
    #     x = int(round(sample_circle[i, 0]))
    #     y = int(round(sample_circle[i, 1]))

    # Please refer to the report for the explanation of this part
    sample_pt = np.zeros((rows, 3))
    for i in range(0, rows):
        cx = sample_circle[i, 0]
        cy = sample_circle[i, 1]
        x = cx + R * 0.5 * 0.707
        y = cy + R * 0.5 * 0.707
        angle = np.arctan2(y, x)
        sample_pt[i, :] = [x, y, angle]
    mid_angle = 0.5 * (np.max(sample_pt[:, 2]) + np.min(sample_pt[:, 2]))
    mid_k = np.tan(mid_angle)

    mid_x = np.dot((np.dot(mid_k, sample_pt[:, 1]) - sample_pt[:, 0]), 1 / (mid_k ** 2 - 1))
    mid_y = np.dot(mid_k, mid_x)

    gate_center = [int(round(0.5 * (np.max(mid_x) + np.min(mid_x)))), int(round(0.5 * (np.max(mid_y) + np.min(mid_y))))]

    return gate_center, sample_circle, sample_pt


def gate_detector_clustering(img_path, k_knnmatch=3):
    target = img_path

    img_query = cv2.imread('pattern.png')  # queryImage
    img_query = cv2.cvtColor(img_query, cv2.COLOR_RGB2GRAY)

    img_train = cv2.imread(target)  # trainImage
    img_train = cv2.cvtColor(img_train, cv2.COLOR_RGB2GRAY)
    # Create mask and dilate it
    mask = cv2.inRange(img_train, 210, 255)
    kernel = np.ones((8, 8), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_query, None)
    kp2, des2 = sift.detectAndCompute(img_train, mask=dilation)

    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=k_knnmatch)

    # Save the matched keypoints into a list
    pt_match = []
    for m in matches:
        for x in m:
            pt_match.append([int(round(kp2[x.trainIdx].pt[0])), int(round(kp2[x.trainIdx].pt[1]))])
    pt_match = np.asarray(pt_match)

    # Execute DBSCAN method to find data clusters
    cluster_labels = DBSCAN(eps=20, min_samples=5).fit_predict(pt_match)
    labels = np.sort(cluster_labels)

    # Idx to be deleted from the labels. If label=-1, this means this point does not belong to any cluster.
    i_2_del = []
    for i, num in enumerate(labels):
        if num < 0:
            i_2_del.append(i)
    labels = np.delete(labels, i_2_del)
    counter = collections.Counter(labels)

    # Sort the labels w.r.t the size of each cluster
    counter = np.asarray([list(counter.keys()), list(counter.values())])
    c = counter[:, counter[1].argsort()]

    # Store the labels of the clusters with max size. Maximum save 4 clusters.
    max_label = []
    cluster = []
    for i in range(0, min(4, c.shape[1])):
        max_label.append(c[0, -1-i])
        cluster.append([])
    # Group the points that belong to the same cluster according to the label
    for pt, label in zip(pt_match, cluster_labels):
        for i, lab in enumerate(max_label):
            if label == lab:
                cluster[i].append(pt)
            else:
                pass
    # Average the x, y of each cluster to find the mean point
    avg_pt = []
    for clst in cluster:
        clst = np.asarray(clst)
        avg_pt.append([int(np.sum(clst[:, 0]) / clst.shape[0]), int(np.sum(clst[:, 1]) / clst.shape[0])])

    return avg_pt

