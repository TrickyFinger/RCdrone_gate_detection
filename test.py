import os
import cv2
import time
import matplotlib.pyplot as plt
import re
from gate_detector import gate_detector_sampling, gate_detector_clustering
import numpy as np


dir = os.path.join(os.getcwd(), 'img_gates')
image_type = 'png'
image_names = []
for file in os.listdir(dir):
    if file.endswith(image_type) and file.startswith('img'):
        image_names.append(os.path.join(dir, file))
    image_names.sort(key=lambda f: int(re.sub('\D', '', f)))
# loop over frames from the video stream
idx = 0
while idx < len(image_names):
    img = cv2.imread(image_names[idx])
    start = time.time()

    [pt1, pt2, pt3, pt4] = gate_detector_clustering(image_names[idx])
    gate_center = [int((pt1[0] + pt2[0] + pt3[0] + pt4[0]) / 4), int((pt1[1] + pt2[1] + pt3[1] + pt4[1]) / 4)]

    # print(time.time() - start)
    if idx > 0:
        if np.sqrt((gate_center[0]-pre_gate_center[0])**2 + (gate_center[1]-pre_gate_center[1])**2) > 40:
            gate_center[0] = int(0.2*gate_center[0] + 0.8*pre_gate_center[0])
            gate_center[1] = int(0.2 * gate_center[1] + 0.8*pre_gate_center[1])

    img_tmp = cv2.circle(img, tuple(pt1), 10, (0, 0, 255), -1)
    img_tmp = cv2.circle(img_tmp, tuple(pt2), 10, (0, 0, 255), -1)
    img_tmp = cv2.circle(img_tmp, tuple(pt3), 10, (0, 0, 255), -1)
    img_tmp = cv2.circle(img_tmp, tuple(pt4), 10, (0, 0, 255), -1)
    img_tmp = cv2.circle(img_tmp, tuple(gate_center), 15, (16, 155, 5), 3)

    pre_gate_center = gate_center
    idx += 1
    cv2.imshow('1', img_tmp)
    cv2.waitKey(200)

    # [gate_center, sample_circle, sample_pt] = gate_detector_sampling(image_names[idx], sample_reject_ratio=3)
    '''
        for i in sample_pt:
            x = int(i[0])
            y = int(i[1])
            img = cv2.circle(img, (x,y), 3, (0, 255, 0), -1)
        img_tmp = cv2.circle(img, tuple(gate_center), 5, (0, 0, 255), -1)
    '''