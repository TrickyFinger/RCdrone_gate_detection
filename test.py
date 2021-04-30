import os
import cv2
import time
import matplotlib.pyplot as plt
import re
from gate_detector import gate_detector_sampling, gate_detector_clustering
import numpy as np

# Sort images and put them in a list
filedir = os.path.join(os.getcwd(), 'img_gates')
image_type = 'png'
image_names = []
for file in os.listdir(filedir):
    if file.endswith(image_type) and file.startswith('img'):
        image_names.append(os.path.join(filedir, file))
    image_names.sort(key=lambda f: int(re.sub('\D', '', f)))

idx = 0
while idx < len(image_names):
    img_tmp = cv2.imread(image_names[idx])
    start = time.time()

    pts = np.asarray(gate_detector_clustering(image_names[idx]))

    # Average the x, y of the corner points as the gate center
    gate_center = [int(sum(pts[:, 0]) / len(pts)), int(sum(pts[:, 1]) / len(pts))]

    # If the center moves too much from the previous one, we won't trust it. We then trust more the previous one.
    prev_weight = 0.8     # Weight of the previous center
    deviation_limit = 40  # Allowed deviation distance
    if idx > 0:
        if np.sqrt((gate_center[0]-prev_gate_center[0])**2 + (gate_center[1]-prev_gate_center[1])**2) > deviation_limit:
            gate_center[0] = int((1-prev_weight)*gate_center[0] + prev_weight*prev_gate_center[0])
            gate_center[1] = int((1-prev_weight)*gate_center[1] + prev_weight*prev_gate_center[1])

    # Draw everything for visualization
    for pt in pts:
        img_tmp = cv2.circle(img_tmp, tuple(pt), 10, (0, 0, 255), -1)
    img_tmp = cv2.circle(img_tmp, tuple(gate_center), 15, (16, 155, 5), 3)

    # Update
    prev_gate_center = gate_center
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