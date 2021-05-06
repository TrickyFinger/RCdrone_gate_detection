# RCdrone_gate_detection
TU Delft MAV course individual assignment

1. Files start with 'test_xxx' were used at the beginning to study how well each approach performs.

2. The final code are in main.py and gate_detector.py

3. main.py: Go through all images in the dataset and display the results of gate detector.

4. gate_detector.py: Contains functions to be executed in the main.py
   - Function 1: gate_detector_sampling.
     This is the function developed by myself which uses circles to sample all the interested area to perform data clustering task. It is slow and lacking accuracy. So I abandoned it after discovering DBSCAN.
   - Function 2: gate_detector_clustering.
     This function uses DBSCAN to perform data clustering task. It returns the coordinates of four corners.
