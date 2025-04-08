import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
from disparity import getDisparityMap
from disparity import plot
from disparity import plot2

f=5806.559 #focal length in pixels
cam0cx=1429.219 #camera center x in pixels
cam0cy=993.403 #camera center y in pixels
cam1cx=1543.51 #camera center x in pixels
cam1cy=993.403 #camera center y in pixels

doffs=114.291 
baseline=174.019 
width=2960 
height=2016

#22.2 mm x 14.8 mm
sensor= 22.2
#3088 x 2056
image_width_pixels = 3088
#just using width for now to get ratio
# Convert to mm
focal_length_mm = f * sensor / image_width_pixels



# Load and preprocess

img1 = cv2.imread("umbrellaL.png")
img2 = cv2.imread("umbrellaR.png")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



def nothing(x):
    pass


def task2():
    # Create window and trackbars
    cv2.namedWindow("Disparity")
    cv2.createTrackbar("NumDisparities", "Disparity", 4, 10, nothing)  
    cv2.createTrackbar("BlockSize", "Disparity", 5, 50, nothing)       # must be odd
    cv2.createTrackbar("Canny Edge Thereshodl 1", "Disparity", 76, 200, nothing)  
    cv2.createTrackbar("Canny Edge Thereshodl 2", "Disparity", 150, 200, nothing)  #


    while True:
        # Get trackbar positions
        numDisparities = cv2.getTrackbarPos("NumDisparities", "Disparity") * 16
        blockSize = cv2.getTrackbarPos("BlockSize", "Disparity")
        thresh1=  cv2.getTrackbarPos("Canny Edge Thereshodl 1", "Disparity")
        thresh2= cv2.getTrackbarPos("Canny Edge Thereshodl 2", "Disparity")

        # Make block size odd and >= 5
        # Sanitize input
        blockSize = blockSize
        if blockSize < 5:
            blockSize = 5
        if blockSize % 2 == 0:
            blockSize += 1

    
       
        # Compute disparity map
        # Apply Canny edge detection
        edge1 = cv2.Canny(gray1, thresh1, thresh2)
        edge2 = cv2.Canny(gray2, thresh1, thresh2)
        disp_map = getDisparityMap(edge1, edge2, numDisparities, blockSize)
        disp_map = cv2.GaussianBlur(disp_map, (5, 5), 0.5)
        disparityImg = np.interp(disp_map, (disp_map.min(), disp_map.max()), (0.0, 1.0))
        #disp_vis = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX)
        #disp_vis = cv2.convertScaleAbs(disp_vis)
        # Show result
        cv2.imshow("Disparity", disparityImg)

        # Exit on ESC or space
        key = cv2.waitKey(1)
        if key == 27 or key == ord(' '):
            break

    cv2.destroyAllWindows()

def task3():
    numDisparities = 64
    blockSize = 15
    thresh1=  76
    thresh2= 150
    edge1 = cv2.Canny(gray1, thresh1, thresh2)
    edge2 = cv2.Canny(gray2, thresh1, thresh2)
    disp_map = getDisparityMap(edge1, edge2, numDisparities, blockSize)
    # Scale f, cx, cy, doffs to match resized image (740x505)
    scale_x = 740 / width
    scale_y = 505 / height
    f_scaled = f * scale_x
    doffs_scaled = doffs * scale_x
    cx = cam0cx * scale_x
    cy = cam0cy * scale_y
    
    #diairyt image h pixel has x y and z value of each pixel as 1 2 and 3 \
    h, w = disp_map.shape
    points = []
    for y in range(0,h):
        for x in range(0,w):
            d = disp_map[y, x]
            Z = (f_scaled * baseline) / (d + doffs_scaled)
            if Z > 5000 or Z <100:  # or whatever threshold you find useful
                continue
            X = (x - cx) * Z / f_scaled
            Y = (y - cy) * Z / f_scaled
            points.append((X, Y, Z))
    plot2(points)
    
    
def Full_pipeline():

    # === Setup ===
    cv2.namedWindow("Disparity")
    cv2.createTrackbar("NumDisparities", "Disparity", 4, 10, nothing)
    cv2.createTrackbar("BlockSize", "Disparity", 15, 50, nothing)
    cv2.createTrackbar("Canny Edge Thereshodl 1", "Disparity", 76, 400, nothing)
    cv2.createTrackbar("Canny Edge Thereshodl 2", "Disparity", 150, 400, nothing)

    while True:
        # === Read slider values ===
        numDisparities = cv2.getTrackbarPos("NumDisparities", "Disparity") * 16
        blockSize = cv2.getTrackbarPos("BlockSize", "Disparity")
        thresh1 = cv2.getTrackbarPos("Canny Edge Thereshodl 1", "Disparity")
        thresh2 = cv2.getTrackbarPos("Canny Edge Thereshodl 2", "Disparity")

        # Sanitize blockSize
        if blockSize < 5:
            blockSize = 5
        if blockSize % 2 == 0:
            blockSize += 1

        # === Compute disparity map ===
        edge1 = cv2.Canny(gray1, thresh1, thresh2)
        edge2 = cv2.Canny(gray2, thresh1, thresh2)
        disp_map = getDisparityMap(edge1, edge2, numDisparities, blockSize)
        #disp_map = cv2.GaussianBlur(disp_map, (5, 5), 0.5)
        disp_vis = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)

        cv2.imshow("Disparity", disp_vis)

        # === Key triggers ===
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == 32:  # SPACE to generate 3D model
            # === Scale camera parameters ===
            scale_x = 740 / width
            scale_y = 505 / height
            f_scaled = f * scale_x
            doffs_scaled = doffs * scale_x
            cx = cam0cx * scale_x
            cy = cam0cy * scale_y

            # === Reconstruct 3D ===
            h, w = disp_map.shape
            points = []
            for y in range(0, h, 2):  # Downsample for speed
                for x in range(0, w, 2):
                    d = disp_map[y, x]
                    if d <= 0.5:
                        continue
                    Z = (f_scaled * baseline) / (d + doffs_scaled)
                    if Z > 5000:
                        continue
                    X = (x - cx) * Z / f_scaled
                    Y = (y - cy) * Z / f_scaled
                    points.append((X, Y, Z))

            print(f"Plotting {len(points)} points...")
            plot(points)

    cv2.destroyAllWindows()

Full_pipeline()