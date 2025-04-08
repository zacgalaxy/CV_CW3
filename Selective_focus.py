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
img1 = cv2.imread("girlL.png")
img2 = cv2.imread("girlR.png")

grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grey2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def nothing(x):
    pass

def Full_pipeline_3dMap():

    # === Setup ===
    cv2.namedWindow("Disparity")
    cv2.createTrackbar("NumDisparities", "Disparity", 1, 10, nothing)
    cv2.createTrackbar("BlockSize", "Disparity", 5, 50, nothing)
    cv2.createTrackbar("Canny Edge Thereshodl 1", "Disparity", 200, 400, nothing)
    cv2.createTrackbar("Canny Edge Thereshodl 2", "Disparity", 200, 400, nothing)

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
        edge1 = cv2.Canny(grey1, thresh1, thresh2)
        edge2 = cv2.Canny(grey2, thresh1, thresh2)
        disp_map = getDisparityMap(edge1, edge2, numDisparities, blockSize)
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
            for y in range(0, h, 2 ):  # Downsample for speed
                for x in range(0, w, 2):
                    d = disp_map[y, x]
                    if d <= 0.5:
                        continue
                    Z = (f_scaled * baseline) / (d + doffs_scaled)
                    if Z > 8500:
                        continue
                    X = (x - cx) * Z / f_scaled
                    Y = (y - cy) * Z / f_scaled
                    points.append((X, Y, Z))

            print(f"Plotting {len(points)} points...")
            plot(points)

    cv2.destroyAllWindows()

def Selective_Focus():
    
    # === Setup ===
    cv2.namedWindow("Disparity")
    cv2.createTrackbar("NumDisparities", "Disparity", 1, 10, nothing)
    cv2.createTrackbar("BlockSize", "Disparity", 5, 255, nothing)
    cv2.createTrackbar("Canny Edge Thereshodl 1", "Disparity", 200, 400, nothing)
    cv2.createTrackbar("Canny Edge Thereshodl 2", "Disparity", 200, 400, nothing)
    cv2.createTrackbar("k", "Disparity", 2, 20, nothing)
    cv2.createTrackbar("Depth Threshold", "Disparity", 100, 255, nothing)

    while True:
        # === Read slider values ===
        numDisparities = cv2.getTrackbarPos("NumDisparities", "Disparity") * 16
        blockSize = cv2.getTrackbarPos("BlockSize", "Disparity")
        thresh1 = cv2.getTrackbarPos("Canny Edge Thereshodl 1", "Disparity")
        thresh2 = cv2.getTrackbarPos("Canny Edge Thereshodl 2", "Disparity")
        k = cv2.getTrackbarPos("k", "Disparity")
        depth_thresh = cv2.getTrackbarPos("Depth Threshold", "Disparity")
        # Sanitize blockSize
        if blockSize < 5:
            blockSize = 5
        if blockSize % 2 == 0:
            blockSize += 1
        if k == 0:
            k = 1


        # === Compute disparity map ===
        edge1 = cv2.Canny(grey1, thresh1, thresh2)
        edge2 = cv2.Canny(grey2, thresh1, thresh2)
        disp_map = getDisparityMap(grey1, grey2, numDisparities, blockSize)
        #disp_map = cv2.medianBlur(disp_map, 5)
        disp_vis = cv2.normalize(disp_map, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)
        cv2.imshow("Disparity", disp_vis)
        
        

        # === Compute depth image ===
        depth = 1.0 / (disp_map + k)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #depth_norm = cv2.medianBlur(depth_norm, 5) 
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imshow("Depth Map", depth_color)
        
        # === Foreground-background segmentation ===
        _, foreground_mask = cv2.threshold(depth_norm, depth_thresh, 255, cv2.THRESH_BINARY_INV)
        soft_mask = cv2.medianBlur(foreground_mask, 11) .astype(np.float32) / 255.0  # Normalize to [0,1]

        #foreground_mask = foreground_mask.astype(np.uint8)
        #background_mask = cv2.bitwise_not(foreground_mask)
        
        # === Effect: Background Grayscale ===
        gray_bg = cv2.cvtColor(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
        img1_float = img1.astype(np.float32)
        # Blend the images
        output = img1_float * soft_mask[:, :, None] + gray_bg * (1 - soft_mask[:, :, None])
        output = np.clip(output, 0, 255).astype(np.uint8)
        cv2.imshow("Selective Focus", output)
        
        # Key handling
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        
    cv2.destroyAllWindows()
    
    
Selective_Focus()



