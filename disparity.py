import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt



# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================




# ================================================
#
def plot2(disparity):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    X = [p[0] for p in disparity]
    Y = [p[1] for p in disparity]
    Z = [p[2] for p in disparity]

    # Just use fixed color, skip cmap
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X, Y, Z, s=0.1, color='black')
    ax1.set_title("3D View")
    ax1.view_init(elev=25, azim=45)

    ax2 = fig.add_subplot(132)
    ax2.scatter(X, Z, s=0.1, color='black')
    ax2.set_title("Top View (X-Z)")

    ax3 = fig.add_subplot(133)
    ax3.scatter(Y, Z, s=0.1, color='black')
    ax3.set_title("Side View (Y-Z)")

    plt.tight_layout()
    plt.savefig('3d_plot_output.pdf', bbox_inches='tight')
    plt.show()


# ================================================
#

def plot(disparity):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    x = [p[0] for p in disparity]
    y = [p[1] for p in disparity]
    z = [p[2] for p in disparity]
    

    # 3D View
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # One subplot, 3D

    ax.scatter(x, y, z, s=0.1, color='green')   # s controls dot size
    ax.view_init(elev=-90, azim=-90)
    ax.set_title("3D View")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig("single _3d_plot_output.pdf", bbox_inches="tight")
    plt.show()
# ================================================
if __name__ == '__main__':

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Get disparity map
    disparity = getDisparityMap(imgL, imgR, 64, 5)

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    # Show result
    cv2.imshow('Disparity', disparityImg)

    # Show 3D plot of the scene
    plot(disparity)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
