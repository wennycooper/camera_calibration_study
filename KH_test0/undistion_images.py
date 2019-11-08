import numpy as np
import cv2
import glob
import codecs, json

# images = glob.glob('chessboard_*')
images = glob.glob('vlc*.png')


# load calibration parameters
file_path = "params.json"
with open(file_path, 'r') as f:
    out = json.load(f)

K = np.array(out["K"])
D = np.array(out["D"])

# Undistortion
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]       # (1080, 1920)
    DIM = img.shape[:2][::-1]  # (1920, 1080)

    mapx,mapy = cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),K,DIM,cv2.CV_16SC2)
    dst = cv2.remap(img,mapx,mapy,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(fname + "_result.jpg", dst)

    cv2.imshow("img", dst)
    cv2.waitKey(1000)

# Re-projection Error
#mean_error = 0
#for i in xrange(len(objpoints)):
#    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#    mean_error += error
#
#print "mean error: ", mean_error/len(objpoints)

