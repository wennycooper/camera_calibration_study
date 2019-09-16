import cv2
import numpy as np
import time
import sys

#cap = cv2.VideoCapture('/home/kkuei/KH_videos/KH-004.avi')
cap = cv2.VideoCapture(sys.argv[1])


if (cap.isOpened() == False):
  print("Error opening video stream")
  exit

imageIndex = 0
while(cap.isOpened()):
  ret, frame = cap.read()

  if ret == True:
    cv2.namedWindow('frame')
    cv2.imshow('frame', frame)

    keyPress = cv2.waitKey(100) & 0xFF
    if keyPress == ord('q'):
      break

    if keyPress == ord('c'):
      cv2.imwrite("chessboard_"+str(imageIndex)+".jpg", frame)
      print("chessboard_"+str(imageIndex))
      imageIndex = imageIndex + 1

  else:  # Break the loop
    print(ret)
    break
    
cap.release()

cv2.destroyAllWindows()
  
