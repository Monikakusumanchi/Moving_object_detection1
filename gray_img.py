import cv2
img = cv2.imread("WIN_20210423_14_57_51_Pro.jpg")
grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threshImg=cv2.threshold(grayImg,195,255,cv2.THRESH_BINARY)[1]
cv2.imwrite("thresholdImage.jpg",threshImg)
