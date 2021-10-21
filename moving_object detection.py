
import cv2
import imutils
import time
vs = cv2.VideoCapture(0)
firstFrame = None
area=500                    #to detect only objects but not air or blanket
while True:
    _,img = vs.read()            #read from cam
    text = "Normal"              #
    img = imutils.resize(img, width=500)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #gray image
    #gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
    if firstFrame is None:
        firstFrame = grayImg
        continue
    imgDiff = cv2.absdiff(firstFrame, grayImg)
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)      #to remove the holes
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,  
           cv2.CHAIN_APPROX_SIMPLE)                        #to find the area of total path
    
    cnts = imutils.grab_contours(cnts)               #to connect the dots
    for c in cnts:                          #for getting  rectangle
         if cv2.contourArea(c) < area:
             continue
         (x, y, w, h) = cv2.boundingRect(c)
         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
         text = "Moving Object detected"
    print(text)
    cv2.putText(img, text, (10, 20),
         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("VideoStream", img)
    #cv2.imshow("Thresh", threshImg)
    #cv2.imshow("Image Difference", imgDiff)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()
