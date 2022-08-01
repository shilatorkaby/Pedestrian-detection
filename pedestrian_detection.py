import cv2
import numpy as np

video_list = ['video.mp4','video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.avi']

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 300)
fontScale = 1.5
fontColor = (255, 255, 255)
thickness = 1
lineType = 1
warning = ["warning! pedestrian!", "warning!"]


for vid in video_list:
    cap = cv2.VideoCapture(vid) #used opencv VideoCapture class in order to open video file
    #cap = cv2.VideoCapture(0) #open the camera

    pedestrian_cascade = cv2.HOGDescriptor()
    pedestrian_cascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #pedestrian_cascade = cv2.CascadeClassifier('model\\cascade.xml')

    i=0
    j = 0
    logger = np.zeros((512, 512, 3), np.uint8)
    while True:
        ret, img = cap.read()

        if type(img) == type(None):
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #pedestrian = pedestrian_cascade.detectMultiScale(gray, 1.3, 2)
        (regions, _) = pedestrian_cascade.detectMultiScale(img,winStride=(4, 4),padding=(4, 4),scale=1.05)

        for (x, y, w, h) in regions:
            cv2.rectangle(img, (x, y),(x + w, y + h),(60, 45, 55), 2)
            if w!=0 and h!=0:
                j+=1

                logger = np.zeros((512, 512, 3), np.uint8)

                cv2.putText(logger, warning[j % 2],bottomLeftCornerOfText,
                            font, fontScale,fontColor,
                            thickness,lineType)
                # Display the image
                cv2.imshow("img", logger)
                cv2.waitKey(1)

        #for (a, b, c, d) in pedestrian:
        #    cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 210), 4)
            # print("there is a person")

        # img = cv2.resize(img, None, None, fx=7, fy=4)
        cv2.imshow('video', img)
        key= cv2.waitKey(1)
        if key == ord(' '):
            key = cv2.waitKey() #press the spacebar - to pause the video
        if key == ord('s'): #save the image
            i+=1
            cv2.imwrite('images\\' + vid[:-4] + "_test" + str(i)+'.jpg',img)
            key = cv2.waitKey()
        if cv2.waitKey(1) == ord('q'): #press 'q' twice to stop the video
            break

    cv2.destroyAllWindows()
