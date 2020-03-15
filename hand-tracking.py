# -*- coding: utf-8 -*-

import cv2 
import numpy as np
import time

# Capture video from webcam
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

start_time = time.time()

# Text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
org = (5, 20)
fontScale = 0.5
color = (0, 0, 255)
thickness = 0

# Rectangle parameters
rect_x1 = 200
rect_y1 = 100
rect_x2 = 300
rect_y2 = 200
rect_b = 0
rect_g = 255
rect_r = 0
rect_thickness = 2


while(True):
    # Capture each frame of webcam video
    ret,frame = cam.read()

    height, width = frame.shape[:2]
    edge = 10

    # rectangle
    # cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (rect_b, rect_g, rect_r), rect_thickness)

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # BGR
    # blue lower: 90 70 70 upper: 150 255 255
    # und  lower: 0 20 70 upper: 20 255 255
    # 
    lower_color = np.array([0, 20, 70])
    upper_color = np.array([20, 255, 255])

    mask = cv2.inRange(img_hsv, lower_color, upper_color)


    # Draw contours
    ret, thresh_img = cv2.threshold(mask, 91, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for c in contours:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)


    result = cv2.bitwise_and(frame, frame, mask = mask)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    moments = cv2.moments(mask, 1)

    dM01 = moments['m01'] # Сумма координат по оси x
    dM10 = moments['m10'] # Сумма координат по оси y
    dArea = moments['m00']

    x = 0
            
    if dArea > 150:
        x = int(dM10 / dArea) 
        y = int(dM01 / dArea)
        cv2.rectangle(frame, (x - (width / 4), y - (height / 4)), (x + (width / 4), y + (height / 4)), (0, 0, 255), 1)

    # FPS
    fps = round (1.0 / (time.time() - start_time), 2)
    cv2.putText(frame, str( fps ), org, font, fontScale, color, thickness, cv2.LINE_AA)
    start_time = time.time()

    # Show
    cv2.imshow('main', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    # Close and break the loop after pressing "esc" key
    if cv2.waitKey(10) == 0x1b:
        break

# close the already opened camera
cam.release()
# close the window and de-allocate any associated memory usage
cv2.destroyAllWindows()