#!/usr/bin/env python
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

while True:
    # Capture each frame of webcam video
    ret,frame = cam.read()

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_color = np.array([90, 70, 70])
    upper_color = np.array([150, 255, 255])

    mask = cv2.inRange(img_hsv, lower_color, upper_color)


    # Draw contours
    ret, thresh_img = cv2.threshold(mask, 91, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for c in contours:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)


    result = cv2.bitwise_and(frame, frame, mask = mask)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    
    # FPS
    fps = round (1.0 / (time.time() - start_time), 2)
    cv2.putText(frame, str( fps ), org, font, fontScale, color, thickness, cv2.LINE_AA)
    start_time = time.time()

    cv2.imshow('main', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    if cv2.waitKey(10) == 0x1b:
        break

cam.release()
cv2.destroyAllWindows()
