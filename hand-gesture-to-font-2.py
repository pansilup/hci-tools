# -*- coding: utf-8 -*-
"""
Author         : pansilup
Data           : Apr-21-2022
What this does :
Gesture input - measures the angle made by thumb with horizontal axis
Output        - variable 'font' is updated realtime based on the gesture input

This program is written on top of the example provided in google mediapipe
https://google.github.io/mediapipe/solutions/hands.html

"""

import cv2
import mediapipe as mp
import numpy as np
import math

#variable 'font' is updated realtime based on the gesture input
font = 10 #initially font is set to 10

font_min = 10
font_max = 20
    
    
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

print("looking around ...")
# For webcam input:
cap = cv2.VideoCapture(0)

f_count_s = 0
f_count_e = 0
dist_prev = 0
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
     
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    i = 0
    if results.multi_hand_landmarks:
      f_count_e = f_count_e + 1 
      for hand_landmarks in results.multi_hand_landmarks:
        #debug : print landmarks
        #print("thumb_tip: \n", hand_landmarks.landmark[4].y)
        #print("thumb_beg: \n", hand_landmarks.landmark[2].y)
        thumb_t = hand_landmarks.landmark[4]
        thumb_s = hand_landmarks.landmark[2]
        dy = thumb_t.y - thumb_s.y
        dx = thumb_t.x - thumb_s.x
        if dx == 0:
            dx = 0.0000001
        tanx = dy/dx
        radx = math.atan(np.sqrt(tanx*tanx)) #angle of thumb
                
        font_tmp = font_min + (font_max-font_min)*(2/math.pi)*radx
        font_tmp = int(font_tmp)
        if font_tmp < font_min:
            font  = font_min
        elif font_tmp > font_max:
            font = font_max
        else:
            font = font_tmp
        print("font based on the angle of the thumb ", font)     
        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    else:
        dist_prev = 0
        print("font(static) : ", font)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
