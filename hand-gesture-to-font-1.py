# -*- coding: utf-8 -*-
"""
Author         : pansilup
Data           : Apr-19-2022
What this does :
Gesture input - measures the distance between the tips of thumb  and index fingers
Output        - variable 'font' is updated realtime based on the gesture input

This program is written on top of the example provided in google mediapipe
https://google.github.io/mediapipe/solutions/hands.html

"""

import cv2
import mediapipe as mp
import numpy as np

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
        #print("thumb_tip: \n", hand_landmarks.landmark[4])
        #print("index_tip: \n", hand_landmarks.landmark[8])
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        dist = np.sqrt((thumb.x-index.x)**2+(thumb.y-index.y)**2+(thumb.x-index.z)**2)
        
        print("distance between thumb and index finger tips : ", dist)
        if dist_prev > 0:
            delta = dist - dist_prev
            font_tmp = font + 30*delta
            if font_tmp < font_min:
                font  = int(font_min)
            elif font_tmp > font_max:
                font = int(font_max)
            else:
                font = int(font_tmp)
            print("gesture input based font : ", font)
        dist_prev = dist
        
        #debug : printing landmarks
        #for point in mp_hands.HandLandmark:
        #    normalizedLandmark = hand_landmarks.landmark[point]
        #    print("points: ", point)
        #    print("landmarks: ", normalizedLandmark)
        
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
