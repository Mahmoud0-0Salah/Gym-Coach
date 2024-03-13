import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
type =-1
text_offset_x = 0
text_offset_y = 15
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = .6
font_thickness = 2
bg_color = (0, 0, 255)  # Background color (black in this case)
text_color = (0, 0, 0)
cap = cv2.VideoCapture(0)

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1300, 700))
    frame.flags.writeable=True
    if (type==-1):
        cv2.putText(frame, "press 1 for Dumbbell Bicep Curl", (450, 25), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, "or q to exit", (450, 60), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.imshow('Mediapipe Feed', frame)
    key=cv2.waitKey(10)
    if key & 0xFF == ord('1'):

        type=1
        break
    elif key & 0xFF == ord('q'): 
        break 
        
cap.release()
cv2.destroyAllWindows()
