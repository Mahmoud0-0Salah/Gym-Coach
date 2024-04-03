import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

text_offset_x = 0
text_offset_y = 15
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.6
font_thickness = 2
bg_color = (0, 0, 255) 
text_color = (0, 0,  0)
cap = cv2.VideoCapture(0)

#####################################################################################################
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
#####################################################################################################
def Dumbbell_Bicep_Curl(frame,image,pose,bg_color,side,stage,text):
                
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True


                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

                # Define rectangle coordinates
                rectangle_coords = ((text_offset_x, text_offset_y + 30), (text_offset_x + 100, text_offset_y - 60))

                # Draw filled rectangle
                cv2.rectangle(image, rectangle_coords[0], rectangle_coords[1], bg_color, cv2.FILLED)
                key=-1
                try:
                    landmarks = results.pose_landmarks.landmark
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                    rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    # Calculate angle
                    angle = round(calculate_angle(shoulder, elbow, wrist))
                    angle2 = round(calculate_angle(rshoulder, relbow, rwrist))

                    if side == -1:
                        cv2.putText(image, "press 1 for left hand or 2 to right hand or q to exit", (500, 25), font, font_scale,
                                    text_color, font_thickness, cv2.LINE_AA)
                    else:
                        cv2.putText(image, "q to exit", (500, 25), font, font_scale,
                                    text_color, font_thickness, cv2.LINE_AA)
                    # Visualize angle
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [1300, 700]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    cv2.putText(image, str(angle2),
                                tuple(np.multiply([relbow[0] - .05, relbow[1]], [1300, 700]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                    if side == -1:
                        key=cv2.waitKey(10)
                        if key & 0xFF == ord('1'):
                            side = 1
                        if key & 0xFF == ord('2'):
                            side = 2
                    elif side == 1:
                        key=cv2.waitKey(10)
                        if angle > 160:
                            stage = "down"
                            text = "Wait"
                            bg_color = (255, 0, 0)
                        if angle < 30 and stage == 'down':
                            stage = "up"
                            bg_color = (0, 255, 0)
                            text = "Correct"
                    elif side == 2:
                        key=cv2.waitKey(10)
                        if angle2 > 160:
                            stage = "down"
                            text = "Wait"
                            bg_color = (255, 0, 0)
                        if angle2 < 30 and stage == 'down':
                            stage = "up"
                            bg_color = (0, 255, 0)
                            text = "Correct"

                    cv2.putText(image, text, (text_offset_x + 25, text_offset_y + 10), font, font_scale, text_color,
                                font_thickness, cv2.LINE_AA)
                except:
                    pass
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                return(image,bg_color,side,stage,text,key)
#####################################################################################################

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1300, 700))
        cv2.putText(frame, "press 1 for Dumbbell Bicep Curl", (450, 25), font, font_scale, text_color, font_thickness,
                        cv2.LINE_AA)
        cv2.putText(frame, "or q to exit", (450, 60), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.imshow('Mediapipe Feed', frame)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('1'):
            cv2.destroyAllWindows()
            side = -1
            text = 'Wait'
            stage = 'down'
            bg_color=(0,0,255)
            while cap.isOpened():
                ret,frame = cap.read()
                frame = cv2.resize(frame, (1300, 700))
                image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                (image,bg_color,side,stage,text,key)=Dumbbell_Bicep_Curl(frame,image,pose,bg_color,side,stage,text)
                if key & 0xFF == ord('q'):
                    break  
                cv2.imshow("Dumbbell Bicep Curl", image)
            
        elif key & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
