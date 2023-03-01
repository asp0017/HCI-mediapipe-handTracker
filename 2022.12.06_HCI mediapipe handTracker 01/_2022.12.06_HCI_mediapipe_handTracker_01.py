import cv2
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils # mediapipe ø�Ϥ�k
mp_drawing_styles=mp.solutions.drawing_styles # mediapipe ø�ϼ˦�
mp_hands=mp.solutions.hands # mediapipe ������x�覡

cap=cv2.VideoCapture(0)

# mediapipe �ҥΰ�����x
with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame=cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img2=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # �N BGR �ഫ�� RGB
        results=hands.process(img2) # ������x
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # �N�`�I�򰩬[ø�s��v����
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow("final",frame)
        if cv2.waitKey(5)==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()