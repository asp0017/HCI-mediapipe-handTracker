import cv2
import mediapipe as mp
import random

mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_hands=mp.solutions.hands

cap=cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    run=True # �O�_��ʸIĲ�Ϧ�m
    while True:
        ret, frame=cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        frame=cv2.resize(frame,(540,320)) # �վ�e���ؤo
        size=frame.shape # ���o���Y�v���ؤo
        w=size[1] # ���o�e���e��
        h=size[0] # ���o�e������
        if run:
            run=False # �p�G�S���I��A�N�@�����|����m
            rx=random.randint(50,w-50) # �H�� x �y��
            ry=random.randint(50,h-100) # �H�� y �y��
            print(rx,ry)
        img2=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(img2)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x=hand_landmarks.landmark[7].x*w # ���o�������� x �y��
                y=hand_landmarks.landmark[7].y*h # ���o�������� y �y��
                print(x,y)
                if x>rx and x<(rx+80) and y>ry and y<(ry+80):
                    run=True
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

        cv2.rectangle(frame,(rx,ry),(rx+80,ry+80),(0,0,255),5) # �e�X�IĲ��
        cv2.imshow("final",frame)
        if cv2.waitKey(5)==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()