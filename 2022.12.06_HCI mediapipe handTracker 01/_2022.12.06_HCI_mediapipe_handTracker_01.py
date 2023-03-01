import cv2
import mediapipe as mp

mp_drawing=mp.solutions.drawing_utils # mediapipe 繪圖方法
mp_drawing_styles=mp.solutions.drawing_styles # mediapipe 繪圖樣式
mp_hands=mp.solutions.hands # mediapipe 偵測手掌方式

cap=cv2.VideoCapture(0)

# mediapipe 啟用偵測手掌
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
        img2=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # 將 BGR 轉換成 RGB
        results=hands.process(img2) # 偵測手掌
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 將節點跟骨架繪製到影像中
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow("final",frame)
        if cv2.waitKey(5)==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()