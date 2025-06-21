
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Volume control setup using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# Utility function to count fingers
def count_fingers(lmList):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if lmList[tips[0]][1] > lmList[tips[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if lmList[tips[id]][2] < lmList[tips[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

def draw_text(img, text, pos=(10, 70), scale=1, color=(0,255,0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

# Gesture dictionary for game
gestures = {0: 'Rock', 2: 'Scissors', 5: 'Paper'}

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to grab frame.")
        continue

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lmList = []
        for id, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if lmList:
            finger_count = count_fingers(lmList)
            draw_text(img, f"Fingers: {finger_count}")

            # Volume control
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            length = math.hypot(x2 - x1, y2 - y1)

            vol = np.interp(length, [20, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

            # Mouse move
            if finger_count == 1:
                x, y = lmList[8][1], lmList[8][2]
                screen_x, screen_y = pyautogui.size()
                pyautogui.moveTo(screen_x - x * 2, y * 2)

            # Left click
            if finger_count == 2:
                pyautogui.click()

            # Screenshot
            if finger_count == 5:
                draw_text(img, "Screenshot!", (10, 150), 1, (0, 0, 255))
                cv2.imwrite(f"screenshot_{int(time.time())}.png", img)
                time.sleep(1)

            # Rock Paper Scissors
            if finger_count in gestures:
                computer = random.choice(list(gestures.values()))
                user = gestures[finger_count]
                draw_text(img, f"You: {user}, CPU: {computer}", (10, 420), 0.8)

    cv2.imshow("Hand Gesture System", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
