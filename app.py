import cv2
import numpy as np
import pytesseract
import mediapipe as mp
import time
import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO/WARN messages

# Set Tesseract path (Modify this based on your installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create a white canvas for drawing
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Open Webcam
cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None  # Previous finger position
drawing = True  # Track if drawing is enabled
pinch_start_time = None  # Timer for pinch detection
fist_start_time = None  # Timer for fist detection
SEARCH_QUERY_FILE = "search_query.txt"
recognized_text_preview = ""  # Store recognized text for preview

def save_recognized_text(text):
    """Saves recognized text to a file, clearing it before each update."""
    with open(SEARCH_QUERY_FILE, "w", encoding="utf-8") as file:
        file.write(text.strip())

def recognize_text():
    """Recognizes handwritten text from the drawing canvas using Tesseract OCR."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6 --oem 3')
    return text.strip()

def detect_pinching(hand_landmarks):
    """Detects if the index finger and thumb are touching for 1 second."""
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    distance = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y]))
    return distance < 0.05

def detect_fingers_together(hand_landmarks):
    """Detects if the index and middle fingers are touching to stop drawing."""
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    distance = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y]))
    return distance < 0.05

def detect_fist(hand_landmarks):
    """Detects if a fist is held to recognize text and save."""
    finger_tips = [8, 12, 16, 20]
    return all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y for tip in finger_tips)

def draw_on_canvas(x, y):
    """Draws on the screen using the detected finger position."""
    global prev_x, prev_y, drawing
    if drawing:
        if prev_x is not None and prev_y is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 5)
        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None  # Stop drawing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            # Toggle drawing mode when index + middle fingers touch
            if detect_fingers_together(hand_landmarks):
                drawing = False
            else:
                drawing = True

            # Draw on the canvas if drawing mode is enabled
            draw_on_canvas(x, y)

            # Detect pinch gesture to clear the screen
            if detect_pinching(hand_landmarks):
                if pinch_start_time is None:
                    pinch_start_time = time.time()
                elif time.time() - pinch_start_time >= 1:
                    canvas[:] = 255  # Clear screen
                    pinch_start_time = None  # Reset timer
            else:
                pinch_start_time = None  # Reset if fingers are apart

            # Detect fist to preview and save recognized text
            if detect_fist(hand_landmarks):
                if fist_start_time is None:
                    fist_start_time = time.time()
                elif time.time() - fist_start_time >= 1 and recognized_text_preview == "":
                    recognized_text_preview = recognize_text()
                elif time.time() - fist_start_time >= 3:
                    save_recognized_text(recognized_text_preview)
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
            else:
                fist_start_time = None  # Reset if fist is not held
                recognized_text_preview = ""  # Reset preview if fist is released

    # Display recognized text preview on the webcam screen
    if recognized_text_preview:
        cv2.putText(frame, f"Preview: {recognized_text_preview}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the drawing on screen
    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Virtual Drawing", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
