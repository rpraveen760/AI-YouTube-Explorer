import cv2
import numpy as np
import webbrowser
import os
import mediapipe as mp
import time
import googleapiclient.discovery
import urllib.request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO/WARN messages

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
TOP_YOUTUBE_OPTIONS_FILE = "top_youtube_options.txt"
VIDEO_TO_BE_TRANSCRIBED_FILE = "video_to_be_transcribed.txt"
THUMBNAIL_SIZE = (120, 80)  # Ensure consistent thumbnail size


def get_video_details(video_url):
    """Fetches the video title, duration, and thumbnail using the YouTube API."""
    video_id = video_url.split("v=")[-1]
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    request = youtube.videos().list(
        part="snippet,contentDetails",
        id=video_id
    )
    response = request.execute()

    if "items" in response and response["items"]:
        video_info = response["items"][0]
        title = video_info["snippet"]["title"]
        duration = video_info["contentDetails"]["duration"]
        thumbnail_url = video_info["snippet"]["thumbnails"]["high"]["url"]
        return title, duration, thumbnail_url

    return None, None, None


def download_thumbnail(thumbnail_url, index):
    """Downloads and resizes the video thumbnail to display in the UI."""
    thumbnail_path = f"thumbnail_{index}.jpg"
    urllib.request.urlretrieve(thumbnail_url, thumbnail_path)
    img = cv2.imread(thumbnail_path)
    if img is not None:
        img = cv2.resize(img, THUMBNAIL_SIZE)  # Resize to fixed size
        cv2.imwrite(thumbnail_path, img)
    return thumbnail_path


def load_video_options():
    """Loads the top 3 video links from the file and fetches their details."""
    if not os.path.exists(TOP_YOUTUBE_OPTIONS_FILE):
        print(" No top YouTube options found.")
        return []

    with open(TOP_YOUTUBE_OPTIONS_FILE, "r", encoding="utf-8") as file:
        video_links = [line.strip() for line in file.readlines() if line.strip()]

    video_details = []
    for i, video_url in enumerate(video_links[:3]):
        title, duration, thumbnail_url = get_video_details(video_url)
        if title and duration and thumbnail_url:
            thumbnail_path = download_thumbnail(thumbnail_url, i)
            video_details.append({
                "url": video_url,
                "title": title,
                "duration": duration,
                "thumbnail": thumbnail_path
            })
    return video_details


def save_selected_video(video_url):
    """Saves the selected video URL to video_to_be_transcribed.txt, clearing previous content first."""
    with open(VIDEO_TO_BE_TRANSCRIBED_FILE, "w", encoding="utf-8") as file:
        file.write(video_url)
    print(f"Selected video saved: {video_url}")


def run_video_selection_ui():
    """Opens the webcam and displays the video selection UI with gesture controls."""
    video_options = load_video_options()
    if not video_options:
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    selected_index = 0
    fist_start_time = None
    finger_hold_start_time = None

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

                fingers = [8, 12, 16]  # Index, middle, ring
                extended_fingers = [
                    hand_landmarks.landmark[f].y < hand_landmarks.landmark[f - 2].y for f in fingers
                ]

                num_fingers = sum(extended_fingers)
                if num_fingers in [1, 2, 3]:
                    if finger_hold_start_time is None:
                        finger_hold_start_time = time.time()
                    elif time.time() - finger_hold_start_time >= 0.3:
                        selected_index = num_fingers - 1
                else:
                    finger_hold_start_time = None

                # Detect fist for confirmation
                if all(hand_landmarks.landmark[f].y > hand_landmarks.landmark[f - 2].y for f in fingers):
                    if fist_start_time is None:
                        fist_start_time = time.time()
                    elif time.time() - fist_start_time >= 2:
                        selected_video = video_options[selected_index]
                        webbrowser.open(selected_video["url"])
                        save_selected_video(selected_video["url"])
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                else:
                    fist_start_time = None  # Reset if fist is not held

        # Display UI with improved alignment
        for i, video in enumerate(video_options):
            y_offset = 120 + i * 140
            color = (0, 255, 0) if i == selected_index else (255, 255, 255)
            cv2.putText(frame, f"{i + 1}. {video['title']}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Duration: {video['duration']}", (50, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            thumbnail = cv2.imread(video["thumbnail"])
            if thumbnail is not None:
                thumbnail = cv2.resize(thumbnail, THUMBNAIL_SIZE)
                frame[y_offset:y_offset + THUMBNAIL_SIZE[1], 500:500 + THUMBNAIL_SIZE[0]] = thumbnail

        cv2.imshow("Select a Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video_selection_ui()