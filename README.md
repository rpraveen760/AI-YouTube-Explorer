# AI YouTube Explorer: Computer Vision-Powered Handwriting & Voice-Driven Q&A

## 📌 Overview
**AI YouTube Explorer** is a cutting-edge project that utilizes **Computer Vision, Speech Recognition, and AI-powered Q&A** to interactively search YouTube, transcribe videos, and answer user queries. The system allows users to perform YouTube searches via **handwriting recognition** or **voice commands**, retrieve video transcriptions, and interact with an AI chatbot for insights.

## 🚀 Features
- **Handwriting Recognition** ✍️: Use your finger or stylus to write a query, which gets converted into text.
- **Voice Commands** 🎤: Speak your query, and the system retrieves the answer from the transcribed video via pinecone 
- **YouTube API Integration** 📺: Searches and filters YouTube videos based on duration and relevance.
- **Gesture-Based Video Selection** ✋: Uses hand-tracking to choose and play a video.
- **Video Transcription** 📝: Retrieves subtitles from selected YouTube videos.
- **AI-Powered Q&A** 🤖: Uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions based on the video content.

## 🏗️ Tech Stack
- **Computer Vision**: OpenCV, MediaPipe (for hand tracking & gesture recognition)
- **Speech-to-Text**: Deepgram API (for transcribing voice input)
- **Natural Language Processing**: OpenAI API (for query refinement & Q&A)
- **Vector Database**: Pinecone (to store & retrieve transcribed video data)
- **Frontend & UI**: Gradio (for interactive AI-driven Q&A)
- **Backend**: Python, Google YouTube API (for video search & metadata retrieval)

## 🛠️ Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/AI-YouTube-Explorer.git
cd AI-YouTube-Explorer
```

### 2️⃣ Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On MacOS/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure API Keys
Create a `.env` file in the root directory and add:
```plaintext
YOUTUBE_API_KEY=your_youtube_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name
```

## ▶️ Usage
Run the main controller script to launch the pipeline:
```bash
python main_controller.py
```
### Workflow:
1. **Handwriting or Voice Input** → User writes or speaks a YouTube query.
2. **Query Refinement** → AI refines the search query.
3. **YouTube Search** → Retrieves top 3 relevant videos.
4. **Gesture-Based Selection** → User selects a video using hand gestures.
5. **Video Transcription & Storage** → Retrieves subtitles and stores them in Pinecone.
6. **AI-Powered Q&A** → Users can ask questions based on the video content.

## 📌 File Structure
```
AI-YouTube-Explorer/
│── app.py                      # Handwriting recognition
│── deepgram_speech_to_text.py   # Voice transcription
│── guardrails_filter.py         # Query refinement
│── youtube_filtered_search.py   # YouTube search
│── youtube_video_selector.py    # Gesture-based selection
│── dynamic_youtube_transcriber.py # Video transcription
│── main_controller.py           # Runs all components
│── .env                         # API keys (ignored in Git)
│── .gitignore                   # Files to exclude from Git
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
```


