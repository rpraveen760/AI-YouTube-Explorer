import os
import hashlib
import pinecone
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])
        return full_text
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None


def generate_unique_id(video_id):
    return hashlib.sha256(video_id.encode()).hexdigest()


def store_transcript(video_id):
    transcript_text = get_transcript(video_id)

    if transcript_text:
        unique_id = generate_unique_id(video_id)

        # Safely clear existing records in Pinecone
        existing_namespaces = index.describe_index_stats().get("namespaces", {})
        if PINECONE_INDEX_NAME in existing_namespaces:
            index.delete(delete_all=True)
            print("Cleared old transcripts from Pinecone.")
        else:
            print("Namespace not found, skipping deletion.")

        embedding = OpenAIEmbeddings().embed_query(transcript_text)

        print("Storing new transcript in Pinecone...")
        index.upsert([
            (
                unique_id,
                embedding,
                {
                    "page_content": transcript_text,
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
            )
        ])

        print(f"New transcript stored for video: https://www.youtube.com/watch?v={video_id}")
    else:
        print("No transcript available for this video.")


if __name__ == "__main__":
    # Automatically read the URL from file
    try:
        with open("video_to_be_transcribed.txt", "r") as file:
            youtube_url = file.read().strip()

        print(f"Using Video URL: {youtube_url}")

        if "youtube.com/watch?v=" in youtube_url or "youtu.be/" in youtube_url:
            video_id = youtube_url.split("v=")[-1].split("&")[0] if "youtube.com" in youtube_url else youtube_url.split("/")[-1]
            store_transcript(video_id)
            print("Video transcription and storage complete. You can now use the Q&A system!")
        else:
            print("Invalid YouTube URL in file.")

    except FileNotFoundError:
        print("File 'video_to_be_transcribed.txt' not found.")