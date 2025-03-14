import os
import asyncio
import sounddevice as sd
import numpy as np
import soundfile as sf
import gradio as gr
import subprocess
from deepgram import Deepgram
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import pinecone
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gradio")

# Load environment variables
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

recording = False  # Global flag for recording status


def start_recording():
    """Starts recording audio until stopped and updates UI status."""
    global recording, audio_data
    recording = True
    print(" Recording started...")
    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        if recording:
            audio_data.append(indata.copy())

    global stream
    stream = sd.InputStream(samplerate=44100, channels=1, dtype='int16', callback=callback)
    stream.start()
    return " Recording... Press Stop & Transcribe to finish."


def stop_recording():
    """Stops recording and saves the audio file."""
    global recording, stream
    recording = False
    stream.stop()
    stream.close()

    audio_array = np.concatenate(audio_data, axis=0)
    filename = "test_audio.wav"
    sf.write(filename, audio_array, 44100)
    print(" Recording saved as", filename)
    return filename


async def transcribe_audio(filename):
    """Sends the recorded WAV audio to Deepgram for transcription and saves it to a file."""
    dg = Deepgram(DEEPGRAM_API_KEY)

    try:
        with open(filename, "rb") as audio_file:
            buffer = audio_file.read()

        response = await dg.transcription.prerecorded(
            {"buffer": buffer, "mimetype": "audio/wav"},
            {"model": "nova", "smart_format": True, "punctuate": True}
        )

        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        print(" Deepgram Response:", transcript)

        # Save transcript to a text file
        with open("audio_query.txt", "w", encoding="utf-8") as file:
            file.write(transcript)
        print(" Transcription saved to audio_query.txt")

        return transcript

    except Exception as e:
        print(f" Deepgram API Error: {e}")
        return "Error processing audio."


def process_audio(_):
    """Handles recording, stopping, transcription, and saving to a file in the Gradio UI."""
    filename = stop_recording()
    transcript = asyncio.run(transcribe_audio(filename))
    return transcript


def load_audio_query():
    """Reads saved query from file."""
    if os.path.exists("audio_query.txt"):
        with open("audio_query.txt", "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""


async def retrieve_summary(query):
    """Retrieve and stream a relevant summary only when a valid query is provided."""
    if not query.strip():
        yield "Please enter a question to generate a summary."
        return

    print(f"🔹 Received query: {query}")

    try:
        vector_store = PineconeVectorStore(index, OpenAIEmbeddings(), "page_content")
        docs = vector_store.similarity_search(query, k=5)
        keyword_filtered_docs = [doc for doc in docs if query.lower() in doc.page_content.lower()]

        if keyword_filtered_docs:
            best_transcript = keyword_filtered_docs[0].page_content
        else:
            best_transcript = docs[0].page_content

        print(" Final Selected Document for Summarization:")
        print(best_transcript[:500])

        response = ""
        for chunk in llm.stream(input=f"Summarize this transcript based on: {query}\n\n{best_transcript}"):
            response += chunk.content if hasattr(chunk, "content") else str(chunk)
            yield response

    except Exception as e:
        print(f" Error in retrieve_summary: {e}")
        yield f"Error: Could not retrieve summary - {e}"


# Gradio UI for Speech-to-Text & Summarization
gram_ui = gr.Blocks()
with gram_ui:
    gr.Markdown("# Deepgram Speech-to-Text for voice query")
    gr.Markdown("Click Start to begin recording and Stop to transcribe the audio.")

    with gr.Row():
        start_btn = gr.Button("Start Recording")
        stop_btn = gr.Button("Stop & Transcribe")

    status_text = gr.Textbox(label="Status", interactive=False)
    query_input = gr.Textbox(label="Enter your question", value=load_audio_query())
    submit_btn = gr.Button("Submit")
    summary_output = gr.Textbox(label="Summary", interactive=False)

    start_btn.click(fn=start_recording, inputs=[], outputs=status_text)
    stop_btn.click(fn=process_audio, inputs=[], outputs=query_input)
    submit_btn.click(fn=retrieve_summary, inputs=query_input, outputs=summary_output)

gram_ui.launch(share=True)