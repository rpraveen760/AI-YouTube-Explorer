import subprocess
import sys
import time
import runpy

def run_script(script_name):
    """Runs a Python script with the virtual environment's Python interpreter."""
    venv_python = sys.executable
    print(f"\nRunning {script_name}...")
    result = subprocess.run(
        [venv_python, script_name],
        capture_output=True,
        text=True,
        encoding='utf-8'  # Explicitly use UTF-8 encoding
    )
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script_name}: {result.stderr}")

def main():
    print("\nStarting AI-Powered YouTube Pipeline \n")
    time.sleep(1)

    scripts = [
        "app.py",
        "guardrails_filter.py",
        "youtube_filtered_search.py",
        "youtube_video_selector.py",
        "dynamic_youtube_transcriber.py",
    ]

    for script in scripts:
        run_script(script)

    # Run Gradio app directly using runpy for proper event loop handling
    print("\nLaunching Gradio Q&A Interface ")
    runpy.run_path("deepgram_speech_to_text.py", run_name="__main__")

    print("\nAI-Powered YouTube Pipeline Completed Successfully! 🎉")
    print("Now, you can ask questions about the video content using your voice.")

if __name__ == "__main__":
    main()
