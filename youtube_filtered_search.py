import os
import googleapiclient.discovery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REFINED_SEARCH_QUERY_FILE = "refined_search_query.txt"
TOP_YOUTUBE_OPTIONS_FILE = "top_youtube_options.txt"


def search_youtube():
    """Searches YouTube using the refined query and filters results."""
    if not os.path.exists(REFINED_SEARCH_QUERY_FILE):
        print(" Refined search query file not found.")
        return

    with open(REFINED_SEARCH_QUERY_FILE, "r", encoding="utf-8") as file:
        search_query = file.read().strip()

    if not search_query:
        print("️ No search query found in refined_search_query.txt.")
        return

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    request = youtube.search().list(
        q=search_query,
        part="snippet",
        type="video",
        videoDuration="medium",  # Ensures videos are between 4-20 mins (closest to 3-15 mins filter)
        maxResults=10,
        relevanceLanguage="en",  # Prioritizes English videos
        videoEmbeddable="true"
    )

    response = request.execute()
    filtered_videos = []

    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        video_title = item["snippet"]["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Get video details (duration & language filtering)
        video_details = youtube.videos().list(
            part="contentDetails,statistics",
            id=video_id
        ).execute()

        if "items" in video_details and len(video_details["items"]) > 0:
            details = video_details["items"][0]
            duration = details["contentDetails"]["duration"]

            # Convert ISO 8601 duration to seconds
            import isodate
            duration_seconds = isodate.parse_duration(duration).total_seconds()

            if 180 <= duration_seconds <= 900:  # Between 3 min and 15 min
                filtered_videos.append(video_url)

        if len(filtered_videos) == 3:
            break  # Stop once we have 3 valid results

    # Save filtered video URLs to top_youtube_options.txt
    with open(TOP_YOUTUBE_OPTIONS_FILE, "w", encoding="utf-8") as file:
        for url in filtered_videos:
            file.write(url + "\n")

    print(" Top 3 YouTube video links saved to top_youtube_options.txt")


if __name__ == "__main__":
    search_youtube()
