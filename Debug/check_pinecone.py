import os
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def check_transcripts():
    """Check stored transcripts in Pinecone."""
    response = index.describe_index_stats()
    print("📌 Pinecone Index Stats:", response)

if __name__ == "__main__":
    check_transcripts()
