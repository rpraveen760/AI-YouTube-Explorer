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


def clear_index():
    """Deletes all stored documents from Pinecone."""
    print("⚠️ WARNING: This will delete all stored transcripts in Pinecone!")
    confirm = input("Type 'YES' to confirm: ")

    if confirm == "YES":
        index.delete(delete_all=True)
        print("✅ All vectors deleted from Pinecone!")
    else:
        print("❌ Deletion canceled.")


if __name__ == "__main__":
    clear_index()
