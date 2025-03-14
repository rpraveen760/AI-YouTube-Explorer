import os
import pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Query for testing
query = "What is the video about?"
embedding = OpenAIEmbeddings().embed_query(query)

# 🔍 Fetch stored vectors similar to the query
response = index.query(vector=embedding, top_k=5, include_metadata=True)

print("\n🔍 Stored Documents in Pinecone (Relevance Order):")
for match in response["matches"]:
    print(f"📌 Score: {match['score']}")
    print(f"📝 Transcript Snippet: {match['metadata'].get('page_content', '')[:300]}...\n")  # Print first 300 chars
