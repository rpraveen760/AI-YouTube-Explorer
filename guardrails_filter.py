import re
import openai
import os
from dotenv import load_dotenv

load_dotenv()

def refine_search_query(user_input):
    """Intelligently refine the query by identifying its category and expanding it accordingly."""
    cleaned_input = re.sub(r"[^a-zA-Z0-9\s]", "", user_input).strip()

    if not cleaned_input:
        return None

    prompt = f"""
    You are an AI designed to classify YouTube search queries into specific genres.

    Given the search query: "{cleaned_input}"

    Task:
    - If it is a well-known **acronym** (e.g., NLP, AI, CNN), expand it with its relevant field.
      - Example: "NLP" → "NLP in Machine Learning"
      - Example: "AI" → "Artificial Intelligence in Computer Science"

    - If it is the **name of a movie**, classify it as a movie.
      - Example: "Inception" → "Inception in Movies"
      - Example: "Interstellar" → "Interstellar in Movies"

    - If it is a **broad topic**, add its relevant category.
      - Example: "Quantum Physics" → "Quantum Physics in Science"
      - Example: "Stock Market" → "Stock Market in Finance"
      - Example: "History of Rome" → "History of Rome in History"

    - If the query is already well-defined, DO NOT ALTER IT.

    **Rules:**
    - Keep refinements minimal and logical.
    - Preserve original meaning at all costs.
    - DO NOT return anything except the refined query.

    Rewritten query:
    """

    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=30,
    )

    refined_query = response.choices[0].message.content.strip()
    return refined_query

# Read the input from search_query.txt and process
with open("search_query.txt", "r", encoding="utf-8") as file:
    original_query = file.read().strip()

refined_query = refine_search_query(original_query)

# Save the refined query into refined_search_query.txt
with open("refined_search_query.txt", "w", encoding="utf-8") as file:
    file.write(refined_query)

print(f" Refined Query Saved: \"{refined_query}\"")
