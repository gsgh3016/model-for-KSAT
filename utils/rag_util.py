import os

from dotenv import load_dotenv
from pinecone import Pinecone


def get_pinecone_index():
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX")

    if not api_key or not environment or not index_name:
        raise ValueError("Pinecone API_KEY, ENVIRONMENT, or INDEX name is missing from environment variables.")

    # Initialize Pinecone and connect to the existing index
    pc = Pinecone(api_key=api_key, environment=environment)
    pinecone_index = pc.Index(index_name)
    print(f"Connected to Pinecone index: {index_name}")

    return pinecone_index, index_name
