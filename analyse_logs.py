# MIT License

# Copyright (c) 2025 Mujaheed Khan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import chromadb
from chromadb.config import Settings
from dynaconf import Dynaconf
import requests
from sentence_transformers import SentenceTransformer

# Load settings from settings.toml
settings = Dynaconf(
    settings_files=["settings.toml"]
)

# Extract system setup configuration
persist_dir = settings.system_setup.persist_dir
collection_name = settings.system_setup.collection_name
ollama_url = settings.system_setup.ollama_url
model_name = settings.system_setup.model_name
n_results = settings.system_setup.n_results

# Initialize ChromaDB client and embedder
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_collection(name=collection_name)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def ask_question(query: str) -> str:
    """
    Send a question to the local LLM using embedded Jenkins log context.

    Args:
        query (str): The natural language question to answer.

    Returns:
        str: The response from the LLM.
    """
    # Embed the question
    query_embedding = embedder.encode(query).tolist()

    # Search Chroma DB for relevant log chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    retrieved_docs = results["documents"][0]
    contexts = "\n-----------\n".join(retrieved_docs)

    # Construct full prompt for the LLM
    full_prompt = f"""You are a world class expert at analyzing Jenkins CI \
logs. Use the logs below to answer the question.

Logs:
{contexts}

Question: {query}
"""

    # Send request to local LLM server
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post(ollama_url, json=payload)
    return response.json().get("response", "[No response]")

# CLI option was avoided on purpose as this tool is meant to be text driven.
# It is far too tedious to type our questions on comandline without some
# interactive output. The user is meant to write his/her/their questions
# in the settings.toml and run this script.


if __name__ == "__main__":
    # Loop through all configured questions (alphabetical order)
    for key, value in sorted(settings.questions.items()):
        print(f"Q.: {value}")
        answer = ask_question(value)
        print(f">>ANS: {answer}\n")
