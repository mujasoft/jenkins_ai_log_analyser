# Copyright 2025 Mujaheed Khan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    full_prompt = f"""You are a world class expert at analyzing Jenkins CI logs.
Use the logs below to answer the question.

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


if __name__ == "__main__":
    # Loop through all configured questions (alphabetical order)
    for key, value in sorted(settings.questions.items()):
        print(f"Q.: {value}")
        answer = ask_question(value)
        print(f">>ANS: {answer}\n")
