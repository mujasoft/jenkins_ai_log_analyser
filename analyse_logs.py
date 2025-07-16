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

settings = Dynaconf(
    settings_files=["settings.toml"]
)

# Extract chromaDB settings.
persist_dir = settings.system_setup.persist_dir
collection_name = settings.system_setup.collection_name
ollama_url = settings.system_setup.ollama_url
model_name = settings.system_setup.model_name
n_results = settings.system_setup.n_results = 3

# Initialize chromaDB client.
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_collection(name=collection_name)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def ask_question(query: str):
    """Give a question to the local LLM.

    Args:
        query (str): Provide a question.
    """    """"""

    # Embed the question.
    query_embedding = embedder.encode(query).tolist()

    # Search Chroma DB.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    retrieved_docs = results["documents"][0]
    contexts = "\n-----------\n".join(retrieved_docs)

    # Create a proper prompt for the LLM.
    full_prompt = f"""You are a world class expert at analyzing Jenkins CI\
logs. Use the logs below to answer the question.

Logs:
{contexts}

Question: {query}
"""

    # Prepare a payload to send to the local LLM.
    payload = {
        "model": model_name,  # Using mistral.
        "prompt": full_prompt,
        "stream": False
    }

    response = requests.post(ollama_url, json=payload)
    return response.json().get("response", "[No response]")


if __name__ == "__main__":

    # Loop through all the questions and print out their answers.
    for key, value in settings.questions.items():

        answer = ask_question(value)
        print(f"{answer}\n")