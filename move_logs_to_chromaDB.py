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

import logging
import re
from pathlib import Path
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import typer

logging.basicConfig(level=logging.INFO, format='%(asctime)s -\
%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="A tool to process Jenkins log files, chunk them by pipeline stages, \
and embed them into a ChromaDB for fast retrieval."
)


def chunk_jenkins_log(filepath: str, marker_regex: str):
    """Given a file, it returns chunks based on stage.

    Args:
        filepath (str): location of file of interest.
        marker_regex (str): a regex string. Tailored to jenkinsfile atm.

    Returns:
        list[dictionary]: returns a list of dictionaries.
    """

    chunks = []
    current_chunk = []
    current_stage = "unknown"

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        match = re.search(marker_regex, line)
        if match:
            # Save the previous chunk
            if current_chunk:
                chunks.append({
                    "text": ''.join(current_chunk),
                    "stage": current_stage,
                    "source": filepath
                })
                current_chunk = []
            current_stage = match.group(1).strip()

        current_chunk.append(line)

    # Save the last chunk
    if current_chunk:
        chunks.append({
            "text": ''.join(current_chunk),
            "stage": current_stage,
            "source": filepath
        })

    return chunks


def chunk_all_logs(log_dir: str = "data/logs", max_workers: int = 4):
    """Goes through all logs in given folder and breaks them into chunks.

    This can be parallelized. The default number of threads is '4'.

    Args:
        log_dir (str, optional): Location of logs. Defaults to "data/logs".
        max_workers (int, optional): Number of parallel threads. Defaults to 4.

    Returns:
        list[dictionary]: returns all chunks in a list.
    """

    all_chunks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file in Path(log_dir).rglob("*.txt"):
            futures.append(executor.submit(chunk_jenkins_log, str(file),
                           r'\[Pipeline\] stage: (.+)'))

        for future in as_completed(futures):
            file_chunks = future.result()
            all_chunks.extend(file_chunks)

    return all_chunks


@app.command()
def add_to_chromadb(local_chromaDB_store: str = "./chroma_store",
                    collection_name: str = "jenksin_logs",
                    log_folder: str = "data/logs",
                    max_workers: int = 4):
    """Chunks all the logs and inserts them to a chromaDB.

    Arg(s):
        local_chromaDB_store (str, optional): name of local store.
                                              Defaults to "./chroma_store".
        collection_name (str, optional): A helpful name for collection.
                                         Defaults to "jenksin_logs".
        log_folder (str, optional): location of log folder.
                                    Defaults to "data/logs".
        max_workers (int, optional): Number of threads to take advantage
                                     of parallel threads. Defaults to 4.
    """

    logger.info("*** Creating chunks from logs based on pipeline stages...")
    chunks = chunk_all_logs(log_folder, max_workers)
    logger.info("*** Done")

    # Initialize ChromaDB client (local store)
    chroma_client = chromadb.PersistentClient(path=local_chromaDB_store)

    # Create or get the collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small and fast

    logger.info(f"*** Embed and add chunks to \"{local_chromaDB_store}\"")
    # Embed and add to ChromaDB
    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk["text"]).tolist()
        collection.add(
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{
                "stage": chunk["stage"],
                "source": chunk["source"]
            }],
            ids=[f"log_chunk_{idx}"]
        )
    logger.info("*** Done")


if __name__ == "__main__":
    app()
