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


from dynaconf import Dynaconf
import logging
import re
from pathlib import Path
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import typer

# Setup logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s -\
%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config. 
settings = Dynaconf(settings_files=["settings.toml"])

# Load typer.
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
def add_to_chromadb(
    local_chromadb_store: str = typer.Option(settings.system_setup.persist_dir,
                                             help="Path to local store."),
    collection_name: str = typer.Option(settings.system_setup.collection_name,
                                        help="A helpful name for collection."),
    log_folder: str = typer.Option(settings.system_setup.log_folder,
                                   help="Location of log folder."),
    max_workers: int = typer.Option(settings.system_setup.no_of_threads,
                                    help="No. of threads used to chunk files.")
):
    """Chunks all the logs and inserts them to a chromaDB."""

    logger.info("*** Creating chunks from logs based on pipeline stages...")
    chunks = chunk_all_logs(log_folder, max_workers)
    logger.info("*** Done")

    # Initialize ChromaDB client (local store)
    chroma_client = chromadb.PersistentClient(path=local_chromadb_store)

    # Create or get the collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small and fast

    logger.info(f"*** Embed and add chunks to \"{local_chromadb_store}\"")
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
