![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/github/license/mujasoft/git_log_analyser)
![Status](https://img.shields.io/badge/status-WIP-orange)
![Demo Available](https://img.shields.io/badge/demo-available-green)


# Jenkins Log Analyser

A tool that uses AI to intelligently analyze Jenkins logs by segmenting, embedding, and answering natural language questions.

CI/CD pipelines generate massive amounts of logs — much of it repetitive, noisy, and hard to digest. This tool streamlines log analysis by using embeddings and a local LLM to summarize pipeline behavior, highlight failure patterns, and answer questions with context-aware insight.

## Why This Exists

- Configure once — no need to frequently modify code
- Easy to run, CI/CD friendly, and CLI-based
- Each component is Unix-like: it does one thing well

## Demo
[Watch Demo (3m)](./demo.mov)


## Technical Summary

This tool reads Jenkins logs, segments them into meaningful pipeline stages, embeds them with a transformer model, and stores them in a local ChromaDB vector database. It then sends queries to a local LLM (e.g., Mistral) via Ollama to answer questions based on log content.

## Tool Overview

The tool is composed of two scripts:

1. move_logs_to_chromadb.py  
   Segments logs, embeds them with SentenceTransformers, and stores them in ChromaDB.  
   Driven via CLI (Typer) or config (settings.toml).

2. analyse_logs.py  
   Reads questions from the same config file and sends them to the local LLM for answers.

## Configuration (settings.toml)

```toml
[system_setup]
persist_dir = "./chroma_store"
collection_name = "jenkins_logs"
ollama_url = "http://localhost:11434/api/generate"
model_name = "mistral"
n_results = 3
no_of_threads = 4
log_folder = "data"

[questions]
question1 = "What is the summary of the results so far?"
question2 = "How often did the checkout phase fail?"
question3 = "Can you tabulate how often the different phases pass or fail?"
question4 = "What seems to be the most common error?"
```

You can modify, add, or remove questions in the [questions] section without touching the code.

## Usage

1. Start your local LLM

```bash
ollama run mistral
```

2. Update your settings

```bash
vim settings.toml
```

3. Run the full pipeline

```bash
./run.sh
```

This will:
- Chunk and embed your logs
- Ask questions and print intelligent answers based on the logs

## Example Output

```text
Q.: How often did the checkout phase fail?
>>ANS: The Checkout phase failed three times due to an "Unknown error exit code: 1".

Q.: Can you tabulate how often the different phases pass or fail?
>>ANS:
| Run | Phase | Result |
|-----|-------|--------|
| 1   | Test  | Passed |
| 2   | Test  | Passed |
| 3   | Test  | Passed |

Q.: What seems to be the most common error?
>>ANS: An "Unknown error" in the Checkout stage with exit code 1 appears frequently.
```

## Requirements

- Python 3.8+
- chromadb
- dynaconf
- typer
- requests
- sentence-transformers
- Ollama (optional, for local LLM)

## License

MIT License – see [LICENSE](./LICENSE)

## Author

Mujaheed Khan  
DevOps | Python | Automation | CI/CD  
GitHub: https://github.com/mujasoft
