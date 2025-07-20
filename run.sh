#!/bin/bash

# MIT License
# Copyright (c) 2025 Mujaheed Khan

# This script chunks Jenkins logs into ChromaDB and analyzes them using a local LLM.

# Color formatting
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function help() {
  echo -e "${YELLOW}Jenkins Log Analyser - CLI Tool${NC}"
  echo ""
  echo "Usage:"
  echo "  $0             Run full pipeline: chunk logs and analyze them"
  echo "  $0 --help      Show this help message"
  echo ""
  echo "Description:"
  echo "  - This tool reads Jenkins logs from a folder (configured in settings.toml),"
  echo "    splits them into meaningful stages, embeds them, and stores them in ChromaDB."
  echo "  - Then it sends questions (defined in settings.toml) to a local LLM like Mistral"
  echo "    via Ollama and prints the responses."
  echo ""
  echo "Author: Mujaheed Khan"
  echo "License: MIT (c) 2025"
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  help
  exit 0
fi

# Run main pipeline
python3 move_logs_to_chromadb.py
python3 analyse_logs.py
