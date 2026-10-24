# gemma-docgen

Multimodal developer CLI documentation assistant.

## Overview

`gemma-docgen` is a command-line tool designed to automate the creation of structured Markdown documentation for software projects. It scans project directories—including source code, configuration files, and architecture diagrams—and utilizes Google's Gemma 4 models via the Gemini API to generate high-quality documentation.

The tool is built with reliability in mind, featuring automatic payload shrinking to handle server overloads (HTTP 500/503/429) and automatic model fallback between Gemma 4 variants.

## Features

* **Multimodal Analysis**: Processes both text-based source files and image-based architecture diagrams.
* **Smart Scanning**: Prioritizes critical files (like `README.md`, `pyproject.toml`, or `package.json`) and entry points (like `main.py` or `app.py`).
* **Security First**: Automatically redacts sensitive information (API keys, tokens, AWS keys, etc.) and ignores actual `.env` files, only reading `.env.example` templates.
* **Robustness**: Implements automatic payload shrinking and model fallback to ensure successful documentation generation even under API constraints.
* **Intelligent Filtering**: Automatically ignores noise (e.g., `node_modules`, `venv`, lock files) and binary files.

## Tech Stack

* **Language**: Python 3
* **AI Model**: Google Gemma 4 (via Gemini API)
* **CLI Framework**: `typer`
* **Terminal UI**: `rich`
* **Image Processing**: `pillow`

## Prerequisites and Installation

### Prerequisites

* Python 3 installed on your system.
* A valid Google Gemini API Key.

### Installation

1. Clone the repository.
2. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Configuration

The tool requires a `GEMINI_API_KEY` to communicate with the Google API. This can be provided via a `.env` file in the project root or the script directory.

| Environment Variable | Description |
| :--- | :--- |
| `GEMINI_API_KEY` | Your Google Gemini API key. |

## Usage

Run the tool using the `main.py` entry point.

```bash
python main.py [OPTIONS] [PATH]
```

*(Note: Specific CLI arguments and flags are not fully detailed in the provided source snippet.)*

## Project Structure

```text
.
├── .env.example      # Template for environment variables
├── main.py           # Main CLI application logic
└── requirements.txt  # Project dependencies
```
