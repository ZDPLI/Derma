# Bio-Medical MultiModal Llama 3 Web App

This project provides a simple Gradio-based web interface for interacting with the [Bio-Medical MultiModal Llama-3 8B V1](https://huggingface.co/ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1) model. The interface mimics the OpenAI chat style and supports image inputs and retrieval-augmented generation (RAG) for contextual memory.

## Features

- Chat interface styled similarly to OpenAI ChatGPT.
- Optional image upload for multimodal queries.
- Retrieval-Augmented Generation using a simple FAISS index built from `data/knowledge_base.txt`.
- Past interactions are inserted back into the RAG index to provide conversational memory.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the web app:

```bash
python app.py
```

The application will load the model (requires internet access on first run) and start a Gradio server. Open the provided URL in your browser to interact with the chatbot.

## Adding Knowledge

You can extend the initial RAG data by editing `data/knowledge_base.txt`. Each line is treated as a separate document. New conversation turns are automatically appended to the in-memory vector store during runtime.

## Notes

Running large models like Llama-3 8B may require substantial system resources. Consider using GPU acceleration or smaller quantized versions for improved performance.
