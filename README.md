# Bio-Medical MultiModal Llama 3 Web App

This project provides a Gradio-based interface for the [Bio-Medical MultiModal Llama-3 8B V1](https://huggingface.co/ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1) model. The chat UI resembles ChatGPT and supports image inputs and retrieval-augmented generation (RAG) for contextual memory.

## Features

- User authentication with simple subscription gating
- Admin panel for managing user accounts
- Chat interface styled like OpenAI ChatGPT
- Optional image upload for multimodal queries
- Retrieval-Augmented Generation using a FAISS index built from `data/knowledge_base.txt`
- Past interactions are inserted back into the RAG store to provide conversational memory

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the web app:

```bash
python app.py
```

Login with one of the users defined in `data/users.json` (default `admin`/`admin` or `demo`/`demo`). The admin account can toggle user subscriptions and create new users.

## Adding Knowledge

Extend the initial RAG data by editing `data/knowledge_base.txt`. Each line is treated as a separate document. New conversation turns are automatically appended to the in-memory vector store during runtime.

## Notes

Running large models like Llama-3 8B may require substantial system resources. Consider using GPU acceleration or a smaller model for improved performance.

