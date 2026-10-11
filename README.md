# Book Q&A Chatbot (RAG with Streamlit)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the content of a PDF. It runs completely locally: no API keys and no cloud services are needed.

**I built this project myself** as part of my university studies (Semester One).

## How it works

1. **Load and split:** `Data.pdf` is loaded and cut into small chunks (`VectorFormation.ipynb`).
2. **Embed and store:** each chunk is turned into a vector with the `BAAI/bge-base-en-v1.5` embedding model and saved in a Chroma vector database (`Client/vectorDb`).
3. **Retrieve:** when you ask a question, two retrievers (MMR and similarity search) find the most relevant chunks. Their results are combined and reordered.
4. **Generate:** a local LLM (WizardCoder Python 7B, GGUF, run through `llama-cpp-python`) writes the answer using only the retrieved context.
5. **Chat:** a Streamlit app provides the chat interface and remembers the last two exchanges.

## Tech stack

- Python
- Streamlit
- LangChain (`langchain-classic`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`)
- ChromaDB
- Sentence Transformers (`BAAI/bge-base-en-v1.5`)
- llama.cpp via `llama-cpp-python`

## Project structure

```
.
├── Client/
│   ├── App.py                  # Streamlit chat app
│   ├── VectorFormation.ipynb   # Builds the vector database from the PDF
│   ├── Data.pdf                # Source document
│   ├── vectorDb/               # Chroma database (created by the notebook)
│   └── models/                 # Put the .gguf model file here (not in Git)
└── README.md
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the dependencies

```bash
pip install --upgrade pip
pip install streamlit langchain-classic langchain-community langchain-chroma \
            langchain-huggingface langchain-text-splitters sentence-transformers \
            pypdf llama-cpp-python jupyterlab ipykernel
```

### 3. Download the language model (about 4 GB)

The model file is too large for GitHub, so download it yourself:

```bash
mkdir -p Client/models
pip install -U huggingface_hub
hf download TheBloke/WizardCoder-Python-7B-V1.0-GGUF \
    wizardcoder-python-7b-v1.0.Q4_K_S.gguf --local-dir Client/models
```

### 4. Build the vector database

Skip this step if `Client/vectorDb` already exists.

```bash
jupyter lab
```

Open `Client/VectorFormation.ipynb` and choose **Run → Run All Cells**.

### 5. Start the app

Run this from the **project root** (the paths in `App.py` depend on it):

```bash
streamlit run Client/App.py
```

Then open the address shown in the terminal (usually `http://localhost:8501`).

## Tips

- The first start takes a minute or two while the models load into memory.
- A 7B model on a laptop answers slowly. That is normal.
- If the app freezes or runs out of memory, lower `n_ctx` to `2048` in `App.py`.
- The vector database must be built with the same embedding model that the app uses.

## Need help?

If you want help running this project, feel free to contact me:

- **Name:** Abdullah Farooqui
- **LinkedIn:** Muhammad Abdullah Farooqui, Vienna, Austria

## License

You are free to use and learn from this project. Please give credit if you build on it.
