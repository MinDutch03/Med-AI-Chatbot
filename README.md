# LLM Medical Chatbot

A Retrieval-Augmented Generation (RAG) based medical chatbot leveraging Large Language Models (LLMs) and a vector database for accurate, context-aware medical question answering.

---

## 🚀 Project Overview

- **Domain:** Medical Question Answering
- **Core Technologies:** Python, React, Qdrant, Mistral-7B (or other LLMs)
- **Features:**
  - Retrieval-Augmented Generation (RAG) pipeline
  - Semantic search over medical knowledge bases (MedQuAD, PubMed, etc.)
  - Modern frontend for chat-based interaction
  - Scalable backend with RESTful APIs

---

## 📁 Repository Structure

- `frontend/` — React-based web UI
- `scripts/` — Data processing, retrieval, and backend scripts
- `models/` — (Large files, download separately)
- `data/` — (Large files, download separately)
- `qdrant_data/` — (Vector DB files, download separately)

---

## ⚡️ Quickstart

### 1. Clone the Repository
```bash
git clone 
cd LLM_Medical-Chatbot
```

### 2. Install Dependencies
- **Backend:**
  - Python 3.8+
  - Install required packages (see scripts/requirements.txt or your script headers)
- **Frontend:**
  - Node.js 18+
  - `cd frontend && npm install`

### 3. Download Large Files
This repository does **not** include large files (models, data, vector DB) due to GitHub size limits.


> **Instructions:**
> 1. Download the files from the scripts folder.
> 2. Place them in the corresponding folders (`models/`, `data/`, `qdrant_data/`).

### 4. Run the Backend
```bash
# Example (adjust as needed)
python scripts/backend_rag.py
```

### 5. Run the Frontend
```bash
cd frontend
npm run dev
```

---

## 🧩 Key Features
- **Retrieval-Augmented Generation:** Combines LLMs with semantic search for accurate, context-rich answers.
- **Medical Knowledge Bases:** Utilizes MedQuAD, PubMed, and other trusted sources.
- **Scalable Architecture:** Modular backend and frontend for easy extension.

---

## 🗂️ Data Sources
- [MedQuAD](https://github.com/abachaa/MedQuAD)
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)

---

## 📝 Customization
- Swap in different LLMs or embedding models as needed.
- Add new data sources by updating the scripts in `scripts/`.

---

## 🤝 Contributing
Pull requests and issues are welcome!

---

## 📄 License
See `MedQuAD-master/LICENSE.txt` and project root for licensing details.

---

## 📬 Contact
For questions or collaboration, open an issue or contact [Duc](https://github.com/MinDutch03). 