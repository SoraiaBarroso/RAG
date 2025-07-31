# Personal RAG Chatbot

This project implements a **Retrieval Augmented Generation (RAG)** chatbot designed to answer questions about my professional background, skills, and experience. It uses a **FastAPI** backend, and integrates **Google's Gemini API** for embeddings and language generation.

---

## ✨ Features

- **Retrieval Augmented Generation (RAG):** Answers questions by retrieving relevant information from a structured personal data source (JSON).
- **FastAPI Backend:** High-performance API with automatic validation and interactive documentation.
- **Google Gemini API Integration:** Uses `models/text-embedding-004` for embeddings and `gemini-2.0-flash` for generation.
- **ChromaDB Vector Store:** Stores and retrieves document embeddings locally.
- **Cloud Deployment:** Deployed in [Render](https://render.com).

---

## ⚙️ How It Works (Architecture)
### 🔧 Backend (FastAPI + RAG Logic)

1. **Data Loading:** Reads from `personal_data.json`.
2. **Document Processing:** Converts JSON into LangChain `Document` objects, then splits into chunks.
3. **Embedding & Indexing:**
   - Uses Google Generative AI Embeddings (`text-embedding-004`).
   - Stores vectors in local **ChromaDB** (created once on startup).
4. **RAG Chain Process:**
   - **Retrieval:** Embed user question and search for relevant chunks.
   - **Augmentation:** Supply retrieved chunks as LLM context.
   - **Generation:** Use Gemini (`gemini-2.0-flash`) to generate answers.
5. **API Endpoint:** `/ask` endpoint receives questions and returns generated answers.

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### ✅ Prerequisites

- Python 3.9+
- `pip`
- Git
- Google Cloud Project with Gemini API enabled and an API key

---

### 1. Clone the Repository

```bash
git clone https://github.com/SoraiaBarroso/RAG.git
```

### 2. Set Up Environment Variables
Create a .env file in your root directory:
```bash
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Your Personal Data
Create a file named data/personal_data.json with the following structure:
```json
{
  "name": "Jane Doe",
  "title": "Software Engineer",
  "work_experience": [
    {
      "company": "Tech Solutions Inc.",
      "position": "Senior Software Engineer",
      "duration": "Jan 2022 - Present",
      "responsibilities": [
        "Led development of scalable microservices.",
        "Mentored junior developers."
      ]
    },
    {
      "company": "Innovate Corp.",
      "position": "Software Developer",
      "duration": "Aug 2019 - Dec 2021",
      "responsibilities": [
        "Developed and maintained web applications.",
        "Collaborated with cross-functional teams."
      ]
    }
  ],
  "education": [
    {
      "degree": "M.Sc. Computer Science",
      "university": "State University",
      "year": "2019"
    }
  ],
  "skills": ["Python", "FastAPI", "React", "AWS", "Machine Learning"]
}
````

### 5. Run the Backend (FastAPI)
```bash
uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
```

## 6. Test with cURL

You can test the `/ask` endpoint directly using `curl` from your terminal.

```bash
curl -X POST "http://127.0.0.1:8000/ask/" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is your work experience?"}'
