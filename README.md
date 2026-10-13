# 📄 RAG Document Bot

Upload any document and ask questions from it. The bot answers ONLY from your document — never makes things up.

## 🌐 Live Demo
👉 [https://annu-rag-document-bot.streamlit.app](https://annu-rag-document-bot.streamlit.app)

## ✨ Features
- Upload any .txt document
- Ask questions in natural language
- Answers grounded only in your document
- Says "I don't know" for off-topic questions
- Chat history preserved during session

## 🛠️ Tech Stack
- Python
- LangChain + FAISS (vector database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq API
- Streamlit

## 🔬 How RAG Works
1. Document split into 500-character chunks
2. Each chunk converted to vector (384 numbers)
3. Question matched to closest chunks by meaning
4. Only relevant chunks sent to AI
5. AI answers from those chunks only

## 🚀 Run Locally
1. Clone this repo
2. Install: `pip install -r requirements.txt`
3. Create `.env`: `GROQ_API_KEY=your_key_here`
4. Run: `streamlit run rag_app.py`

## 👩‍💻 Built By
Anushka — AI Developer from Indore, India
