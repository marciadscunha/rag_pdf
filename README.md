# 📄 RAG PDF com LangChain

Sistema de conversação com PDFs utilizando:

- LangChain
- RAG (Retrieval-Augmented Generation)
- OpenAI
- FAISS
- Streamlit

O projeto permite carregar documentos PDF e conversar com eles utilizando IA generativa.

---

# Funcionalidades

✅ Upload e leitura de PDFs  
✅ Chunking automático de documentos  
✅ Embeddings com OpenAI  
✅ Busca vetorial com FAISS  
✅ Conversação contextual  
✅ Memória de conversa  
✅ Interface web com Streamlit  
✅ Arquitetura modular pronta para evolução

---

# Arquitetura RAG

```text
PDFs
 ↓
Loader (PyMuPDF)
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS Vector Store
 ↓
Retriever
 ↓
LLM (OpenAI)
 ↓
Resposta contextual
