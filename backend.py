from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pypdf.errors import PdfReadError, PdfStreamError
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain
)


# =========================
# CONFIGURAÇÕES
# =========================

_ = load_dotenv(find_dotenv())

FOLDER_FILES = Path(__file__).parent / "files"
MODEL_NAME = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL = "text-embedding-3-small"


# =========================
# VALIDAÇÕES
# =========================

def valida_openai_key():

    if not os.getenv("OPENAI_API_KEY"):

        raise ValueError(
            "OPENAI_API_KEY não encontrada no ambiente."
        )


# =========================
# IMPORTAÇÃO DOS PDFs
# =========================

def importa_documentos():

    documentos = []

    st.write(f"Pasta de PDFs: {FOLDER_FILES}")

    if not FOLDER_FILES.exists():

        st.error("Pasta files não existe.")

        return documentos

    arquivos = list(FOLDER_FILES.glob("*"))

    st.write(f"Arquivos encontrados: {arquivos}")

    if not arquivos:

        st.error("Nenhum arquivo encontrado na pasta files.")

        return documentos

    for arquivo in arquivos:

        st.write(f"Processando: {arquivo.name}")

        if arquivo.suffix.lower() == ".pdf":

            try:

                loader = PyMuPDFLoader(str(arquivo))

                docs = loader.load()

                st.success(
                    f"{arquivo.name} carregado "
                    f"({len(docs)} páginas)"
                )

                documentos.extend(docs)

            except Exception as e:

                st.error(
                    f"Erro ao carregar {arquivo.name}"
                )

                st.exception(e)

    st.write(
        f"Total documentos carregados: "
        f"{len(documentos)}"
    )

    return documentos

# =========================
# CHUNKING
# =========================

def split_documentos(documentos):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documentos)

    for i, doc in enumerate(chunks):

        source = doc.metadata.get("source", "")

        doc.metadata["source"] = (
            Path(source).name
        )

        doc.metadata["chunk_id"] = i

    print(f"Total de chunks: {len(chunks)}")

    return chunks


# =========================
# VECTOR STORE
# =========================

def cria_vector_store(documentos):

    if not documentos:

        raise ValueError(
            "Lista de documentos vazia."
        )

    print(
        f"Criando embeddings para "
        f"{len(documentos)} chunks..."
    )

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )

    print("FAISS criado com sucesso.")

    return vector_store


# =========================
# MEMORY
# =========================

def cria_memoria():

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return memory


# =========================
# CHAT MODEL
# =========================

def cria_chat_model():

    chat = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0
    )

    return chat


# =========================
# CHAIN CONVERSACIONAL
# =========================

@st.cache_resource
def cria_chain_conversa():

    valida_openai_key()

    documentos = importa_documentos()

    chunks = split_documentos(documentos)

    vector_store = cria_vector_store(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    memory = cria_memoria()

    chat_model = cria_chat_model()

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )

    print("Chain criada com sucesso.")

    st.session_state["chain"] = chat_chain

    return chat_chain
