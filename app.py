import streamlit as st
from pathlib import Path
from langchain.memory import ConversationBufferMemory
import time
from backend import cria_chain_conversa, FOLDER_FILES

# folder_files = Path(__file__).parent / "files"
FOLDER_FILES.mkdir(exist_ok=True)
folder_files = FOLDER_FILES



def chat_app():
    st.header("Bem vindo ao GPT", divider=True)

    if not "chain" in st.session_state:
        st.error("Faça o upload de pdfs para comecar")
        st.stop()

    chain = st.session_state["chain"]
    memory = chain.memory
    mensagens = memory.load_memory_variables({})["chat_history"]

    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    nova_mensagem = st.chat_input("Converse com seus documentos")
    if nova_mensagem:
        chat = container.chat_message("human")
        chat.markdown(nova_mensagem)
        chat = container.chat_message("ai")
        chat.markdown("Gerando Resposta")
        chain.invoke({"question": nova_mensagem})
        st.rerun()
        
def save_uploaded_files(uploaded_files, folder):
    """Salva aquivos enviados em uma pasta especifica"""
    for file in folder.glob("*.pdf"):
        file.unlink()
    # Salvar arquivos enviados
    for file in uploaded_files:
        (folder / file.name).write_bytes(file.read())



def main():
    with st.sidebar:
        st.header("Upload PDFS")
        uploaded_pdfs = st.file_uploader("Adicione arquivo pdfs",
                                       type="pdf",
                                       accept_multiple_files=True)
        if uploaded_pdfs:
            save_uploaded_files(uploaded_pdfs, folder_files)
            st.success(f"{len(uploaded_pdfs)} arquivo(s) salvo(s) com sucesso!")

        label_botao = "Inicializar Chat"
        if "chain" in st.session_state:
            label_botao = "Atualizar Chat"

        if st.button(label_botao, use_container_width=True):
            if (len(list(folder_files.glob("*.pdf")))) == 0:
                st.error("Adicione arquivo(s) pdf para inicializar o chat")
            else:
                st.success("Inicializando o Chat ...")
                cria_chain_conversa()
                st.rerun()


        
if __name__ == "__main__":
    main()
    chat_app()
