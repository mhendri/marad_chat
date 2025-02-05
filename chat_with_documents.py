import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


st.set_page_config(page_title="MARAD Chat", page_icon="📄", initial_sidebar_state="collapsed")
st.header('MARAD Chat')
st.write('Ask me questions about maritime policy, fuel standards, shipping, decarbonization, and more!')


class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()

    def save_file(self, file):
        folder = 'corpus'
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner('Analyzing documents...')
    def setup_qa_chain(self, corpus):
        print("called")
        vectordb = utils.load_or_preload_documents(corpus)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 3, 'fetch_k': 5}
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",  # Explicitly set output_key to avoid multiple key errors
            return_messages=True
        )

        # Custom prompt to ensure no query repetition
        custom_prompt = PromptTemplate.from_template(
    """
    You are a helpful AI assistant. Answer the user's question concisely and accurately.
    
    Do **not** repeat or rephrase the user's question in your response. Provide direct, relevant, and well-structured answers.

    If the context is not relevant, state that you don't have enough information.

    --- 
    Context:
    {context}
    
    Chat History:
    {chat_history}

    ---
    User's Question: {question}
    AI Response:
    """
)

        # Use ConversationalRetrievalChain to support chat history
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask me anything!")

        # Initialize chat history in session state if not present
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Check if QA chain is already in session state
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = self.setup_qa_chain('corpus')

        qa_chain = st.session_state.qa_chain  # Retrieve persisted QA chain

        if user_query:
            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())

                # Pass chat history correctly
                result = qa_chain.invoke(
                    {"question": user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]  # ConversationalRetrievalChain uses "answer"

                # Store conversation history in session state
                st.session_state["chat_history"].append({"role": "user", "content": user_query})
                st.session_state["chat_history"].append({"role": "assistant", "content": response})

                # Display the response
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})
                utils.print_qa(CustomDocChatbot, user_query, response)

                # Show references
                for idx, doc in enumerate(result.get('source_documents', []), 1):
                    filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page_num = doc.metadata.get('page', 'N/A')
                    ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)


if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()
