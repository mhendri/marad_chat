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
st.write('Ask me questions about maritime policy, fuel standards, shipping, decorbonization and more!')


class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()
        # self.vectordb = utils.load_or_preload_documents()

    def save_file(self, file):
        folder = 'corpus'
        if not os.path.exists(folder):
            os.makedirs(folder)

        # this has no name because it is not the ST file upload object
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, corpus):
        # Load documents
        print("called")
        vectordb = utils.load_or_preload_documents(corpus)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 3, 'fetch_k': 5}
        )

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # custom_prompt = PromptTemplate.from_template(
        #     """
        #     You are a helpful AI assistant. Answer the user’s question concisely and accurately. 
        #     Do not rephrase or repeat the question in your response. Provide direct answers.

        #     Context:
        #     {context}

        #     Chat History:
        #     {chat_history}

        #     User: {question}
        #     AI:
        #     """
        # )

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False,
            # combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask me anything!")

        # Check if QA chain is already in session state
        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = self.setup_qa_chain('corpus')

        qa_chain = st.session_state.qa_chain  # Retrieve persisted QA chain

        if user_query:
            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                result = qa_chain.invoke(
                    {"question": user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["answer"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})
                utils.print_qa(CustomDocChatbot, user_query, response)

                # Show references
                for idx, doc in enumerate(result['source_documents'], 1):
                    filename = os.path.basename(doc.metadata['source'])
                    page_num = doc.metadata['page']
                    ref_title = f":blue[Reference {idx}: *{filename} - page.{page_num}*]"
                    with st.popover(ref_title):
                        st.caption(doc.page_content)



if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()



## have system stop repeating query during answer - it started once main function was updated