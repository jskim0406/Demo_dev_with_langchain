import re
import openai
import argparse
import streamlit as st

from io import BytesIO
from pypdf import PdfReader
from langchain.docstore.document import Document

from langchain import LLMChain, OpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent # Key part of the app

# inspired, reference, credit: https://medium.com/@avra42/how-to-build-a-personalized-pdf-chat-bot-with-conversational-memory-965280c160f8


def text_custom(font_size, text):
    '''
    font_size := ['b', 'm', 's']
    '''
    result=f'<p class="{font_size}-font">{text}</p>'
    return result


@st.cache_data
def _chunks_from_pdf(file:BytesIO):
    # 1. pdf에서 texts 추출(texts: [text, text, text, ...])
    pdf = PdfReader(file)
    texts = [p.extract_text() for p in pdf.pages]
    pages = [Document(page_content=text) for text in texts]
    
    # Add page numbers as metadata
    for i, page in enumerate(pages):
        page.metadata['page'] = i+1

    # 2. Chunking & metadata 추가(chunk index 정보)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=['\n\n', '\n', '.', '!', '?', ',', ' ', ''],
        chunk_overlap=200
    )
    chunks=text_splitter.split_documents(pages)
    return chunks


@st.cache_data
def _embed(embeddeing_model, api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, model=embeddeing_model)

    with st.spinner("Now.. we are indexing the data.. Please wait.."):
        index = FAISS.from_documents(chunks, embeddings)

    st.success("It's Done!", icon='✅')
    return index


def _init_qa_chain(index, api_key, chain_type='map_reduce'):

    qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=api_key),
                chain_type = chain_type,
                retriever=index.as_retriever(),
            )
    return qa


def main():
    # Basic UI setup
    st.set_page_config(
        page_title="Hello, Welcome to Question Answering on Your own PDF file",
        layout="wide",  # {wide, centered}
    )
    # reference
    ## https://discuss.streamlit.io/t/change-input-text-font-size/29959/4
    ## https://discuss.streamlit.io/t/change-font-size-in-st-write/7606/2
    st.markdown("""<style>.b-font {font-size:25px !important;}</style>""", unsafe_allow_html=True)    
    st.markdown("""<style>.m-font {font-size:20px !important;}</style>""" , unsafe_allow_html=True)    
    st.markdown("""<style>.s-font {font-size:15px !important;}</style>""" , unsafe_allow_html=True)    
    tabs_font_css = """<style>div[class*="stTextInput"] label {font-size: 15px;color: white;}</style>"""
    st.write(tabs_font_css, unsafe_allow_html=True)

    st.title("QA Bot who has memory 😲 on your own PDF file.")
    t = "Long PDF file ... hard to read it all, right? Just post it here and ask questions. 😋"
    st.markdown(text_custom('m', t), unsafe_allow_html=True)

    api_key = st.text_input(
        "Enter Open AI Key.", placeholder = "sk-...", type="password")

    loaded_file = st.file_uploader("**Please upload your PDF file here.**", 
                                   type = ["pdf"])
    
    with st.sidebar:
        embeddeing_model = st.selectbox(
            label='Embedding Model',
            options=['text-embedding-ada-002']
        )

        chat_model = st.selectbox(
            label='Chat Model',
            options=["gpt-3.5-turbo"]
        )

        chain = st.radio(
            label='Chain type',
            options=['stuff',
                    'map_reduce']
        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.7,
        )

        st.markdown(
            """
            **Blog post:** \n
            [*Blog title*](URL)

            **Code:** \n
            [*Github*](https://github.com/jskim0406/SimpleRec_w_langchain.git)
            """
        )

    if loaded_file:
        fname = loaded_file.name

        global chunks
        chunks=_chunks_from_pdf(loaded_file)
        
        index = _embed(embeddeing_model, api_key)

        ret_qa_chain = _init_qa_chain(index, api_key, chain_type=chain)

        tools = [Tool(
            name="PDF QA System",
            func=ret_qa_chain.run,
            description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
            )
        ]
        
        prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                    You have access to a single tool:"""
        suffix = """Begin!"
        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,  # Single tool. "RetrievalQA"
            prefix=prefix,
            suffix=suffix,
            input_variables=["chat_history", "input", "agent_scratchpad"]
        )

        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(memory_key='chat_history')
        
        llm_chain = LLMChain(
            llm=OpenAI(
                temperature=temperature, 
                openai_api_key=api_key,
                model_name=chat_model
            ),
            prompt=prompt
        )

        # Define 5. Agent executor
        agent = ZeroShotAgent(llm_chain=llm_chain,
                              tools=tools,   # Ret QA chain,
                              verbose=True)
        
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True,
            memory=st.session_state.memory
        )

        # Get question And Answer. Run !
        user_question = st.text_input(
            "**Please insert your question.**",
            placeholder=f'Ask me anything from {fname}'
        )

        if user_question:
            with st.spinner('Running to answer your question ..'):
                result = agent_chain.run(user_question)
                st.info(result, icon='🤖')

        # Allow the user to view 
        # the conversation history and other information stored in the agent's memory
        with st.expander("History / Memory"):
            st.session_state.memory


if __name__=='__main__':
    
    main()
