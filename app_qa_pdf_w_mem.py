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



def arg_parser():
    # Arguement parsing..
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", type=str, required=True, help="If you don't know key value, Just ask jskim")
    args = parser.parse_args()
    return args


@st.cache_data
def _chunks_from_pdf(file:BytesIO):
    """
    Args
        1. file: PDF file 그 잡채.
    
    1. pdf에서 text 추출
    2. text를 Document class로 변환
    2. Chunking
    """    
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
def _embed():
    embeddings = OpenAIEmbeddings(openai_api_key=args.apikey)

    with st.spinner("Now.. we are indexing the data.. Please wait.."):
        index = FAISS.from_documents(chunks, embeddings)

    st.success("It's Done!", icon='✅')
    return index


def _init_qa_chain(args, index, chain_type='map_reduce'):

    qa = RetrievalQA.from_chain_type(
                llm=OpenAI(openai_api_key=args.apikey),
                chain_type = chain_type,
                retriever=index.as_retriever(),
            )
    return qa


def main(args):
    # Basic UI setup
    st.title("QA 🤖 with Memory 🧠")
    st.markdown(
        "in progress"
        )
    st.sidebar.markdown(
        "still in progress"
        )

    # Load file setting
    loaded_file = st.file_uploader("**Please upload your PDF file here.**", 
                                   type = ["pdf"])
    if loaded_file:
        fname = loaded_file.name

        # make chunks
        global chunks
        chunks=_chunks_from_pdf(loaded_file)

        # show page content(extra utility)
        if chunks:
            with st.expander("Show Page Content", expanded=False):
                sel_pagenum = st.number_input(
                    label="Select page number",
                    min_value=1,
                    max_value=len(chunks),
                    step=1
                )
                chunks[sel_pagenum-1]

        # Embed chunks into dense vector space(= indexing)
        index = _embed()

        # Define 1. Retrieval Q&A chain
        ret_qa_chain = _init_qa_chain(args, index)

        # Define 2. 'Tools' and 'Prompt'ing In 'Agent' (Key part of this..)
        tools = [Tool(
            name="PDF QA System",
            func=ret_qa_chain.run,
            description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
            )]
        
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

        # Define 3. Memory! 
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key='chat_history'
            )
        
        # Define 4. llm chain
        # RetrievalQA는 검색을 위한 chain
        # LLM chain은 chatbot에서 대화를 하기 위한 LM chain
        llm_chain = LLMChain(
            llm=OpenAI(
            temperature=0.3, 
            openai_api_key=args.apikey,
            model_name='gpt-3.5-turbo'
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
    
    global args
    args = arg_parser()

    main(args)
