# PDF -> BERT
'''
Embedding
    - Open AI embedding (ing)
	- huggingface(Sentence Transformer) embedding
	- Vicuna embedding
LLM
	- OpenAI LLM (ing)
    - huggingface
	- Vicuna
    - MPT
Reference
    - main architecture of this code
        - https://medium.com/the-techlife/using-huggingface-openai-and-cohere-models-with-langchain-db57af14ac5b
    - huggingface embeddings
        - https://huggingface.co/blog/getting-started-with-embeddings
'''

import openai
import argparse
import streamlit as st

from io import BytesIO
from pypdf import PdfReader

from langchain import PromptTemplate, LLMChain, HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.output_parsers import RegexParser



def arg_parser():
    # Arguement parsing..
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", type=str, required=True, help="If you don't know key value, Just ask jskim")
    parser.add_argument("--hfkey", type=str, default='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', help='HuggingFace api key')
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
    fname = file.name
    pdf = PdfReader(file)
    texts = [p.extract_text() for p in pdf.pages]
    pages = [Document(page_content=text) for text in texts]
    
    # Add page numbers as metadata
    for i, page in enumerate(pages):
        page.metadata['source'] = f'{fname}-{i+1} page'

    # 2. Chunking & metadata 추가(chunk index 정보)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=['\n\n', '\n', '.', '!', '?', ',', ' ', ''],
        chunk_overlap=200
    )
    chunks=text_splitter.split_documents(pages)
    return chunks


@st.cache_data
def _embed(emb_provider):
    # OpenAI
    if emb_provider == 'OpenAI':
        embeddings = OpenAIEmbeddings(openai_api_key=args.apikey)

    # Huggingface(Sentence Transformer)
    elif emb_provider == 'HF':
        hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Vicuna
    elif emb_provider == 'VC':
        pass

    with st.spinner("Now.. we are indexing the data.. Please wait.."):
        index = FAISS.from_documents(chunks, embeddings)

    st.success("It's Done!", icon='✅')
    return index


def _getmodel(model_provider):
    '''
    [temperature]
    temperature indicates the degree of randomness of the output. 
    So with a higher temperature, you might sometimes get better answers but sometimes worse answers.
    '''

    if model_provider == 'OpenAI':
        # model = OpenAI(model_name="text-davinci-003", openai_api_key=args.apikey)
        model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=args.apikey)
        
    elif model_provider == 'HF':
        model = HuggingFaceHub(repo_id="facebook/mbart-large-50",
                               model_kwargs={"temperature": 0, "max_length":200},
                               huggingfacehub_api_token=args.hfkey)
        
    elif model_provider == 'VC':
        pass

    elif model_provider == 'mpt':
        pass

    return model


def _getchain(model, chain_type):

    return_intermediate_steps = False

    if chain_type == 'refine':
        qa_chain = load_qa_with_sources_chain(model,
                                              chain_type = chain_type,
                                              return_intermediate_steps=return_intermediate_steps)
    elif chain_type == 'stuff':
        qa_chain = load_qa_with_sources_chain(model, 
                                              chain_type = chain_type)
    else:
        qa_chain = load_qa_with_sources_chain(model, 
                                              chain_type = chain_type,
                                              return_intermediate_steps=return_intermediate_steps)
        
    return qa_chain


def main(args):
    st.title("QA 🤖 on PDF with ALL LLM provider.")
    st.markdown(
        "In progress .."
    )
    st.sidebar.markdown(
        "In progress .. "
    )
    # 1. load file
    loaded_file = st.file_uploader("**Please upload your PDF file here.**", 
                                   type = ["pdf"])
    if loaded_file:
        # 2. make chunks
        global chunks
        chunks=_chunks_from_pdf(loaded_file)        

        # import IPython; IPython.embed(); exit(1)

        # 3. show page content(extra utility)
        if chunks:
            with st.expander("Show Page Content", expanded=False):
                sel_pagenum = st.number_input(
                    label="Select page number",
                    min_value=1,
                    max_value=len(chunks),
                    step=1
                )
                chunks[sel_pagenum-1].page_content

    
        # 4. embed chunks into dense vectorspace (= indexing)
        ########## [here] ##########
        emb_provider = 'OpenAI'
        if emb_provider:
            index = _embed(emb_provider)
    

        # 5. define llm model
        ########## [here] ##########
        model_provider = 'OpenAI'
        if model_provider:
            model = _getmodel(model_provider)

            # 6. define chain ['stuff', 'map_reduce', 'refine']
            ########## [here] ##########
            # chain_type = 'refine'
            # chain_type = 'stuff'
            chain_type = 'map_reduce'
            qa_chain = _getchain(model, chain_type)

            if qa_chain:
                # 7. query insert and get answer
                # query = "Any question that you want to ask the model"
                ########## [here] ##########
                query = "Waht is the 'BERT'?"
                documents = index.similarity_search(query)
                result = qa_chain({"input_documents": documents, "question": query},
                                return_only_outputs=True)


                print(result)
            


if __name__=='__main__':
    
    global args
    args = arg_parser()
    main(args)