import argparse
from PIL import Image
import streamlit as st 
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from transformers import GPT2TokenizerFast
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.docstore.document import Document

from langchain.vectorstores import FAISS

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings



def main():

    # Arguement parsing..
    parser = argparse.ArgumentParser(description='Get API key..')
    parser.add_argument("--apikey", type=str, required=True, help="If you don't know key value, Just ask jskim")
    parser.add_argument("--lang", type=str, default='ko', help="If you don't know key value, Just ask jskim")
    args = parser.parse_args()


    ########## API setting ##########
    API=args.apikey
    # If an API key has been provided, create an OpenAI language model instance
    if not API:
        # If an API key hasn't been provided, display a warning message
        st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")


    # 1. get text data from external source(Youtube video transcription)
    # Indexes: Accessing external data
    # Langchain에서 제공하는 Document loader를 통해 다양한 source의 데이터를 load가능
    # 참고: https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
    # Youtube video
    video_url = "https://www.youtube.com/watch?v=-hdlF7UN5KA&t" # korean(HD현대)
    # video_url = "https://www.youtube.com/watch?v=SvBR0OGT5VI" # English(TED)
    documents = YoutubeLoader.from_youtube_url(video_url, language=args.lang).load()


    # 2. text preprocessing(Chunking)
    # Text chunking(Split..)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer,
                                                                 chunk_size=800, 
                                                                 chunk_overlap=200)
    docs = text_splitter.split_text(documents[0].page_content)

    # Text wrapping -> 
    # Langchain의 Document class
    '''
    Reference: https://github.com/hwchase17/langchain/blob/master/langchain/text_splitter.py
    시간을 많이 잡아먹은 Debugging point..
    
    vectorstore를 만드는 FAISS, chroma 모두 .from_documents 함수를 사용한다.
    이 함수는 인자로 langchain의 Document 객체를 받는다.
    정확히는 Document객체의 attribute인 Document.page_content를 인식하도록 설계되어있음.
    
    문제는 여기서 발생함.
    기본 text_splitter 'CharacterTextSplitter'는 sepearator='\n\n'으로 주어져있어
    본 분석 대상 text에서 text split이 수행되지 않음(본 text는 seperator를 특정하기 어려운 text)
    따라서 huggingface의 tokenizer를 활용해 token별로 split을 해야함.

    하지만 HF gpt-2 tokenizer와 호환되는 'TokenTextSplitter'는 split_documents가 구현되어 있지 않음.
    오로지 split_text만 구현되어 있음.
    따라서 'TokenTextSplitter'로 split하면 그 결과물은 Document객체가 아닌 text로 반환됨.
    따라서 'TokenTextSplitter'로 split한 결과물은 vectorstore에 전달될 수 없음.

    이를 해결하기 위해 'TokenTextSplitter'로 split 반환된 [text1:str, text2:str, ...]를
    langchain의 Document객체로 wrapping해줌
    [Document1:object, Document2:object, ...]와 같이 변환해줌.
    
    이게 바로 아래의 new_docs를 채워가는 라인의 역할.
    '''
    new_docs = []
    for chunk in docs:
        new_docs.append(Document(page_content=chunk))
    
    # 3. define embedding model & provider
    model_provider='OpenAI'
    if model_provider=='OpenAI':
        embeddings = OpenAIEmbeddings(openai_api_key=API)
    elif model_provider=='HF':
        embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


    # 4. create embedding vectorstore(Vector DB) to use as the index
    db = FAISS.from_documents(new_docs, embeddings)


    # 5. Make chin for `question-answering` task with an information retriever
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=API, max_tokens=500),
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True)

    query = "정주영 정신이란 뭐야? 500자 이내로 짧게 요약해줘."
    result = qa({"query": query})

    print(result['result'])


if __name__=='__main__':
    main()