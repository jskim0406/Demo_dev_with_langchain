import argparse
import streamlit as st 
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.docstore.document import Document

from langchain.vectorstores import FAISS

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


def text_custom(font_size, text):
    '''
    font_size := ['b', 'm', 's']
    '''
    result=f'<p class="{font_size}-font">{text}</p>'
    return result


def main():

    st.set_page_config(
        page_title="Hello, Welcome to Question Answering on Youtube video page",
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

    st.title("Youtube QA Bot")
    t = "Watch all Youtube videos... Sometimes it's hard, right? Just throw us a URL and ask. We'll answer anything. 😋"
    st.markdown(text_custom('m', t), unsafe_allow_html=True)

    user_in_url = st.text_input(
        "Please enter Youtube URL.",
        placeholder = "https://www.youtube.com/watch?v=reUZRyXxUs4&t=1s",
    )

    if user_in_url:
        width = 40
        side = max((100 - width) / 2, 0.01)
        _, container, _ = st.columns([side, width, side])
        container.video(data=user_in_url)

        t = "Good. From now on, Let's activate the 'Question Answering Bot 🤖' for this YouTube video."
        st.markdown(text_custom('m', t), unsafe_allow_html=True)

    user_in_lang = st.text_input(
        "Tell us what language the Youtube video is in (please enter 'ko' for Korean and 'en' for English).",
        placeholder = "en",
    )

    user_question = st.text_input(
            "Please enter your questions in the video.",
            placeholder = "How AI Could Empower Any Business?"
    )
    

    api_key = st.text_input(
        "Enter Open AI Key.",
        placeholder = "sk-...",
        type="password"
    )


    with st.sidebar:
        embeddeing_model = st.selectbox(
            label='Embedding Model',
            options=['text-embedding-ada-002']
        )

        llm_model = st.selectbox(
            label='LLM Model',
            options=["text-davinci-003",
                     "text-curie-001",
                     "text-babbage-001",
                     "text-ada-001"]
        )

        chain = st.radio(
            label='Chain type',
            options=['stuff',
                    'map_reduce',
                    'refine']
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

    if st.button("Hey ChatGPT. Answer the question right now."):
        API=api_key
        if not API:
            st.warning("Enter your OPENAI API-KEY. If you don't have one Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).")


        # 1. get text data from external source(Youtube video transcription)
        # 참고: https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
        documents = YoutubeLoader.from_youtube_url(user_in_url, language=user_in_lang).load()

        # 2. text preprocessing(Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=['\n\n', '\n', '.', '!', '?', ',', ' ', ''],
            chunk_overlap=200
        )
        docs=text_splitter.split_text(documents[0].page_content)
        new_docs = [Document(page_content=chunk) for chunk in docs]
        
        # 3. define embedding model & provider
        embeddings = OpenAIEmbeddings(openai_api_key=API, model=embeddeing_model)
        
        # 4. create embedding vectorstore(Vector DB) to use as the index
        db = FAISS.from_documents(new_docs, embeddings)

        # 5. Make chin for `question-answering` task with an information retriever
        retriever = db.as_retriever()

        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=API, model=llm_model, temperature=temperature),
            chain_type=chain, 
            retriever=retriever, 
            return_source_documents=True)

        query = user_question
        result = qa({"query": query})

        st.success(result['result'])


if __name__=='__main__':
    main()