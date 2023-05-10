import os
from llama_index import GPTVectorStoreIndex
from llama_index import download_loader
import streamlit as st

import argparse

def arg_parser():
    # Arguement parsing..
    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", type=str, required=True, help="If you don't know key value, Just ask jskim")
    args = parser.parse_args()
    os.environ['OPENAI_API_KEY'] = args.apikey
    return args

@st.cache_resource
def _index_from_wiki(keyword):
# create a wikipedia download loader object
    WikipediaReader = download_loader("WikipediaReader")

    # load the wikipedia reader object
    loader = WikipediaReader()
    documents = loader.load_data(pages=[keyword])

    # construct the index with the Wikipedia document
    index_wiki = GPTVectorStoreIndex.from_documents(documents)

    return index_wiki

def main():
    # Basic UI setup
    st.title("QA 🤖 about area you are looking for")
    st.markdown(
        "in progress"
        )
    st.sidebar.markdown(
        "still in progress"
        )

    # Load file setting
    keyword = st.selectbox("**Please select area you're looking for.**", 
                                   ('Seoul', 'Incheon', 'Daegu'))
    if keyword:
        index_wiki=_index_from_wiki(keyword)
        query_engine = index_wiki.as_query_engine()

        # Get question And Answer. Run !
        user_question = st.text_input(
            "**Please insert your question.**",
            placeholder=f'Ask me anything from {keyword}'
        )
            
        if user_question:
            with st.spinner('Running to answer your question ..'):
                result = query_engine.query(user_question).response
                st.info(result, icon='🤖')

if __name__=='__main__':
    arg_parser()
    main()