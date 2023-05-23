import os
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from llama_index import download_loader
import streamlit as st

@st.cache_resource
def _index_from_wiki(keyword, api_key):
# create a wikipedia download loader object
    WikipediaReader = download_loader("WikipediaReader")

    # load the wikipedia reader object
    loader = WikipediaReader()
    documents = loader.load_data(pages=[keyword])

    # construct the index with the Wikipedia document
    index_wiki = VectorstoreIndexCreator(embedding=OpenAIEmbeddings(open_api_key=api_key)).from_documents(documents)

    return index_wiki

def main():
    # Basic UI setup
    st.title("QA 🤖 about area you are looking for")
    st.markdown(
        "in progress"
        )
    with st.sidebar:
        embeddeing_model = st.selectbox(
            label='Embedding Model',
            options=['text-embedding-ada-002']
        )

        chat_model = st.selectbox(
            label='Chat Model',
            options=["gpt-3.5-turbo"]
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
    api_key = st.text_input(
        "Enter Open AI Key.", placeholder = "sk-...", type="password")
    os.environ['OPENAI_API_KEY'] = api_key
    if api_key:
        # Load file setting
        keyword = st.selectbox("**Please select area you're looking for.**", 
                                    ('Seoul', 'Incheon', 'Daegu'))
        if keyword:
            index_wiki=_index_from_wiki(keyword, api_key)

            # Get question And Answer. Run !
            user_question = st.text_input(
                "**Please insert your question.**",
                placeholder=f'Ask me anything from {keyword}'
            )
                
            if user_question:
                with st.spinner('Running to answer your question ..'):
                    result = index_wiki.query(user_question, llm=OpenAI(temperature=temperature, openai_api_key=api_key, model_name=chat_model))
                    st.info(result, icon='🤖')

if __name__=='__main__':
    main()
