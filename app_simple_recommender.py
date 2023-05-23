import argparse
import databutton as db
import streamlit as st 
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate 


def text_custom(font_size, text):
    '''
    font_size := ['b', 'm', 's']
    '''
    result=f'<p class="{font_size}-font">{text}</p>'
    return result


def main():
 
    st.set_page_config(
        page_title="Hello, Welcome to Simple song recommender page",
        layout="wide",  # {wide, centered}
    )
    # reference
    ## https://discuss.streamlit.io/t/change-input-text-font-size/29959/4
    ## https://discuss.streamlit.io/t/change-font-size-in-st-write/7606/2
    st.markdown("""<style>.b-font {font-size:25px !important;}</style>""", unsafe_allow_html=True)    
    st.markdown("""<style>.m-font {font-size:20px !important;}</style>""" , unsafe_allow_html=True)    
    st.markdown("""<style>.s-font {font-size:15px !important;}</style>""" , unsafe_allow_html=True)    
    tabs_font_css = """<style>div[class*="stTextInput"] label {font-size: 15px;color: black;}</style>"""
    st.write(tabs_font_css, unsafe_allow_html=True)


    st.title("💵 Simple `Song` Recommender")
    t = "It's hard to find music to comfort me, right? ChatGPT can help."
    st.markdown(text_custom('b', t), unsafe_allow_html=True)
    t = "If you tell us about 'Mood' and 'Song Genre', ChatGPT will recommend Song that suits you now! 😋"
    st.markdown(text_custom('m', t), unsafe_allow_html=True)

    with st.sidebar:
        model = st.selectbox(
            label='Model',
            options=['gpt-3.5-turbo']
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

    mood = st.text_input(
        "Tell me how you feel right now. And press Enter.",
        placeholder = "I am exhausted. I want to cheer up.",
    )

    genre = st.text_input(
        "Please tell me the genre of music you want to listen to. And press Enter.",
        placeholder = "K-pop"
    )

    api_key = st.text_input(
        "Enter Open AI Key.",
        placeholder = "sk-...",
        type="password"
    )

    if api_key:
        chatopenai = ChatOpenAI(model_name=model, temperature=temperature, openai_api_key=api_key)
    else:
        st.warning("Enter your OPENAI API-KEY. If you don't have one Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")

    if st.button("Hey ChatGPT. It's time to show us what you recommend."):

        template="""
        You've heard a friend talk about feeling like this.
        Your friend says {mood}

        You want to recommend a song for a friend.
        Recommend songs within the {genre} genre that match your friend's mood.
        """

        prompt = PromptTemplate(
            input_variables=["mood", "genre"],
            template=template
        )
        chatchain = LLMChain(llm=chatopenai, prompt=prompt)
        
        st.success(
            chatchain({'mood': f'{mood}', 'genre': f'{genre}'})['text']
        )


if __name__=='__main__':
    main()