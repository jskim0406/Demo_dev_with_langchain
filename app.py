import argparse
import databutton as db
import streamlit as st 
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate 
from PIL import Image



def main():
    # Basic setup of the app(Header, Subheader, ..)
    '''
    st.title('text')
    name = 'text'
    st.text('제 이름은 {} 입니다.'.format(name)) # 작은 글씨
    st.header('이 영역은 헤더 영역')  # 제목같은 큰 글씨
    st.subheader('이 영역은 subheader영역')  # 제목보다는 작은 글씨
    st.success('작업이 성공했을때 사용하자')         # 녹색 영역
    st.warning('경고 문구를 보여주고 싶을때 사용하자')   # 노란색 영역
    st.info('정보를 보여주고 싶을때 사용하자')  # 파란색 영역
    st.error('문제가 발생했을때 사용')  # 레드 영역    
    '''
    #################################
    ########## Basic setup ##########
    #################################
    im=Image.open('imgs/HD_ksoe.png')
    st.set_page_config(
        page_title="Hello",
        page_icon=im,
        layout="wide",  # {wide, centered}
    )

    st.title("💵 Simple `Company Name` Recommender")

    st.markdown("회사 이름 짓기 참 어렵죠? ChatGPT가 이 고민을 해결해드립니다.")
    st.markdown("판매하고자 하는 `물품명`과 `마케팅 대상`을 말씀해주시면, ChatGPT가 적절한 회사명을 추천해줍니다! :)")

    # Add a text input box for the user's question
    user_product = st.text_input(
        "Enter Your `Product` which you usually make: ",
        placeholder = "Mobile Phone",
    )

    user_target = st.text_input(
        "Enter your 'Target audience' who you want to advertise to",
        placeholder = "a group of teenage customers who consume cell phones"
    )

    # Arguement parsing..
    parser = argparse.ArgumentParser(description='Get API key..')
    parser.add_argument("--apikey", type=str, required=True, help="If you don't know key value, Just ask jskim")
    args = parser.parse_args()

    #################################
    ########## API setting ##########
    #################################
    API=args.apikey
    # If an API key has been provided, create an OpenAI language model instance
    if API:
        chatopenai = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=API)
    else:
        # If an API key hasn't been provided, display a warning message
        st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")

    #################################
    ########## LLM Chaining #########
    ################################# 
    # "text-davinci-003"(llm model)가 "gpt-3.5-turbo"(Chatmodel) 보다 10배는 더 비싸다고 함. 따라서 아래 코드 중 ChatOpenAI를 활용하는 것을 추천
    if st.button("Hey ChatGPT. It's time to show us what you recommend."):

        template_listup="""
        You are the CEO who want to establish a company who makes {product} for {audience}.
        What is a good name for a company who makes {product} for {audience}? 
        Just say a company name you want to recommend.\n\n
        """
        prompt = PromptTemplate(
            input_variables=["product", "audience"],
            template=template_listup
        )
        chatchain = LLMChain(llm=chatopenai, prompt=prompt)
        
        st.success(
            chatchain({'product': f'{user_product}', 'audience': f'{user_target}'})['text']
        )


if __name__=='__main__':
    main()