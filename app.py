from langchain_community.llms import HuggingFaceHub

import streamlit as st


def llm_answer(query):
    model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    llm = HuggingFaceHub(repo_id=model)
    completion = llm(query)
    return completion


st.set_page_config(page_title='LangChain Demo', page_icon=":robot:")
st.header("LangChain Demo")


def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


user_input = get_text()
response = llm_answer(user_input)

submit = st.button("Generate")

if submit:
    st.subheader("Answer: ")
    st.write(response)
