from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

import streamlit as st


def llm_answer(query):
    template = """Question: {question}
Answer:"""
    prompt = PromptTemplate.from_template(template)
    model = 'mistralai/Mistral-7B-Instruct-v0.2'
    llm = HuggingFaceEndpoint(
        repo_id=model,
        max_length=128,
        temperature=0.5,
        max_new_tokens=200,
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    completion = llm_chain.run(query)
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
