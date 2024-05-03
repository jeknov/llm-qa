import os

from langchain.llms import HuggingFaceHub

os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN


def llm_answer(model='google/flan-t5-large', query="What is the currency of Lithuania?"):
    llm = HuggingFaceHub(repo_id=model)
    completion = llm(query)
    print("\n")
    print(completion)


if __name__ == "__main__":
    llm_answer()
