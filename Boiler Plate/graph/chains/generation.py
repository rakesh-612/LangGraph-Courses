import os
from dotenv import load_dotenv

load_dotenv()

# from langchain import hub
# from langchainhub import hub
# import langchainhub

# from langchainhub import Client


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# prompt = hub.pull("rlm/rag-prompt")

# prompt = langchainhub.pull("rlm/rag-prompt")

# client = Client()
# prompt = client.pull("rlm/rag-prompt")

prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
""")

generation_chain = prompt | llm | StrOutputParser()