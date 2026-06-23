import os
from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()