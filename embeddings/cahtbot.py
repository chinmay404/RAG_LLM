# from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)


load_dotenv()
key = os.getenv('GOOGLE_API_KEY')
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro", temperature=0.1, api_key=key)
llm = Ollama(model="qwen2:1.5b")

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def read_chroma():
    vectorstore_disk = Chroma(
        persist_directory="./chroma_db",       
        embedding_function=gemini_embeddings  
    )
    retriever = vectorstore_disk.as_retriever()
    return retriever


def main():
    bot_template = """
    Your job is to use data Which is provided to you and answer user accordingly.
    If you don't know the answer, just say that i don't know about the question that user have asked.
    make sure that youll give your best to answer the question.

    {context}


    """

    bot_template_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=bot_template,
            input_variables=["context"]
        )
    )

    user_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )

    message = [
        bot_template_prompt,
        user_prompt_template
    ]

    final_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=message
    )

    chain = final_template | llm | StrOutputParser()

    context = read_chroma()
    question = "What is Persons name?"

    output = chain.invoke({"context": context, "question": question})
    print(output)
