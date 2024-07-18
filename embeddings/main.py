# from langchain import PromptTemplate

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader

import os


load_dotenv()
key = os.getenv('GOOGLE_API_KEY')
LANGCHAIN_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0.1, api_key=key)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def simple_chat(llm):
    parser = StrOutputParser()
    system_template = "You Are a {pro} and you have experties in that. answer the question accordingly if its not realted to your profession or experties give Humerous response that you dont know about the question that user have asked"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]

    )

    chain = prompt_template | llm | parser

    print(chain.invoke(
        {"pro": "Doctor", "text": "hi, Doctor How to code in python"}))


def chat_bot(llm):
    # loader = WebBaseLoader(
    #     "https://www.teachermagazine.com/in_en/articles/teachers-reading-for-pleasure")
    # loader = PyPDFLoader("./doc.pdf")
    # data = loader.load_and_split()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=1)
    # all_splits = text_splitter.split_documents(data)
    # put_data_in_vectorstore(all_splits)
    # read_chroma()
    print("There are", langchain_chroma._collection.count(), "in the collection")




    # vectorstore_disk = Chroma(
    #     persist_directory="./chroma_db",
    #     embedding_function=gemini_embeddings
    # )


def read_chroma():
    vectorstore_disk = Chroma(
        persist_directory="./chroma_db",       # Directory of db
        embedding_function=gemini_embeddings   # Embedding model
    )

    retriever = vectorstore_disk.as_retriever()
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.\n
    Question: {question} \nContext: {context} \nAnswer:"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    print(f"RETRIVER : {retriever}")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )

    output = rag_chain.invoke("Tell me Summury ")

    print(output)


if __name__ == '__main__':
    # simple_chat(llm)
    # genrator_chain()
    chat_bot(llm)
