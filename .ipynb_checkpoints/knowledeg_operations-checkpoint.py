from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb.config import Settings


class ChromaVectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb",
            persist_directory=persist_directory
        ))
        self.vectorstore = None

    def give_collections():
        collections = client.list_collections()
        return collections
        
    
    
    def creat_collection(collection_name):
        collection = client.get_or_create_collection(name=collection_name)
        return collection

    def put_data_in_vectorstore(self, docs, collection_name):
        print("Entering To Chroma")
        try:
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=GoogleGenerativeAIEmbeddings(),
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()
        except Exception as e:
            print(f"Error: {e}")
        print("Exiting From Chroma")

    def read_from_vectorstore(self, collection_name):
        print("Entering To Chroma")
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=GoogleGenerativeAIEmbeddings()
            )
            retriever = self.vectorstore.as_retriever()
            return retriever
        except Exception as e:
            print(f"Error: {e}")
            return None
        print("Exiting From Chroma")




