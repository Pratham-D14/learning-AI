from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
import os

# Installation Required:
# pip install -qU langchain-community pypdf
# pip install -U langchain-text-splitters
# pip install -qU langchain-openai
# pip install -qU langchain-qdrant


load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load this file in python program
print(pdf_path)

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()  # indexing

# you can get the indexed value
print(docs[21])

# Splitting the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
# chunk overlap helps the current chunk to have some context / recap from the previous chunk so it does not lose the context

# we have successfully split the document into the chunks
chunks = text_splitter.split_documents(documents=docs)

# Vector Embedding
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")  # Chatgpt model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="https://492e0cb7-45fc-4959-b659-6b36d2ae5afc.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=os.environ["QDRANT_API_KEY"],
    collection_name="learning_rag",
)

print("Indexing of documents are done")
