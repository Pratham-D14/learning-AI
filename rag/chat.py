# This is an Retrieval part of the RAG Model
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import requests
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

qdrant_client = QdrantClient(
    url="https://492e0cb7-45fc-4959-b659-6b36d2ae5afc.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=os.environ["QDRANT_API_KEY"],
)

print(qdrant_client.get_collections())

openai_client = OpenAI(
    api_key="AIzaSyAeuwv3_VJ2YhRfwJNvneacn9UMyICGI6c",
    base_url="https://generativelanguage.googleapis.com/v1beta/",
)

# Vector Embedding
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# `embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")` is creating an instance of the
# `OpenAIEmbeddings` class with the specified model parameter set to "text-embedding-3-large". This
# line of code initializes a model for embedding text data, which is commonly used in natural language
# processing tasks like similarity search or text generation. The model "text-embedding-3-large"
# likely refers to a specific pre-trained model for generating embeddings of text data.

vector_db = QdrantVectorStore(
    client=qdrant_client,
    collection_name="learning_rag",
    embedding=embedding_model,
)

# Take user input
user_query = input("👉 Ask Something: ")

# Relevant chunks from the vector db
search_result = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join(
    [
        f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
        for result in search_result
    ]
)

SYSTEM_PROMPT = f"""
You are a helpful AI Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more 

Context:
{context}
"""

response = openai_client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ],
)

# url = "http://localhost:11434/api/chat"
# message_history = [
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user", "content": user_query},
# ]

# payload = {
#     "model": "llama3.2:3b",
#     "messages": message_history,
#     "stream": False,
# }

# response = requests.post(url, json=payload)

print(f"🤖: {response.choices[0].message.content}")
# The line `# print(f"response")` is a comment in Python code. Comments are used to provide
# explanations or notes within the code for better understanding. In this case, the comment is
# indicating that the following line is meant to print something related to the response, but the
# actual printing statement is commented out (denoted by the `#` symbol at the beginning of the line).

# print(response.status_code)
# print(response.text)
