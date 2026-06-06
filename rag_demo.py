from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

load_dotenv()

# Load documents
loader = TextLoader("data.txt")
docs = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Vector DB
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Retriever
retriever = vectorstore.as_retriever()

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro"
)

# Query
question = "What is the capital of India?"

# Retrieve
context = retriever.invoke(question)

# Generate
prompt = f"""
Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print(response.content)