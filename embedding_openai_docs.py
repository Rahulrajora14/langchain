from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = "text-embedding-3-small",dimensions = 32)

documents= [
    " what is the capital of india",
    " what is the capital of USA",
    " what is the capital of Australia",
    " what is the capital of UK"

]
result = embedding.embed_query(documents)

print (str(result))


