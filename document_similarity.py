from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

documents = [
    "Bumrah is a bowler",
    "Virat Kohli is a batsman",
    "Virat Kohli is a top batsman",
    "Bumrah is a top bowler"
]

text = " top batsman"

query_vector = embedding.embed_query(text)
doc_vectors = embedding.embed_documents(documents)

scores = cosine_similarity([query_vector], doc_vectors)[0]

best_match = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

print(best_match)
print("Most similar document:", documents[best_match[0]])