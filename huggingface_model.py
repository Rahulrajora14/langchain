from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    model="google/flan-t5-large",   # ✅ guaranteed working
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

response = client.text_generation(
    "What is the capital of India?"
)

print(response)