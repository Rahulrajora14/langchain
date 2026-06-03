from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Template 1
template1 = PromptTemplate(
    template='Generate 5 interesting facts about:\n{text}',
    input_variables=["text"]
)

# Template 2
template2 = PromptTemplate(
    template='Generate 5 quiz questions on the topic:\n{text}',
    input_variables=["text"]
)

# Template 3 (FIXED)
template3 = PromptTemplate(
    template='Merge both into a single document:\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}',
    input_variables=["notes", "quiz"]
)

# Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Parser
parser = StrOutputParser()

# Parallel execution
parallel_chain = RunnableParallel({
    "notes": template1 | model | parser,
    "quiz": template2 | model | parser
})

# Merge chain
merge_chain = template3 | model | parser

# Final chain
chain = parallel_chain | merge_chain

# Input text
text = "Samsung Galaxy S24..."

# Run
result = chain.invoke({"text": text})

print(result)