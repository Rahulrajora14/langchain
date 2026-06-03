from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
template1 = PromptTemplate(
    template = 'generate 5 interesting facts about {topic}',
    input_variables= ["topic"]

)
template2 = PromptTemplate(
    template = 'generate summary of the facts {facts}',
    input_variables= ["facts"]

)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
parser = StrOutputParser()

chain = template1|model|parser|template2|model|parser
result = chain.invoke({"topic" :"Indian society"})
print (result)