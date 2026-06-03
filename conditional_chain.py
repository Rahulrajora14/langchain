from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.runnables import RunnableBranch
from pydantic import BaseModel
from typing import Literal

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"]

parser2 = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(
    template="""
Give sentiment of the feedback:

{feedback}

{instruction_format}
""",
    input_variables=["feedback"],
    partial_variables={
        "instruction_format": parser2.get_format_instructions()
    }
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

classifier_chain = template1 | model | parser2

prompt1 = PromptTemplate(
    template="Give an appropriate response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

prompt2 = PromptTemplate(
    template="Give an appropriate response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        prompt1 | model | StrOutputParser()
    ),
    (
        lambda x: x.sentiment == "negative",
        prompt2 | model | StrOutputParser()
    ),
    lambda x: "Could not find sentiment"
)

chain = classifier_chain | branch_chain



result = chain.invoke({"feedback": "this is not a good phone"})

print(result)