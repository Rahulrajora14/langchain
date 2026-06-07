from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# Load API Key
load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --------------------
# TOOLS
# --------------------

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def square(num: int) -> int:
    """Return square of a number."""
    return num * num


# List of tools
tools = [multiply, add, square]

# --------------------
# PROMPT
# --------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.
            Use tools whenever required.
            Solve problems step by step.
            """
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

# --------------------
# AGENT
# --------------------

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# --------------------
# RUN
# --------------------

response = agent_executor.invoke(
    {
        "input": "Multiply 25 and 10 and then square the result."
    }
)

print("\nFinal Answer:")
print(response["output"])