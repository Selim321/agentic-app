from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
import os

search = TavilySearchResults(max_results=1, tavily_api_key=os.getenv("TAVILY_API_KEY"))


python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes Python code and returns the result.",
    func=python_repl.run,
)

from langchain_groq import ChatGroq
import os 

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY")
)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tools = [search, repl_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# for step in agent_executor.stream(
#     {
#         "input": (
#             "Create a pie chart of the top 5 most used programming languages in 2025."
#         )
#     }
# ):
#     if "output" in step:
#         print(step["output"])

for step in agent_executor.stream(
    (
        {
            "input": (
                "What is the temperature in Algiers?"
            )
        }
    ),
):
    if "output" in step:
        print(step["output"])