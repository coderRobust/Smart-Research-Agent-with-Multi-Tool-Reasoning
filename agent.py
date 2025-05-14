from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Tool: Web Search
@tool
def web_search(query: str) -> str:
    """Useful for finding real-time information using a web search engine."""
    return TavilySearchResults().run(query)

# Tool: Structuring output
parser = JsonOutputParser()
prompt = PromptTemplate.from_template(
    "Extract structured answer for: {question}\n\nAnswer:"
)

# Graph: Define LLM node
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define nodes
graph = StateGraph()
graph.add_node("Search", ToolNode(web_search))
graph.add_node("LLM", llm)
graph.set_entry_point("Search")
graph.add_edge("Search", "LLM")
graph.add_edge("LLM", END)
graph.set_finish_point(END)

# Memory checkpointing
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Run
if __name__ == "__main__":
    question = "What is latest update about Open AI?"
    output = app.invoke({"input": HumanMessage(content=question)})
    print("\n🔍 Final Answer:\n", output["LLM"])
