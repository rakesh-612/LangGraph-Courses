from chains_schemas import AnswerQuestion, ReviseAnswer
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from dotenv import load_dotenv

load_dotenv()

tavily_tool = TavilySearch(max_results=3)  # reduced from 5 to save tokens


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            run_queries,
            name=AnswerQuestion.__name__,
            description="Run search queries to improve the answer.",
        ),
        StructuredTool.from_function(
            run_queries,
            name=ReviseAnswer.__name__,
            description="Run search queries to improve the revised answer.",
        ),
    ]
)
