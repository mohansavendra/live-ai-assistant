from tavily import TavilyClient
from langchain_core.tools import tool
from config import TAVILY_API_KEY, MAX_SEARCH_RESULTS

tavily = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def web_search(query: str) -> str:
    """
    Search the internet for real-time information.
    Use this for current events, recent data, facts that may have changed,
    or anything requiring up-to-date verification.
    """
    try:
        response = tavily.search(
            query=query,
            max_results=MAX_SEARCH_RESULTS,
            search_depth="advanced",
            include_answer=True,
        )

        # Build a clean, structured result string for the LLM
        output = []

        if response.get("answer"):
            output.append(f"QUICK ANSWER: {response['answer']}\n")

        output.append("SOURCES:")
        for i, result in enumerate(response.get("results", []), 1):
            output.append(
                f"\n[{i}] {result['title']}\n"
                f"    URL: {result['url']}\n"
                f"    {result['content'][:400]}..."
            )

        return "\n".join(output)

    except Exception as e:
        return f"Search failed: {str(e)}. Try rephrasing the query."


@tool
def verify_fact(claim: str) -> str:
    """
    Verify a specific claim or fact using live web search.
    Use this when the user asks to double-check or validate something.
    """
    query = f"verify fact: {claim}"
    try:
        response = tavily.search(
            query=query,
            max_results=3,
            search_depth="advanced",
            include_answer=True,
        )

        output = [f"VERIFICATION RESULTS FOR: '{claim}'\n"]

        if response.get("answer"):
            output.append(f"Summary: {response['answer']}\n")

        for i, result in enumerate(response.get("results", []), 1):
            output.append(
                f"[{i}] {result['title']} — {result['url']}\n"
                f"    {result['content'][:300]}...\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Verification failed: {str(e)}"


# Export all tools as a list for the agent
ALL_TOOLS = [web_search, verify_fact]
