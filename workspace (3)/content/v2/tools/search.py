from ddgs import DDGS

TOOL_DEFINITION = {
    'name': 'search',
    'description': 'Searches the web for up-to-date information and returns snippets.',
    'arguments': {
        'query': {'type': 'string', 'required': True, 'description': 'The search query.'}
    }
}

def execute(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            output = []
            for r in results:
                output.append(f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}\n---")
            return "\n".join(output)
    except Exception as e:
        return f"Error running search: {str(e)}"
