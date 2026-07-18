import json

TOOL_DEFINITION = {
    'name': 'ask_human',
    'description': 'Pauses execution and asks a human for guidance or information. Only use in absolute emergency case, Always try to solve a task yourself or find possible workarounds. You can also write your own tools if required.',
    'arguments': {
        'question': {'type': 'string', 'required': True}
    }
}

def execute(question: str) -> str:
    print(f"\n[AGENT IS ASKING]: {question}")
    return input("Human Response: ")
