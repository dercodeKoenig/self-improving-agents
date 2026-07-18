import os

TOOL_DEFINITION = {
    'name': 'exit_agent',
    'description': 'Terminates the agent process. Use this when the goal is achieved or a fatal error occurs.',
    'arguments': {
        'final_response': {
            'type': 'string',
            'required': False,
            'description': 'A final summary or message to print before exiting.'
        }
    }
}

def execute(final_response: str = "No final response provided.") -> None:
    print(f"\n[AGENT EXITING]: {final_response}")
    # os._exit is used to ensure the process dies immediately regardless of threads
    os._exit(0)
