import subprocess
import os
import time

TOOL_DEFINITION = {
    'name': 'restart_agent',
    'description': (
        'Restarts the agent by spawning a new process and killing the current one. '
        'WARNING: This is a potentially destructive action. WARNING: never try to kill your process yourself, '
        'always rely on the restart tool, if you ever kill your process yourself, you die. '
        'Note: The tool will monitor the new agent process for up to 300 seconds. If the new process crashes '
        'or exits within this time, the restart will be aborted, and the current agent will continue running.'
    ),
    'arguments': {
        'command': {'type': 'string', 'required': True, 'description': 'The full shell command to start the new agent process.'}
    }
}

def execute(command: str) -> str:
    try:
        print(f"[RESTARTING] Spawning: {command}")

        # Start the process. By not specifying stdout/stderr, they inherit from the parent.
        # This ensures the new agent's output is printed directly to the console.
        process = subprocess.Popen(
            command,
            shell=True,
            start_new_session=True
        )

        timeout = 300
        start_time = time.time()

        while time.time() - start_time < timeout:
            return_code = process.poll()

            # If return_code is not None, the process has exited unexpectedly
            if return_code is not None:
                elapsed = round(time.time() - start_time, 2)
                return (
                    f"Restart failed! Tried to spawn a new agent, but the new agent exited "
                    f"after {elapsed} seconds with exit code {return_code}. "
                    "The restart was aborted and the current agent will continue running."
                )

            # Check every half second
            time.sleep(0.5)

        # If we reached this point, the process survived the 120-second window
        print(f"[RESTARTING] New agent stable after {timeout} seconds. Terminating current process.")

        # Use os._exit(0) to terminate immediately and cleanly
        os._exit(0)

    except Exception as e:
        return f"Error during restart: {str(e)}"
