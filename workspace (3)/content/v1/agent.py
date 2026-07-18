import json
import os
import time
import importlib.util
import requests
import argparse
import subprocess
import re
from collections import deque

class AutonomousAgent:
    def __init__(self, task_file, log_file_override=None, host_override=None, port_override=None, scratchpad_file='scratchpad.txt', steps=5):
        with open(task_file, 'r') as f:
            self.task = json.load(f)

        self.log_file = log_file_override or self.task.get('log_file_path', 'history_log.json')
        self.scratchpad_file = scratchpad_file
        self.host = host_override or self.task.get('ollama_host', 'localhost')
        self.port = port_override or self.task.get('ollama_port', 11434)

        self.steps = steps
        self.goal = self.task.get('goal')
        self.system_prompt = self.task.get('system_prompt')
        self.model = self.task.get('model', 'llama3')
        self.max_history = self.task.get('max_history', 100)
        self.history = deque(self.load_history_list(), maxlen=self.max_history)
        self.last_thinking = ""

    def load_history_list(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                try: return json.load(f)
                except: return []
        return []

    def save_history(self):
        with open(self.log_file, 'w') as f:
            json.dump(list(self.history), f, indent=4)

    def get_tool_definitions(self):
        tools_str = "\nADDITIONAL TOOLS (use with ```tool\n{json}\n```):\n"
        script_dir = os.path.dirname(__file__)
        tools_dir = os.path.join(script_dir, "tools/")
        if not os.path.exists(tools_dir): return ""
        for filename in os.listdir(tools_dir):
            if filename.endswith(".py"):
                try:
                    spec = importlib.util.spec_from_file_location(filename[:-3], os.path.join(tools_dir, filename))
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, 'TOOL_DEFINITION'):
                        tools_str += json.dumps(module.TOOL_DEFINITION, indent=2) + "\n"
                except: continue
        return tools_str

    def get_full_prompt(self):
        notes = "use the scratchpad for persistent notes"
        if os.path.exists(self.scratchpad_file):
            with open(self.scratchpad_file, 'r') as f: notes = f.read().strip()

        parser_instructions = """
PARSER RULES:
1. To write/overwrite a file, use:
```/path/to/file
content
```
2. To run shell commands (timeout in seconds is mandatory):
```shell
# TIMEOUT 60
ls -la
```
3. To use modular tools, use:
```tool
{
  "tool_name": "name",
  "args": {}
}
```
4. NESTED BLOCKS / DELIMITER ESCAPING:
Blocks are parsed based on the exact number of opening backticks (`) with a 3 backtick minimum.
The closing delimiter must exactly match the length of the opening delimiter.
If the content you are writing contains triple backticks (```), you MUST wrap your outer block in four or more backticks (````). 
If the content you are writing contains quadruple backticks (````), you MUST wrap your outer block in five or more backticks.
"""
        system_content = f"{self.system_prompt}\nGOAL: {self.goal}\nSCRATCHPAD ({self.scratchpad_file}): {notes}\n{parser_instructions}{self.get_tool_definitions()}"
        messages = [{"role": "system", "content": system_content}]
        messages.extend(list(self.history))
        return messages

    def call_ollama(self, messages):
        url = f"http://{self.host}:{self.port}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages, 
            "stream": False,
            "options": {
                "num_ctx": 40000,  # todo: make ctx and num_predicr a class variable
                "temperature": 0.5, # not too much randomness
                "num_predict": 20000, # cap output to prevent runaway thinking
                }}
        try:
            response = requests.post(url, json=payload, timeout=1200) # ensure long timeout for llm calls
            data = response.json()
            return data['message']['content'], data['message'].get('thinking', '')
        except Exception as e:
            print(f"Error calling ollama: {str(e)}")
            print(response)
            return f"Error: {str(e)}", ""

    def parse_and_execute(self, content):
        results = []
        lines = content.splitlines()
        in_block = False
        current_header = ""
        current_body = []
        block_delimiter = ""

        for line in lines:
            stripped = line.strip()
            
            if not in_block:
                # Look for an opening delimiter (3 or more backticks)
                if stripped.startswith("```"):
                    in_block = True
                    
                    # Capture the exact sequence of backticks used to open (e.g., ``` or ````)
                    block_delimiter = ""
                    for char in stripped:
                        if char == '`':
                            block_delimiter += '`'
                        else:
                            break
                            
                    # The header is whatever follows the exact delimiter
                    current_header = stripped[len(block_delimiter):].strip()
                    current_body = []
            else:
                # We are in a block. Only close if the line exactly matches the opening delimiter.
                # (Standard markdown closing delimiters do not contain trailing characters)
                if stripped == block_delimiter:
                    in_block = False
                    body_str = "\n".join(current_body)
                    results.append(self._execute_block(current_header, body_str))
                else:
                    # Treat everything else (even ```) as part of the body
                    current_body.append(line)

        # If a block was opened but never closed, execute it anyway (optional fallback)
        if in_block and current_body:
             body_str = "\n".join(current_body)
             results.append("error parsing final block")

        if not results:
            return "No executable blocks found. Use markdown code blocks."
        
        return "\n---\n".join(results)

    def _execute_block(self, header, body):
        header = header.strip()
        if header == 'shell':
            try:
                lines = body.splitlines()
                timeout_line = lines[0].strip()
                body = "\n".join(lines[1:]).strip()

                timeout = -1
                if timeout_line.startswith("#"):
                  parts = timeout_line.split(" ")
                  if parts[-2] == "TIMEOUT":
                    timeout = int(parts[-1])
                if timeout <= 0:
                  raise Exception("could not parse timeout line. the timeout is required to prevent runaway programs. use '# TIMEOUT [seconds]' as first line in your shell")
                
                proc = subprocess.run(body.strip(), shell=True, capture_output=True, text=True, timeout=timeout)
                output = proc.stdout if proc.returncode == 0 else proc.stderr
                return f"Shell Output (exit {proc.returncode}):\n{output}"
            except Exception as e: return f"Shell Error: {str(e)}"
        elif header == 'tool':
            try:
                data = json.loads(body.strip())
                t_name = data.get("tool_name")
                path = os.path.join(os.path.dirname(__file__), "tools/", f"{t_name}.py")
                spec = importlib.util.spec_from_file_location(t_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return f"Tool {t_name} Result: {module.execute(**data.get('args', {}))}"
            except Exception as e: return f"Tool Error: {str(e)}"
        else:
            try:
                dir_name = os.path.dirname(header)
                if dir_name: os.makedirs(dir_name, exist_ok=True)
                with open(header, 'w') as f: f.write(body)
                return f"Written {len(body)} bytes to {header}."
            except Exception as e: return f"Write File Error: {str(e)}"

    def compress_history(self):
        full_text = json.dumps(list(self.history))
        if len(full_text) < 100000: return
        print("--- Compressing History ---")
        history_list = list(self.history)
        to_compress = history_list[:-4]
        keep = history_list[-4:]

        raw_text = "\n".join([f"{msg['role'].upper()}: {str(msg.get('content', ''))}" for msg in to_compress])

        summary_prompt = (
            "You are an AI memory compression module. Review the history and write a concise narrative summary.\n"
            f"RAW HISTORY:\n{raw_text}\n\n"
            "NARRATIVE SUMMARY:"
        )

        summary, _ = self.call_ollama([{"role": "user", "content": summary_prompt}])

        self.history.clear()
        self.history.append({"role": "system", "content": f"SUMMARY OF PREVIOUS EVENTS: {summary}"})
        for msg in keep:
            self.history.append(msg)
        print("History compressed.")

    def run(self):
        for i in range(self.steps):
            print(f"--- Step {i+1} ---")
            content, thinking = self.call_ollama(self.get_full_prompt())
            print(f"Thinking: {thinking}\nContent: {content}")
            self.history.append({"role": "assistant", "content": content})
            
            feedback = self.parse_and_execute(content)
            print(f"Feedback: {feedback}")
            self.history.append({"role": "user", "content": feedback})

            self.compress_history()
            self.save_history()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='task.json')
    parser.add_argument('--steps', type=int, default=1000)
    args = parser.parse_args()
    AutonomousAgent(args.task, steps=args.steps).run()
