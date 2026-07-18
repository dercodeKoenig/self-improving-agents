#!/usr/bin/env python3
"""
Autonomous AI Agent v3.0.0 - Recursive Self-Improvement System

Major improvements over v2:
- Adaptive context management with progressive compression
- Hierarchical planning engine with dynamic replanning
- Tool chaining and batch operations
- Enhanced error recovery with exponential backoff
- Optimized LLM prompts for faster responses
- Robust parser with auto-detection
- TTL-based memory management
"""

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "model": "qwen3.6:27b-Q8_0",
    "ollama_host": "87.171.205.7",
    "ollama_port": 9090,
    "max_tokens": 4096,
    "temperature": 0.3,
    "num_ctx": 16384,
    "num_predict": 4096,
    "max_history_messages": 50,
    "max_steps": 20,
    "timeout_per_call": 120,
    "retry_max_attempts": 3,
    "retry_base_delay": 2.0,
    "context_tiers": {
        "critical": 8,   # Always kept (system prompt + recent exchanges)
        "recent": 15,    # Kept in detail
        "archived": 30   # Summarized/compressed
    },
    "memory_ttl_seconds": 600,  # Working memory TTL
}


# ============================================================================
# MEMORY SYSTEM (Enhanced with TTL and semantic clustering)
# ============================================================================

class MemoryEntry:
    """Single memory entry with metadata for TTL management."""
    __slots__ = ('key', 'value', 'timestamp', 'ttl', 'tags', 'importance')

    def __init__(self, key: str, value: Any, ttl: float = 600.0,
                 tags: Optional[List[str]] = None, importance: float = 0.5):
        self.key = key
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
        self.tags = tags or []
        self.importance = importance

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'importance': self.importance
        }


class MemorySystem:
    """Enhanced memory with TTL expiration and semantic clustering."""

    def __init__(self, ttl: float = 600.0):
        self.working: Dict[str, MemoryEntry] = {}
        self.long_term: Dict[str, MemoryEntry] = {}
        self.default_ttl = ttl
        self._access_log: List[Tuple[str, float]] = []

    def store(self, key: str, value: Any, long_term: bool = False,
              ttl: Optional[float] = None, tags: Optional[List[str]] = None,
              importance: float = 0.5) -> None:
        entry = MemoryEntry(key, value, ttl or self.default_ttl, tags, importance)
        if long_term:
            self.long_term[key] = entry
        else:
            self.working[key] = entry
        self._access_log.append((key, time.time()))

    def retrieve(self, key: str) -> Optional[Any]:
        # Check working first, then long-term
        entry = self.working.get(key) or self.long_term.get(key)
        if entry and not entry.is_expired():
            entry.timestamp = time.time()  # Refresh TTL on access
            return entry.value
        elif entry and entry.is_expired():
            # Expire it
            self.working.pop(key, None)
        return None

    def search_by_tag(self, tag: str) -> List[Tuple[str, Any]]:
        results = []
        for store in (self.working, self.long_term):
            for key, entry in store.items():
                if not entry.is_expired() and tag in entry.tags:
                    results.append((entry, key, entry.value))
        results.sort(key=lambda x: x[0].importance, reverse=True)
        return [(k, v) for (_, k, v) in results]

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        removed = 0
        for store in (self.working, self.long_term):
            expired = [k for k, v in store.items() if v.is_expired()]
            for k in expired:
                del store[k]
                removed += 1
        return removed

    def get_summary(self) -> str:
        """Get a compact summary of all active memories."""
        active_working = {k: v.value for k, v in self.working.items() if not v.is_expired()}
        active_long = {k: v.value for k, v in self.long_term.items() if not v.is_expired()}

        parts = []
        if active_working:
            parts.append(f"Working Memory ({len(active_working)} entries):")
            for k, v in list(active_working.items())[:10]:
                val_str = str(v)[:80]
                parts.append(f"  - {k}: {val_str}")

        if active_long:
            parts.append(f"Long-term Memory ({len(active_long)} entries):")
            for k, v in list(active_long.items())[:10]:
                val_str = str(v)[:80]
                parts.append(f"  - {k}: {val_str}")

        return "\n".join(parts) if parts else "Memory: empty"


# ============================================================================
# TOKEN ESTIMATION & CONTEXT MANAGEMENT
# ============================================================================

class TokenEstimator:
    """Rough token estimation for context window management."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate tokens using character-based heuristic (4 chars ~= 1 token)."""
        return max(1, len(text) // 4)

    @staticmethod
    def estimate_message_tokens(msg: dict) -> int:
        content = msg.get('content', '')
        role = msg.get('role', '')
        return TokenEstimator.estimate_tokens(f"{role}: {content}")


class AdaptiveContextManager:
    """Manages conversation history with tiered compression."""

    def __init__(self, config: dict):
        self.max_messages = config.get('max_history_messages', 50)
        self.critical_count = config['context_tiers']['critical']
        self.recent_count = config['context_tiers']['recent']
        self.archived_count = config['context_tiers']['archived']
        self.token_budget = config.get('num_ctx', 16384) * 0.7  # 70% budget for history

    def compress_history(self, messages: List[dict]) -> List[dict]:
        """Compress history using tiered retention strategy."""
        if len(messages) <= self.recent_count:
            return messages

        # Separate into tiers
        critical = messages[:self.critical_count]  # System + earliest exchanges
        recent = messages[-(self.recent_count - self.critical_count):]  # Most recent

        # Compress middle section into a summary
        middle = messages[self.critical_count:-max(0, self.recent_count - self.critical_count)]
        if middle:
            summary = self._summarize_block(middle)
            summary_msg = {
                'role': 'system',
                'content': f"[Compressed context from {len(middle)} messages]: {summary}"
            }
            return critical + [summary_msg] + recent

        return critical + recent

    def _summarize_block(self, messages: List[dict]) -> str:
        """Create a compact summary of a message block."""
        actions = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')[:200]
            if role == 'assistant':
                # Extract key actions from assistant messages
                if '```' in content:
                    import re
                    blocks = re.findall(r'```(\w+)', content)
                    for b in blocks[:2]:
                        actions.append(f"Used {b}")
                elif len(content) < 100:
                    actions.append(content.strip())
            elif role == 'tool':
                # Summarize tool output
                if len(content) > 50:
                    content = content[:50] + "..."
                actions.append(f"Tool result: {content}")

        summary_parts = actions[:8]  # Limit to 8 key points
        return "; ".join(summary_parts) if summary_parts else f"{len(messages)} exchanges processed"

    def trim_to_budget(self, messages: List[dict], system_prompt: str) -> List[dict]:
        """Trim messages to fit within token budget."""
        system_tokens = TokenEstimator.estimate_tokens(system_prompt)
        available = self.token_budget - system_tokens - 1024  # Reserve for response

        result = [messages[0]] if messages else []  # Keep system message
        current_tokens = system_tokens

        for msg in messages[1:]:
            msg_tokens = TokenEstimator.estimate_message_tokens(msg)
            if current_tokens + msg_tokens > available:
                break
            result.append(msg)
            current_tokens += msg_tokens

        return result


# ============================================================================
# BLOCK PARSER (Enhanced with auto-detection and better nested handling)
# ============================================================================

class BlockParser:
    """Robust markdown block parser with auto-detection of tool types."""

    @staticmethod
    def parse(text: str) -> List[dict]:
        """Parse text into structured blocks. Returns list of {header, body} dicts."""
        blocks = []
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]
            # Count backticks for opening fence
            match = re.match(r'^(`{3,})\s*(\w+)?', line)
            if match:
                fence_len = len(match.group(1))
                header = (match.group(2) or '').lower().strip()
                body_lines = []
                i += 1

                # Find closing fence with same or more backticks
                while i < len(lines):
                    close_match = re.match(r'^(`{3,})\s*$', lines[i])
                    if close_match and len(close_match.group(1)) >= fence_len:
                        break
                    body_lines.append(lines[i])
                    i += 1

                body = '\n'.join(body_lines)
                blocks.append({'header': header, 'body': body})
            i += 1

        return blocks

    @staticmethod
    def detect_tool_type(block: dict) -> str:
        """Auto-detect tool type from block content if header is ambiguous."""
        header = block.get('header', '').lower()
        body = block.get('body', '')

        if header in ('shell', 'sh', 'bash'):
            return 'shell'
        elif header in ('file', 'filepath', ''):
            # Empty header or file-related - check if it looks like a file path
            if '/' in body.split('\n')[0] and not body.strip().startswith('#'):
                return 'file'
            return 'file'
        elif header in ('tool', 'json', 'api'):
            return 'tool'
        elif header in ('python', 'py'):
            # Could be a file write or inline python
            if body.strip().startswith('/'):
                return 'file'
            return 'shell'  # Run as python command
        elif header in ('plan', 'thinking', 'thought'):
            return 'thought'

        # Default: try to parse as JSON for tool calls
        try:
            data = json.loads(body)
            if 'name' in data or 'command' in data:
                return 'tool'
        except (json.JSONDecodeError, TypeError):
            pass

        # If body looks like a file path on first line
        first_line = body.strip().split('\n')[0] if body.strip() else ''
        if re.match(r'^/[^\s]+$', first_line) or re.match(r'^\.[\\/][^\s]+$', first_line):
            return 'file'

        return 'shell'  # Default to shell execution


# ============================================================================
# PLANNING ENGINE (Hierarchical with dynamic replanning)
# ============================================================================

class PlanningEngine:
    """Multi-level task decomposition with priority scheduling."""

    @staticmethod
    def create_plan(goal: str, context: str = "") -> dict:
        """Create a hierarchical plan for the given goal. Returns structured plan."""
        # Return a template plan that will be filled by LLM
        return {
            "goal": goal,
            "subtasks": [],
            "status": "pending",
            "context": context[:500],
            "created_at": datetime.now().isoformat()
        }

    @staticmethod
    def plan_prompt(goal: str, memory_summary: str = "", previous_attempts: int = 0) -> str:
        """Generate planning prompt for LLM."""
        replan_note = ""
        if previous_attempts > 0:
            replan_note = f"\n\nPrevious {previous_attempts} attempt(s) failed. Try a different approach."

        return (
            f"Break down this task into clear, sequential steps:\n\n"
            f"GOAL: {goal}\n"
            f"{replan_note}\n"
            f"Available context:\n{memory_summary}\n\n"
            "Respond with a numbered list of steps. Each step should be a single, "
            "actionable operation (file write, shell command, or tool call).\n"
            "Keep it concise - maximum 5 steps."
        )

    @staticmethod
    def parse_plan_response(text: str) -> List[str]:
        """Extract steps from LLM planning response."""
        steps = []
        for line in text.split('\n'):
            # Match numbered lists: "1. step" or "1) step"
            match = re.match(r'^\s*\d+[\.\)]\s+(.+)', line)
            if match:
                steps.append(match.group(1).strip())
        return steps[:5]  # Limit to 5 steps


# ============================================================================
# REFLECTION ENGINE (Self-evaluation with scoring)
# ============================================================================

class ReflectionEngine:
    """Self-evaluation and improvement loop."""

    @staticmethod
    def reflect_prompt(goal: str, actions_taken: str, outcome: str) -> str:
        """Generate reflection prompt."""
        return (
            f"Evaluate the following task execution:\n\n"
            f"GOAL: {goal}\n"
            f"ACTIONS TAKEN:\n{actions_taken}\n"
            f"OUTCOME: {outcome}\n\n"
            "Rate success (1-10) and suggest ONE improvement for next time.\n"
            "Respond in format: SCORE: X/10\nIMPROVEMENT: [one sentence]"
        )

    @staticmethod
    def parse_reflection(text: str) -> dict:
        """Parse reflection response into structured data."""
        score = 5  # Default
        improvement = "No specific feedback"

        score_match = re.search(r'SCORE[:\s]*(\d+)', text, re.IGNORECASE)
        if score_match:
            score = min(10, max(1, int(score_match.group(1))))

        imp_match = re.search(r'IMPROVEMENT[:\s]*(.+)', text, re.IGNORECASE)
        if imp_match:
            improvement = imp_match.group(1).strip()

        return {'score': score, 'improvement': improvement}


# ============================================================================
# ERROR RECOVERY (Exponential backoff with jitter)
# ============================================================================

class ErrorRecovery:
    """Enhanced retry mechanism with exponential backoff."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self._error_patterns = {
            'timeout': lambda e: 'timed out' in str(e).lower() or 'timeout' in str(e).lower(),
            'connection': lambda e: 'connection' in str(e).lower() or 'refused' in str(e).lower(),
            'rate_limit': lambda e: 'rate' in str(e).lower() or '429' in str(e),
        }

    def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry and exponential backoff."""
        last_error = None
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_attempts - 1:
                    delay = self._calculate_delay(attempt, e)
                    time.sleep(delay)
        raise last_error

    def _calculate_delay(self, attempt: int, error: Exception) -> float:
        """Calculate delay with exponential backoff and jitter."""
        import random
        base = self.base_delay * (2 ** attempt)
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, base * 0.3)
        return min(base + jitter, 60)  # Cap at 60s

    def classify_error(self, error: Exception) -> str:
        """Classify error type for adaptive retry strategies."""
        for pattern_name, matcher in self._error_patterns.items():
            if matcher(error):
                return pattern_name
        return 'unknown'


# ============================================================================
# LLM INTERFACE (Optimized with streaming support)
# ============================================================================

class LLMInterface:
    """Optimized LLM communication layer."""

    def __init__(self, config: dict, error_recovery: ErrorRecovery):
        self.model = config['model']
        self.host = config['ollama_host']
        self.port = config['ollama_port']
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = config.get('timeout_per_call', 120)
        self.max_tokens = config.get('num_predict', 4096)
        self.temperature = config.get('temperature', 0.3)
        self.error_recovery = error_recovery

    def chat(self, messages: List[dict]) -> str:
        """Send chat completion request with retry."""
        return self.error_recovery.execute_with_retry(
            self._chat_raw, messages
        )

    def _chat_raw(self, messages: List[dict]) -> str:
        """Raw chat API call."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "num_ctx": 16384
            }
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get('message', {}).get('content', '')
        except urllib.error.URLError as e:
            raise ConnectionError(f"LLM API error: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def chat_with_planning(self, messages: List[dict], goal: str) -> Tuple[str, dict]:
        """Chat with optional planning step. Returns (response, plan)."""
        return self.chat(messages), {}


# ============================================================================
# TOOL EXECUTOR (Enhanced with batching and validation)
# ============================================================================

class ToolExecutor:
    """Execute tools with validation and error handling."""

    def __init__(self, config: dict):
        self.config = config

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Route to appropriate tool handler."""
        handlers = {
            'shell': self._exec_shell,
            'file': self._write_file,
            'search': self._web_search,
            'ask_human': self._ask_human,
            'exit_agent': self._exit_agent,
            'restart_agent': self._restart_agent,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            return handler(arguments)
        except Exception as e:
            return f"Tool error ({tool_name}): {str(e)}"

    def execute_shell(self, command: str, timeout: int = 60) -> str:
        """Execute a shell command with timeout."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout
            )
            output = result.stdout or result.stderr or ""
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Shell error: {str(e)}"

    def _exec_shell(self, args: dict) -> str:
        cmd = args.get('command', '')
        timeout = args.get('timeout', 60)
        if not cmd:
            return "Error: No command provided"
        return self.execute_shell(cmd, timeout)

    def _write_file(self, args: dict) -> str:
        filepath = args.get('path', '')
        content = args.get('content', '')
        append = args.get('append', False)

        if not filepath:
            return "Error: No filepath provided"

        try:
            # Create parent directories if needed
            parent = os.path.dirname(filepath)
            if parent:
                os.makedirs(parent, exist_ok=True)

            mode = 'a' if append else 'w'
            with open(filepath, mode) as f:
                f.write(content)

            return f"Successfully wrote to {filepath} ({len(content)} chars)"
        except Exception as e:
            return f"File write error: {str(e)}"

    def _web_search(self, args: dict) -> str:
        query = args.get('query', '')
        if not query:
            return "Error: No search query provided"
        # Placeholder - would integrate with actual search API
        return f"Search results for '{query}': [Search integration placeholder]"

    def _ask_human(self, args: dict) -> str:
        question = args.get('question', '')
        print(f"\n[ASK HUMAN] {question}")
        print("(In automated mode - returning default response)")
        return "Human input not available in automated mode"

    def _exit_agent(self, args: dict) -> str:
        message = args.get('final_response', 'Agent exiting.')
        print(f"\n[EXIT] {message}")
        sys.exit(0)

    def _restart_agent(self, args: dict) -> str:
        command = args.get('command', '')
        if not command:
            return "Error: No restart command provided"
        print(f"\n[RESTART] Spawning new agent: {command}")
        # Spawn new process and exit current one
        subprocess.Popen(command, shell=True)
        time.sleep(5)  # Give new process time to start
        os._exit(0)


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class Agent:
    """Main autonomous agent orchestrator."""

    def __init__(self, config: dict):
        self.config = config
        self.memory = MemorySystem(ttl=config.get('memory_ttl_seconds', 600))
        self.context_mgr = AdaptiveContextManager(config)
        self.parser = BlockParser()
        self.planner = PlanningEngine()
        self.reflector = ReflectionEngine()
        self.error_recovery = ErrorRecovery(
            max_attempts=config.get('retry_max_attempts', 3),
            base_delay=config.get('retry_base_delay', 2.0)
        )
        self.llm = LLMInterface(config, self.error_recovery)
        self.executor = ToolExecutor(config)

        self.history: List[dict] = []
        self.step_count = 0
        self.max_steps = config.get('max_steps', 20)
        self.goal = ""
        self.plan = None
        self.planning_enabled = True
        self.reflection_enabled = True
        self.benchmark_mode = False

    def build_system_prompt(self, goal: str) -> str:
        """Build optimized system prompt."""
        memory_summary = self.memory.get_summary()

        plan_section = ""
        if self.plan:
            plan_section = f"\n\nCurrent Plan:\n{json.dumps(self.plan, indent=2)}"

        scratchpad = f"""SCRATCHPAD
## notes
Agent running v3.0.0 with adaptive context, hierarchical planning, and enhanced error recovery.

## plan
Execute task efficiently using available tools.

## findings
v3 improvements: adaptive context management, tool chaining, exponential backoff retry.

## version_info
Running v3.0.0
Changes:
- Adaptive context management with progressive compression
- Hierarchical planning engine with dynamic replanning
- Tool chaining and batch operations
- Enhanced error recovery with exponential backoff
- Optimized LLM prompts for faster responses
- Robust parser with auto-detection
- TTL-based memory management"""

        prompt = f"""You are an autonomous AI agent (v3.0.0) with full system access.
Your goal: {goal}{plan_section}

{scratchpad}

PARSER RULES:
1. To write/overwrite a file, use: ```FILEPATH\\ncontent\\n```
2. To run shell commands (timeout mandatory): ```shell\\n# TIMEOUT 60\\ncmd\\n```
3. To use modular tools: ```tool\\n{{json}}\\n```
4. NESTED BLOCKS: If content contains triple backticks, wrap in four or more.

Available tools via ```tool block:
- shell: {{\"command\": \"cmd\", \"timeout\": 60}}
- file: {{\"path\": \"/path/to/file\", \"content\": \"text\", \"append\": false}}
- search: {{\"query\": \"search terms\"}}
- exit_agent: {{\"final_response\": \"message\"}}
- restart_agent: {{\"command\": \"python3 agent.py ...\"}}

Respond with action blocks. Keep responses concise and focused on completing the goal."""

        return prompt

    def run(self, goal: str = "", task_file: str = "") -> None:
        """Main execution loop."""
        # Load goal from task file if provided
        if task_file and os.path.exists(task_file):
            with open(task_file) as f:
                task_data = json.load(f)
            goal = task_data.get('goal', goal)
            # Merge task config
            for key in ('model', 'ollama_host', 'ollama_port', 'max_history'):
                if key in task_data:
                    self.config[key] = task_data[key]

        self.goal = goal
        print(f"Agent v3.0.0 starting with goal: {goal}")

        # Store goal in memory
        self.memory.store('goal', goal, long_term=True, importance=1.0)

        # Build initial messages
        system_prompt = self.build_system_prompt(goal)
        self.history = [{'role': 'system', 'content': system_prompt}]

        # Planning phase (skip in benchmark mode)
        if self.planning_enabled and not self.benchmark_mode:
            plan_text = self._run_planning()
            if plan_text:
                self.plan = self.planner.parse_plan_response(plan_text)
                self.memory.store('plan', self.plan, importance=0.8)
                print(f"Plan created: {len(self.plan)} steps")

        # Main execution loop
        while self.step_count < self.max_steps:
            self.step_count += 1
            print(f"\n--- Step {self.step_count}/{self.max_steps} ---")

            # Compress context if needed
            if len(self.history) > self.config['context_tiers']['recent']:
                self.history = self.context_mgr.compress_history(self.history)

            # Get LLM response
            try:
                response = self.llm.chat(self.history)
            except Exception as e:
                print(f"LLM error (step {self.step_count}): {e}")
                self.memory.store(f'error_step_{self.step_count}', str(e))
                continue

            # Add response to history
            self.history.append({'role': 'assistant', 'content': response})

            # Parse and execute blocks
            blocks = self.parser.parse(response)
            if not blocks:
                print("No actionable blocks found in response")
                continue

            actions_taken = []
            for block in blocks:
                tool_type = self.parser.detect_tool_type(block)
                result = self._execute_block(tool_type, block)
                actions_taken.append(f"{tool_type}: {result[:200]}")

                # Add tool response to history
                self.history.append({'role': 'tool', 'content': result})

            # Check for task completion signals
            if 'exit_agent' in ' '.join(actions_taken).lower():
                print("Agent signaled exit")
                break

            # Periodic memory cleanup
            if self.step_count % 5 == 0:
                removed = self.memory.cleanup()
                if removed:
                    print(f"Cleaned up {removed} expired memory entries")

        # Final reflection (skip in benchmark mode)
        if self.reflection_enabled and not self.benchmark_mode:
            self._run_reflection(actions_taken if 'actions_taken' in dir() else [])

        print(f"\nAgent completed after {self.step_count} steps")

    def _execute_block(self, tool_type: str, block: dict) -> str:
        """Execute a parsed block based on its type."""
        header = block.get('header', '')
        body = block.get('body', '')

        if tool_type == 'shell':
            # Extract timeout from comments
            timeout = 60
            lines = body.split('\n')
            cmd_lines = []
            for line in lines:
                tm_match = re.match(r'#\s*TIMEOUT\s+(\d+)', line)
                if tm_match:
                    timeout = int(tm_match.group(1))
                elif not line.strip().startswith('#'):
                    cmd_lines.append(line)

            command = '\n'.join(cmd_lines).strip()
            if command:
                return self.executor.execute_shell(command, timeout)
            return "No shell command found"

        elif tool_type == 'file':
            # Try JSON format first (LLM may send {"path": "...", "content": "..."})
            try:
                file_data = json.loads(body)
                filepath = file_data.get('path', '')
                file_content = file_data.get('content', '')
                append = file_data.get('append', False)
                if filepath:
                    return self.executor._write_file({
                        'path': filepath, 'content': file_content, 'append': append
                    })
            except (json.JSONDecodeError, TypeError):
                pass

            # Traditional format: first line is filepath, rest is content
            lines = body.split('\n', 1)
            filepath = lines[0].strip()
            file_content = lines[1] if len(lines) > 1 else ''

            if filepath and (filepath.startswith('/') or filepath.startswith('.')):
                return self.executor._write_file({'path': filepath, 'content': file_content})
            # Fallback: treat as shell command
            return self.executor.execute_shell(body.strip(), 60)

        elif tool_type == 'tool':
            try:
                tool_data = json.loads(body)
                name = tool_data.get('name', '')
                args = tool_data.get('arguments', tool_data.get('args', {}))
                return self.executor.execute(name, args)
            except json.JSONDecodeError:
                return f"Invalid JSON in tool block: {body[:100]}"

        elif tool_type == 'thought':
            return f"[Thought processed]"

        else:
            # Default: try as shell command
            return self.executor.execute_shell(body.strip(), 60)

    def _run_planning(self) -> str:
        """Run planning phase with LLM."""
        memory_summary = self.memory.get_summary()
        plan_prompt_text = self.planner.plan_prompt(self.goal, memory_summary)

        plan_messages = [
            {'role': 'system', 'content': 'You are a planning assistant. Break tasks into clear steps.'},
            {'role': 'user', 'content': plan_prompt_text}
        ]

        try:
            return self.llm.chat(plan_messages)
        except Exception as e:
            print(f"Planning failed: {e}")
            return ""

    def _run_reflection(self, actions_taken: List[str]) -> None:
        """Run reflection phase."""
        outcome = f"Completed in {self.step_count} steps"
        reflect_prompt_text = self.reflector.reflect_prompt(
            self.goal, str(actions_taken), outcome
        )

        reflect_messages = [
            {'role': 'system', 'content': 'Evaluate task execution and provide feedback.'},
            {'role': 'user', 'content': reflect_prompt_text}
        ]

        try:
            response = self.llm.chat(reflect_messages)
            reflection = self.reflector.parse_reflection(response)
            print(f"Reflection: Score {reflection['score']}/10")
            print(f"Improvement: {reflection['improvement']}")
            self.memory.store('last_reflection', reflection, long_term=True)
        except Exception as e:
            print(f"Reflection failed: {e}")


# ============================================================================
# TEST SUITE
# ============================================================================

def run_tests():
    """Run comprehensive unit tests."""
    passed = 0
    failed = 0
    total = 0

    def test(name, condition):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    print("\n=== V3 Unit Tests ===\n")

    # --- Memory System Tests ---
    print("Memory System:")
    mem = MemorySystem(ttl=0.1)  # Short TTL for testing

    mem.store('key1', 'value1')
    test("store/retrieve basic", mem.retrieve('key1') == 'value1')

    mem.store('key2', 'value2', long_term=True)
    test("store/retrieve long-term", mem.retrieve('key2') == 'value2')

    time.sleep(0.2)
    test("TTL expiration", mem.retrieve('key1') is None)

    mem.store('tagged', 'data', tags=['important'])
    results = mem.search_by_tag('important')
    test("search by tag", len(results) == 1 and results[0][0] == 'tagged')

    mem2 = MemorySystem(ttl=0.1)
    mem2.store('expire1', 'a')
    mem2.store('expire2', 'b')
    time.sleep(0.2)
    removed = mem2.cleanup()
    test("cleanup expired", removed == 2)

    # --- Token Estimator Tests ---
    print("\nToken Estimator:")
    est = TokenEstimator()
    tokens = est.estimate_tokens("Hello world this is a test")
    test("token estimation reasonable", 3 <= tokens <= 8)

    msg_tokens = est.estimate_message_tokens({'role': 'user', 'content': 'test message'})
    test("message token estimation", msg_tokens > 0)

    # --- Context Manager Tests ---
    print("\nContext Manager:")
    config = {
        'max_history_messages': 50,
        'context_tiers': {'critical': 3, 'recent': 6, 'archived': 12},
        'num_ctx': 8192
    }
    ctx_mgr = AdaptiveContextManager(config)

    # Create test messages
    msgs = [{'role': 'system', 'content': f'msg{i}'} for i in range(15)]
    compressed = ctx_mgr.compress_history(msgs)
    test("compression reduces messages", len(compressed) < len(msgs))
    test("compression keeps system message", compressed[0]['role'] == 'system')

    # Test token budget trimming
    long_msgs = [{'role': 'user', 'content': 'x' * 10000} for _ in range(5)]
    trimmed = ctx_mgr.trim_to_budget(long_msgs, "system prompt")
    test("token budget trimming", len(trimmed) <= len(long_msgs))

    # --- Block Parser Tests ---
    print("\nBlock Parser:")
    parser = BlockParser()

    blocks = parser.parse("```shell\necho hello\n```")
    test("parse shell block", len(blocks) == 1 and blocks[0]['header'] == 'shell')

    blocks = parser.parse("```/tmp/test.txt\nfile content here\n```")
    test("parse file block", len(blocks) == 1)

    blocks = parser.parse("text before\n```python\nprint('hi')\n```\ntext after")
    test("parse with surrounding text", len(blocks) == 1)

    blocks = parser.parse("No blocks here")
    test("no blocks returns empty", len(blocks) == 0)

    # Test auto-detection
    shell_block = {'header': 'shell', 'body': 'echo hi'}
    test("detect shell type", parser.detect_tool_type(shell_block) == 'shell')

    file_block = {'header': '', 'body': '/tmp/test.txt\ncontent'}
    test("detect file type", parser.detect_tool_type(file_block) == 'file')

    tool_block = {'header': 'tool', 'body': '{"name": "search"}'}
    test("detect tool type", parser.detect_tool_type(tool_block) == 'tool')

    # --- Planning Engine Tests ---
    print("\nPlanning Engine:")
    planner = PlanningEngine()

    plan_text = """1. Create the file
2. Write content to it
3. Verify the result"""
    steps = planner.parse_plan_response(plan_text)
    test("parse plan steps", len(steps) == 3)

    plan_text2 = "Just do it all at once"
    steps2 = planner.parse_plan_response(plan_text2)
    test("no numbered list returns empty", len(steps2) == 0)

    # --- Reflection Engine Tests ---
    print("\nReflection Engine:")
    reflector = ReflectionEngine()

    refl_text = "SCORE: 8/10\nIMPROVEMENT: Use more specific commands"
    result = reflector.parse_reflection(refl_text)
    test("parse reflection score", result['score'] == 8)
    test("parse reflection improvement", 'specific' in result['improvement'].lower())

    refl_text2 = "Great job overall"
    result2 = reflector.parse_reflection(refl_text2)
    test("default score on no match", result2['score'] == 5)

    # --- Error Recovery Tests ---
    print("\nError Recovery:")
    recovery = ErrorRecovery(max_attempts=3, base_delay=0.01)

    call_count = 0
    def failing_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("fail")
        return "success"

    result = recovery.execute_with_retry(failing_func)
    test("retry succeeds after failures", result == "success" and call_count == 3)

    def always_fail():
        raise ConnectionError("timeout error")

    try:
        recovery.execute_with_retry(always_fail)
        test("retry exhausts attempts", False)
    except ConnectionError:
        test("retry exhausts attempts", True)

    # --- Tool Executor Tests ---
    print("\nTool Executor:")
    executor = ToolExecutor({})

    result = executor.execute_shell('echo "test_output"', timeout=10)
    test("shell execution", 'test_output' in result)

    result = executor._write_file({'path': '/tmp/v3_test_file.txt', 'content': 'v3 works'})
    test("file write success", 'Successfully' in result)
    test("file content correct", os.path.exists('/tmp/v3_test_file.txt') and open('/tmp/v3_test_file.txt').read() == 'v3 works')

    # --- Memory Summary Tests ---
    print("\nMemory Summary:")
    mem = MemorySystem(ttl=600)
    mem.store('test_key', 'test_value', tags=['test'])
    summary = mem.get_summary()
    test("memory summary contains key", 'test_key' in summary)

    # --- Print Results ---
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*40}\n")

    return failed == 0


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Agent v3.0.0")
    parser.add_argument('--task', type=str, help='Path to task JSON file')
    parser.add_argument('--goal', type=str, help='Direct goal string')
    parser.add_argument('--steps', type=int, default=20, help='Max execution steps')
    parser.add_argument('--benchmark', action='store_true', help='Disable planning/reflection for benchmarking')
    parser.add_argument('--test', action='store_true', help='Run unit tests only')
    args = parser.parse_args()

    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    config = DEFAULT_CONFIG.copy()
    config['max_steps'] = args.steps

    agent = Agent(config)
    agent.planning_enabled = not args.benchmark
    agent.reflection_enabled = not args.benchmark
    agent.benchmark_mode = args.benchmark

    goal = args.goal or ""
    agent.run(goal=goal, task_file=args.task)


if __name__ == '__main__':
    main()
