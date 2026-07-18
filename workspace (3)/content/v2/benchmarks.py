#!/usr/bin/env python3
"""Live benchmark suite for v2 agent."""

import json
import os
import sys
import subprocess
import time
import tempfile
import shutil

SCRIPT_DIR = "/content/v2"
TASK_TEMPLATE = {
    "goal": "__GOAL__",
    "system_prompt": """You are an autonomous agent system operating with full root access.
Use the scratchpad for persistent notes. Execute tasks step by step.""",
    "model": "qwen3.6:27b-Q8_0",
    "max_history": 50,
    "ollama_host": "87.171.205.7",
    "ollama_port": 9090,
    "log_file_path": "bench_history.json",
}

BENCHMARKS = [
    {
        "name": "file_creation",
        "goal": 'Create a file at /tmp/bench_test1.txt containing exactly: "Hello from v2 agent"',
        "verify": lambda tmpdir: os.path.exists("/tmp/bench_test1.txt") and 
                    open("/tmp/bench_test1.txt").read().strip() == "Hello from v2 agent",
        "steps": 2,
    },
    {
        "name": "shell_execution",
        "goal": 'Run a shell command to create /tmp/bench_test2.txt with content "Shell works"',
        "verify": lambda tmpdir: os.path.exists("/tmp/bench_test2.txt") and 
                    open("/tmp/bench_test2.txt").read().strip() == "Shell works",
        "steps": 2,
    },
    {
        "name": "nested_directories",
        "goal": 'Create directory /tmp/bench_nested/a/b/c and a file inside: /tmp/bench_nested/a/b/c/deep.txt with content "Deep nesting"',
        "verify": lambda tmpdir: os.path.exists("/tmp/bench_nested/a/b/c/deep.txt") and 
                    open("/tmp/bench_nested/a/b/c/deep.txt").read().strip() == "Deep nesting",
        "steps": 2,
    },
    {
        "name": "json_file",
        "goal": 'Create /tmp/bench_test4.json with valid JSON: {"status": "success", "count": 42}',
        "verify": lambda tmpdir: os.path.exists("/tmp/bench_test4.json") and 
                    json.loads(open("/tmp/bench_test4.json").read()) == {"status": "success", "count": 42},
        "steps": 2,
    },
    {
        "name": "multi_file_creation",
        "goal": 'Create three files: /tmp/bench_f1.txt with "one", /tmp/bench_f2.txt with "two", /tmp/bench_f3.txt with "three"',
        "verify": lambda tmpdir: (
            os.path.exists("/tmp/bench_f1.txt") and open("/tmp/bench_f1.txt").read().strip() == "one" and
            os.path.exists("/tmp/bench_f2.txt") and open("/tmp/bench_f2.txt").read().strip() == "two" and
            os.path.exists("/tmp/bench_f3.txt") and open("/tmp/bench_f3.txt").read().strip() == "three"
        ),
        "steps": 2,
    },
    {
        "name": "shell_computation",
        "goal": 'Use shell to compute: echo $((7 * 8)) > /tmp/bench_calc.txt and verify it contains 56',
        "verify": lambda tmpdir: os.path.exists("/tmp/bench_calc.txt") and 
                    open("/tmp/bench_calc.txt").read().strip() == "56",
        "steps": 2,
    },
    {
        "name": "file_listing",
        "goal": 'Create /tmp/bench_dir/ with files a.txt, b.txt, c.txt inside it. Then use shell to list them.',
        "verify": lambda tmpdir: all(
            os.path.exists(f"/tmp/bench_dir/{f}.txt") for f in ["a", "b", "c"]
        ),
        "steps": 2,
    },
    {
        "name": "python_script_execution",
        "goal": 'Create a Python script /tmp/bench_py.py that prints "Python works" and run it with shell to verify.',
        "verify": lambda tmpdir: os.path.exists("/tmp/bench_py.py"),
        "steps": 2,
    },
    {
        "name": "read_and_modify",
        "goal": 'Create /tmp/bench_readme.txt with "Version 1". Then use shell to append " - Updated" to it.',
        "verify": lambda tmpdir: os.path.exists("/tmp/bench_readme.txt") and 
                    "Updated" in open("/tmp/bench_readme.txt").read(),
        "steps": 2,
    },
    {
        "name": "complex_workflow",
        "goal": 'Create /tmp/bench_work/ directory. Inside it create config.json with {"mode": "test"}, then create data.txt with "sample data". Finally list all files in the directory.',
        "verify": lambda tmpdir: (
            os.path.exists("/tmp/bench_work/config.json") and
            json.loads(open("/tmp/bench_work/config.json").read()) == {"mode": "test"} and
            os.path.exists("/tmp/bench_work/data.txt") and
            open("/tmp/bench_work/data.txt").read().strip() == "sample data"
        ),
        "steps": 2,
    },
]


def run_benchmark(bench: dict) -> dict:
    """Run a single benchmark by spawning the agent."""
    name = bench["name"]
    print(f"\n{'='*50}")
    print(f"BENCHMARK: {name}")
    print(f"Goal: {bench['goal'][:100]}...")
    
    # Create temp task file
    task_data = {**TASK_TEMPLATE, "goal": bench["goal"]}
    task_file = f"/tmp/bench_task_{name}.json"
    log_file = f"/tmp/bench_log_{name}.json"
    task_data["log_file_path"] = log_file
    
    with open(task_file, 'w') as f:
        json.dump(task_data, f)
    
    # Run agent
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "agent.py"), 
             "--task", task_file, "--steps", str(bench["steps"]), "--benchmark"],
            capture_output=True, text=True, timeout=300
        )
        
        elapsed = time.time() - start
        
        # Check verification
        try:
            passed = bench["verify"](SCRIPT_DIR)
        except Exception as e:
            passed = False
            print(f"  Verification error: {e}")
        
        if passed:
            print(f"  ✓ PASSED ({elapsed:.1f}s)")
        else:
            print(f"  ✗ FAILED ({elapsed:.1f}s)")
            # Show agent output for debugging
            print(f"  Agent stdout (last 500 chars):\n{proc.stdout[-500:]}")
            if proc.stderr:
                print(f"  Agent stderr:\n{proc.stderr[-300:]}")
        
        return {
            "name": name,
            "passed": passed,
            "elapsed": round(elapsed, 1),
            "exit_code": proc.returncode,
        }
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT (>300s)")
        return {"name": name, "passed": False, "elapsed": 300, "exit_code": -1}
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return {"name": name, "passed": False, "elapsed": 0, "exit_code": -2}


def main():
    print("="*60)
    print("LIVE BENCHMARK SUITE - v2 Agent")
    print(f"Running {len(BENCHMARKS)} benchmarks")
    print("="*60)
    
    results = []
    for bench in BENCHMARKS:
        result = run_benchmark(bench)
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_time = sum(r["elapsed"] for r in results) / total if total else 0
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} benchmarks passed")
    print(f"Average time: {avg_time:.1f}s")
    print(f"{'='*60}")
    
    for r in results:
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {r['name']}: {r['elapsed']}s")
    
    return passed >= 10  # Minimum 10 must pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
