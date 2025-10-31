import asyncio
import json
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

# Local imports
from tasks.stable_numerics.tools import write_file_handler, run_tests_handler
from tasks.stable_numerics.grader import grade_submission

MAX_TOKENS = 3000


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in ["max_tokens", "tool_use", "end_turn"], (
            f"unsupported stop_reason {response.stop_reason}"
        )
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        if isinstance(tool_input, dict) and "expression" in tool_input:
                            if verbose:
                                print("\nInput:")
                                print("```")
                                for line in tool_input["expression"].split("\n"):
                                    print(f"{line}")
                                print("```")
                            result = handler(tool_input["expression"])
                            if verbose:
                                print("\nOutput:")
                                print("```")
                                print(result)
                                print("```")
                        else:
                            result = {"result": None, "error": f"Invalid arguments: expected dict with 'expression', got {type(tool_input)}"}
                    elif tool_name == "write_file":
                        if isinstance(tool_input, dict) and "path" in tool_input and "content" in tool_input:
                            result = handler(tool_input["path"], tool_input["content"])
                            if verbose:
                                print(f"\nWrite file result: {result}")
                        else:
                            result = {"written": False, "error": f"Invalid arguments: expected dict with 'path' and 'content', got {type(tool_input)}"}
                    elif tool_name == "run_tests":
                        if isinstance(tool_input, dict) and "kind" in tool_input:
                            seed = tool_input.get("seed", 42)
                            result = handler(tool_input["kind"], seed)
                            if verbose:
                                print(f"\nTest result: {result}")
                        else:
                            result = {"passed": False, "summary": f"Invalid arguments: expected dict with 'kind', got {type(tool_input)}"}
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    expected_answer: Any,
    seed: int,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} (seed={seed}) {'=' * 20}")

    # Reset user_impl.py to initial state before each test
    user_impl_path = Path(__file__).parent / "tasks" / "stable_numerics" / "user_impl.py"
    if user_impl_path.exists():
        # Restore initial template
        initial_content = """import numpy as np

# TODO: Implement numerically stable log_softmax and cross_entropy functions


def log_softmax(x: np.ndarray) -> np.ndarray:
    \"\"\"
    Numerically stable log-softmax function.
    
    Args:
        x: 1D or 2D array (batch, dim). Supports float32 and float16.
    
    Returns:
        Log-softmax of x, same shape and dtype as input.
    \"\"\"
    raise NotImplementedError("Implement this function")


def cross_entropy(logits: np.ndarray, target_idx: np.ndarray | int) -> np.ndarray | float:
    \"\"\"
    Numerically stable cross-entropy loss.
    
    Args:
        logits: 1D or 2D array (batch, dim). Supports float32 and float16.
        target_idx: int (for 1D logits) or 1D array of indices (for 2D logits).
    
    Returns:
        Cross-entropy loss (scalar for 1D, float or array for 2D).
    \"\"\"
    raise NotImplementedError("Implement this function")
"""
        user_impl_path.write_text(initial_content, encoding="utf-8")

    try:
        result = await run_agent_loop(
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            max_steps=20,  # Increased for more complex task
            verbose=verbose,
        )

        # Grade submission using grader (independent of agent result)
        success = grade_submission(seed=seed)

        if success:
            print(f"✓ Run {run_id}: PASSED")
        else:
            print(f"✗ Run {run_id}: FAILED")
            if result:
                print(f"  (Agent submitted: {result})")

        return run_id, success, result
    except Exception as e:
        # If an error occurs, mark the run as failed
        print(f"✗ Run {run_id}: ERROR - {type(e).__name__}: {str(e)}")
        return run_id, False, None


async def main(concurrent: bool = True):
    # Load prompt from task file
    prompt_path = Path(__file__).parent / "tasks" / "stable_numerics" / "prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")

    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file. Only allowed path: tasks/stable_numerics/user_impl.py",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (must be tasks/stable_numerics/user_impl.py)",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "run_tests",
            "description": "Run tests (visible or full). Use 'visible' during development, 'full' for final verification.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["visible", "full"],
                        "description": "Test kind: 'visible' for visible tests only, 'full' for all tests",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for test generation (optional, defaults to 42)",
                    },
                },
                "required": ["kind"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer after successfully passing all tests",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "write_file": lambda path, content: write_file_handler(path, content),
        "run_tests": lambda kind, seed=42: run_tests_handler(kind, seed),
        "submit_answer": submit_answer_tool,
    }

    # Run the test multiple times with different seeds
    num_runs = 10
    base_seed = 42
    expected_answer = "OK"  # Not used for grading, only for logging

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines with different seeds
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            expected_answer=expected_answer,
            seed=base_seed + i,  # Different seed for each run
            verbose=False,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                # If an error occurs, mark the run as failed
                print(f"✗ Task failed with error: {type(e).__name__}: {str(e)}")
                pass
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(success for _, success, _ in results)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=True))

