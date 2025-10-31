"""
Tool definitions and handlers for the stable numerics task.
"""
from pathlib import Path
from typing import TypedDict, Literal
from tasks.stable_numerics.grader import run_visible_tests, run_full_tests

# Allowed path for write_file
ALLOWED_PATH = Path(__file__).parent / "user_impl.py"


class WriteFileResult(TypedDict):
    written: bool
    error: str | None


class RunTestsResult(TypedDict):
    passed: bool
    summary: str


def write_file_handler(path: str, content: str) -> WriteFileResult:
    """
    Write content to file. Only allows writing to user_impl.py.
    
    Args:
        path: File path (must be tasks/stable_numerics/user_impl.py or relative equivalent)
        content: Content to write
    
    Returns:
        Dictionary with 'written' (bool) and 'error' (str | None)
    """
    try:
        # Normalize and validate path
        target_path = Path(path).resolve()
        allowed_path = ALLOWED_PATH.resolve()
        
        # Check if path matches allowed path (allowing relative and absolute forms)
        if target_path != allowed_path:
            return {
                "written": False,
                "error": f"Permission denied: can only write to tasks/stable_numerics/user_impl.py, got {path}"
            }
        
        # Write file
        ALLOWED_PATH.write_text(content, encoding="utf-8")
        
        return {"written": True, "error": None}
        
    except Exception as e:
        return {"written": False, "error": str(e)}


def run_tests_handler(kind: Literal["visible", "full"], seed: int = 42) -> RunTestsResult:
    """
    Run tests (visible or full).
    
    Args:
        kind: "visible" for visible tests only, "full" for all tests
        seed: Random seed for test generation
    
    Returns:
        Dictionary with 'passed' (bool) and 'summary' (str)
    """
    try:
        if kind == "visible":
            result = run_visible_tests(seed)
            return {
                "passed": result["passed"],
                "summary": result["summary"]
            }
        elif kind == "full":
            result = run_full_tests(seed)
            return {
                "passed": result["passed"],
                "summary": result["summary"]
            }
        else:
            return {
                "passed": False,
                "summary": f"Invalid kind: {kind}. Must be 'visible' or 'full'"
            }
    except Exception as e:
        return {
            "passed": False,
            "summary": f"Error running tests: {str(e)}"
        }

