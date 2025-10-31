# RL-env

RL task for testing LLM capabilities on numerically stable implementations.

## Setup

1. Set up `ANTHROPIC_API_KEY` environment variable:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Run the task:
   ```bash
   uv run main.py
   ```

The script will run the task multiple times (default: 10 runs) with different seeds and report the pass rate.

## API Configuration

```bash
export API_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_api_key_here
uv run main.py
```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.
