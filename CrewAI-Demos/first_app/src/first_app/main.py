#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from first_app.crew import FirstApp
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def load_transcript(filename: str) -> str:
    """
    Load a meeting transcript from the transcripts folder
    """
    base_dir = Path(__file__).resolve().parent
    trans_dir = base_dir / "transcripts"
    path = trans_dir / filename
    
    if not path.exists():
        raise FileNotFoundError("File Not found")
    return path.read_text(encoding="utf-8")


def run():
    """
    Run the crew.
    """
    inputs = {
        'transcript': load_transcript('meeting_trans.txt')
    }

    try:
        FirstApp().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        FirstApp().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        FirstApp().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        FirstApp().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = FirstApp().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
