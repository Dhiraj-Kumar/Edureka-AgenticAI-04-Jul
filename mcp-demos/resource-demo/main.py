import json
from pathlib import Path
from mcp.server import MCPServer

mcp = MCPServer("course-tracker")

DATA_FILE = Path(__file__).parent / "courses.json"

def load_courses() -> dict:
    return json.loads(DATA_FILE.read_text())

def save_courses(data: dict) -> None:
    DATA_FILE.write_text(json.dumps(data, indent=2))

# Define Resources
@mcp.resource("tracker://courses")
def list_courses() -> str:
    """List of courses"""
    courses = load_courses()
    lines = [f"{cid}: {p['name']} ({p['status']})" for cid, p in courses.items()]
    return "\n".join(lines)

@mcp.tool()
def update_status(course_id: str, status: str) -> str:
    """Change a courses's status (persists tp projects.json)"""
    courses = load_courses()
    if course_id not in courses:
        raise ValueError(f"No course '{course_id}'")
    courses[course_id]["status"] = status
    save_courses(courses)
    return f"{course_id} -> {status}"

@mcp.prompt()
def status_report(course_id: str) -> str:
    """Draft a status update message for a course."""
    courses = load_courses()
    c = courses.get(course_id)
    if not c:
        raise ValueError(f"No course '{course_id}'")
    return (
        f"Write a on paragraph status update for stackholders about {c['name']}. Current status: {c['status']}. Hours logged: {c['hours']}. Owner: {c['owner']}. Keep it factual, no fillers."
    )


if __name__ == "__main__":
    mcp.run()