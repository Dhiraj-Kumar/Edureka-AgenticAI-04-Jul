from mcp.server import MCPServer
import sqlite3
import uuid
from pathlib import Path

DB_NAME=str(Path(__file__).parent/"tasks.db")

mcp = MCPServer("TaskPlanner")

def getConnection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = getConnection()
    cursor = conn.cursor()
    
    cursor.execute("""
    Create Table if not exists tasks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        priority TEXT NOT NULL,
        deadline TEXT NOT NULL,
        status TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

init_db()

@mcp.tool()
def createTask(title: str, description: str, priority: str, deadline: str, status: str):
    """
    Create a new task and saves it in SQLite DB.
    Status can be either completed or pending. Default is pending
    """
    conn = getConnection()
    cursor = conn.cursor()
    task_id = str(uuid.uuid4())
    cursor.execute("""
    insert into tasks (id, title, description, priority, deadline, status) values (?,?,?,?,?,?)
    """, (task_id, title, description, priority, deadline, status))
    conn.commit()
    conn.close()
    return "Task Created Successfully"

@mcp.tool()
def deleteTask(id: str):
    """
    Deletes a task from DB.
    """
    conn = getConnection()
    cursor = conn.cursor()
    
    cursor.execute("Select * from tasks where id=?", (id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return f"Task with ID {id} not found."
    
    cursor.execute("Delete from tasks where id=?", (id,))
    
    conn.commit()
    conn.close()
    return "Task Deleted Successfully"

@mcp.tool()
def getTask(id: str):
    """
    Get a task from DB based on ID.
    """
    conn = getConnection()
    cursor = conn.cursor()
    
    cursor.execute("Select * from tasks where id=?", (id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return f"Task with ID {id} not found."
    
    cols = [d[0] for d in cursor.description]
    task = dict(zip(cols, row))
    
    conn.close()
    return task

@mcp.tool()
def getAllTasks(x=None):
    """
    Returns all tasks
    """
    conn = getConnection()
    cursor = conn.cursor()
    
    cursor.execute("Select title, description, priority, deadline, status from tasks")
    
    cols = [d[0] for d in cursor.description]
    tasks = [dict(zip(cols, row)) for row in cursor.fetchall()]
    
    conn.close()
    return tasks

if __name__ == "__main__":
    mcp.run()