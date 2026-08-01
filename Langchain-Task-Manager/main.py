from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import sqlite3

load_dotenv()

con = sqlite3.connect("tasksdb.sqlite3", check_same_thread=False)
cursor = con.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS todos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        todo TEXT NOT NULL,
        iscompleted BOOLEAN NOT NULL DEFAULT 0
    )
""")
con.commit()

@tool
def add_todo(todo: str):
    """
    Add a new todo in todos table
    """
    cursor.execute("INSERT INTO todos (todo, iscompleted) VALUES (?, ?)", (todo, False))
    con.commit()
    return cursor.lastrowid

@tool
def get_all_todos():
    """
    Get all todos from todos table
    """
    cursor.execute("SELECT id, todo, iscompleted FROM todos ORDER BY id")
    todos = cursor.fetchall()
    return todos

@tool
def get_todo_by_id(todo_id: str):
    """
    Get todo item from todos table based on todo_id
    """
    cursor.execute("SELECT id, todo, iscompleted FROM todos WHERE id=?", (todo_id,))
    return cursor.fetchone()

@tool
def set_todo_completed(todo_id, completed=True):
    """
    Update the status of a todo item to True or False in todos table based on todo task completion
    """
    cursor.execute("UPDATE todos SET iscompleted=? WHERE id=?", (completed, todo_id))
    con.commit()

@tool
def delete_todo(todo_id):
    """
    Delete a todo item from todos table based on todo
    """
    cursor.execute("DELETE FROM todos WHERE id=?", (todo_id,))
    con.commit()

llm = ChatOpenRouter(
    model="gpt-4o-mini"
)

agent = create_agent(
    model=llm,
    system_prompt="You are an AI todo assistant who helps in managing tasks for the user. You call tools to perform different actions. You are strictly managing the todos and do not answer queries related to anything else.",
    tools=[add_todo, get_all_todos, get_todo_by_id, set_todo_completed, delete_todo]
)

while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    })
    print(response["messages"][-1].content)