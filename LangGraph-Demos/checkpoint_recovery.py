from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict
import sqlite3

payment_should_fail = True
courier_should_fail = True

class OrderState(TypedDict):
    order_id: str
    done: list[str]

def validate_order(state: OrderState):
    print("RUN validate order")
    return {"done": state["done"]+["validate_order"]}

def reserve_stock(state: OrderState):
    print("RUN reserve stock")
    return {"done": state["done"]+["reserve_stock"]}

def charge_payment(state: OrderState):
    global payment_should_fail
    if payment_should_fail:
        payment_should_fail = False
        raise RuntimeError("Payment gateway timed out")
    print("RUN Charge Payment")
    return {"done": state["done"]+["charge_payment"]}

def book_courier(state: OrderState):
    global courier_should_fail
    if courier_should_fail:
        courier_should_fail = False
        raise RuntimeError("Courier API returned 503")
    print("RUN Book Courier")
    return {"done": state["done"]+["book_courier"]}

def send_confirmation(state: OrderState):
    print("RUN Send Confirmation")
    return {"done": state["done"]+["send_confirmation"]}

builder = StateGraph(OrderState)

builder.add_node("validate_order", validate_order)
builder.add_node("reserve_stock", reserve_stock)
builder.add_node("charge_payment", charge_payment)
builder.add_node("book_courier", book_courier)
builder.add_node("send_confirmation", send_confirmation)

builder.add_edge(START, "validate_order")
builder.add_edge("validate_order", "reserve_stock")
builder.add_edge("reserve_stock", "charge_payment")
builder.add_edge("charge_payment", "book_courier")
builder.add_edge("book_courier", "send_confirmation")
builder.add_edge("send_confirmation", END)

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(conn))

config = {"configurable": {"thread_id": "order-001"}}
payload = {"order_id": "order-001", "done": []}
attempt = 0

while True:
    attempt+=1
    print(f"\nAttempt No. {attempt}")
    try:
        final = graph.invoke(payload, config)
        break
    except RuntimeError as e:
        snapshot = graph.get_state(config)
        print(f"CRASH {e}")
        print(f"Data in SQLite: {snapshot.values['done']}")
        print(f"Resume Point: {snapshot.next[0]}")
        payload=None

print(f"Completed in {attempt} attempts")
print(f"Steps executed once each {final['done']}")