import sqlite3, os

os.makedirs("output", exist_ok=True)

conn = sqlite3.connect("shopease.db")
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    product TEXT,
    category TEXT,
    city TEXT,
    quantity INTEGER,
    price REAL
);
INSERT INTO orders (product, category, city, quantity, price) VALUES
('Laptop','Electronics','Mumbai',2,60000),
('Phone','Electronics','Delhi',5,30000),
('Desk','Furniture','Mumbai',3,8000),
('Chair','Furniture','Pune',10,2500),
('Notebook','Stationery','Delhi',50,50),
('Monitor','Electronics','Pune',4,15000);
""")
conn.commit()
conn.close()