"""Sample SQLite database for demos and Streamlit Cloud."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    email TEXT,
    signup_date TEXT
);

CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT,
    city TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    stock INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    employee_id INTEGER,
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

SEED_SQL = """
DELETE FROM order_items;
DELETE FROM orders;
DELETE FROM products;
DELETE FROM employees;
DELETE FROM customers;

INSERT INTO customers (customer_id, name, city, email, signup_date) VALUES
(1, 'Ayesha Khan', 'Lahore', 'ayesha@example.com', '2024-01-12'),
(2, 'Bilal Ahmed', 'Karachi', 'bilal@example.com', '2024-02-03'),
(3, 'Sara Malik', 'Islamabad', 'sara@example.com', '2024-02-18'),
(4, 'Omar Farooq', 'Lahore', 'omar@example.com', '2024-03-01'),
(5, 'Nina Patel', 'Karachi', 'nina@example.com', '2024-03-22'),
(6, 'Hassan Raza', 'Lahore', 'hassan@example.com', '2024-04-09'),
(7, 'Fatima Noor', 'Multan', 'fatima@example.com', '2024-05-14'),
(8, 'James Lee', 'Islamabad', 'james@example.com', '2024-06-02');

INSERT INTO employees (employee_id, name, title, city) VALUES
(1, 'Ali Rehman', 'Sales Manager', 'Lahore'),
(2, 'Zara Siddiqui', 'Account Executive', 'Karachi'),
(3, 'Daniel Costa', 'Account Executive', 'Islamabad');

INSERT INTO products (product_id, name, category, unit_price, stock) VALUES
(1, 'Wireless Mouse', 'Electronics', 25.00, 120),
(2, 'USB-C Hub', 'Electronics', 45.00, 80),
(3, 'Office Chair', 'Furniture', 180.00, 35),
(4, 'Standing Desk', 'Furniture', 420.00, 18),
(5, 'Notebook Pack', 'Stationery', 12.50, 300),
(6, 'Whiteboard Markers', 'Stationery', 8.00, 250),
(7, 'Noise Cancelling Headphones', 'Electronics', 199.00, 40),
(8, 'Monitor 27in', 'Electronics', 280.00, 22);

INSERT INTO orders (order_id, customer_id, employee_id, order_date, status) VALUES
(1, 1, 1, '2025-11-05', 'completed'),
(2, 2, 2, '2025-11-12', 'completed'),
(3, 3, 3, '2025-12-01', 'completed'),
(4, 4, 1, '2025-12-18', 'completed'),
(5, 1, 1, '2026-01-08', 'completed'),
(6, 5, 2, '2026-01-15', 'completed'),
(7, 6, 1, '2026-02-02', 'completed'),
(8, 7, 2, '2026-02-20', 'shipped'),
(9, 8, 3, '2026-03-05', 'completed'),
(10, 4, 1, '2026-03-22', 'completed'),
(11, 2, 2, '2026-04-10', 'completed'),
(12, 1, 1, '2026-05-01', 'completed'),
(13, 6, 1, '2026-06-11', 'completed'),
(14, 5, 2, '2026-07-03', 'completed'),
(15, 3, 3, '2026-07-19', 'pending');

INSERT INTO order_items (order_item_id, order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 2, 25.00),
(2, 1, 5, 4, 12.50),
(3, 2, 7, 1, 199.00),
(4, 2, 2, 2, 45.00),
(5, 3, 3, 1, 180.00),
(6, 4, 4, 1, 420.00),
(7, 4, 6, 5, 8.00),
(8, 5, 8, 1, 280.00),
(9, 5, 1, 3, 25.00),
(10, 6, 2, 4, 45.00),
(11, 7, 7, 2, 199.00),
(12, 8, 5, 10, 12.50),
(13, 9, 3, 2, 180.00),
(14, 10, 8, 1, 280.00),
(15, 11, 1, 5, 25.00),
(16, 11, 2, 1, 45.00),
(17, 12, 4, 1, 420.00),
(18, 13, 7, 1, 199.00),
(19, 14, 2, 3, 45.00),
(20, 15, 6, 8, 8.00),
(21, 15, 5, 6, 12.50);
"""


def create_sample_database(path: str | Path) -> str:
    """Create (or refresh) the demo retail analytics SQLite database."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SEED_SQL)
        conn.commit()
    return str(db_path.resolve())
