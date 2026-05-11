import csv
import random
import uuid
from datetime import datetime, timedelta

products = [
    "Laptop", "Mouse", "Keyboard", "Monitor", "Headphones",
    "Webcam", "USB Hub", "SSD", "RAM", "Mousepad"
]

statuses = ["pending", "completed", "shipped", "delivered", "cancelled"]

def generate_orders(n=100, filename="data/orders.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "customer_id", "product", "amount", "status"])

        for _ in range(n):
            order_id   = str(uuid.uuid4())[:8].upper()
            customer_id = f"CUST{random.randint(1000, 9999)}"
            product    = random.choice(products)
            amount     = round(random.uniform(5.00, 1500.00), 2)
            status     = random.choice(statuses)

            writer.writerow([order_id, customer_id, product, amount, status])

    print(f"Generated {n} rows → {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a sample orders CSV.")
    parser.add_argument("-n", "--rows",     type=int, default=100,          help="Number of rows (default: 100)")
    parser.add_argument("-o", "--output",   type=str, default="orders.csv", help="Output filename")
    args = parser.parse_args()

    generate_orders(n=args.rows, filename=args.output)