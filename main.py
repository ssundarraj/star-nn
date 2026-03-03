# main.py — Entry point for your KNN classifier
#
# Once you've implemented the functions in knn.py and csv_loader.py,
# this program will:
#   1. Load the Iris dataset from a CSV file.
#   2. Split it into training and test sets.
#   3. Run KNN classification and print accuracy.
#
# Run with: python main.py

from knn import evaluate
from csv_loader import load_csv

data = load_csv("data/iris.csv")

if not data:
    print("Failed to load data. Check the file path.")
    exit(1)

print(f"Loaded {len(data)} data points.")

# Simple train/test split: first 80% train, last 20% test
split = len(data) * 80 // 100
train = data[:split]
test = data[split:]

print(f"Training set: {len(train)} points")
print(f"Test set:     {len(test)} points")

# Try different values of K
for k in [1, 3, 5, 7]:
    accuracy = evaluate(test, train, k)
    print(f"K={k}  accuracy={accuracy * 100:.1f}%")
