# csv_loader.py — Load CSV data into (features, label) tuples
#
# A CSV (comma-separated values) file looks like:
#   5.1,3.5,1.4,0.2,Iris-setosa
#   7.0,3.2,4.7,1.4,Iris-versicolor
#
# Each line is one data point. The last column is the label,
# and all other columns are numeric features.

# ============================================================
# STEP 6: Load CSV data
# ============================================================
# Read a CSV file and return a list of (features, label) tuples.
# Assume: no header row, last column is the label, all others are numeric.
#
# Python I/O hints:
#   - open("path.csv") returns a file object you can loop over line by line
#   - line.strip() removes trailing newline/whitespace
#   - line.split(",") splits on commas into a list of strings
#   - float("3.14") converts a string to a float

def load_csv(filepath: str) -> list[tuple[list[float], str]]:
    """Load a CSV file and return a list of (features, label) tuples."""
    # TODO: implement this
    # Hint: open the file, loop over lines, split each line, convert features to float
    pass
