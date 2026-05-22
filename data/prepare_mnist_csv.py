#!/usr/bin/env python3
import csv
import gzip
import struct
from pathlib import Path


def load_mnist(images_path: Path, labels_path: Path):
    with gzip.open(images_path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic: {magic}")
        pixels = f.read(count * rows * cols)

    with gzip.open(labels_path, "rb") as f:
        magic, label_count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic: {magic}")
        labels = f.read(label_count)

    if count != label_count:
        raise ValueError(f"image count {count} != label count {label_count}")

    dim = rows * cols
    for i in range(count):
        features = [pixels[i * dim + j] / 255.0 for j in range(dim)]
        yield features, str(labels[i])


def write_csv(data_dir: Path, output_name: str, images_name: str, labels_name: str):
    output_path = data_dir / output_name
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for features, label in load_mnist(data_dir / images_name, data_dir / labels_name):
            writer.writerow([*features, label])
    print(f"wrote {output_path}")


def main():
    data_dir = Path(__file__).resolve().parent

    write_csv(
        data_dir,
        "mnist_train.csv",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    )
    write_csv(
        data_dir,
        "mnist_test.csv",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    )


if __name__ == "__main__":
    main()
