# mnist_loader.py — Load MNIST from the original IDX binary format
#
# The IDX format stores images and labels in separate files:
#   - Images: 16-byte header (magic, count, rows, cols) + raw pixel bytes
#   - Labels: 8-byte header (magic, count) + raw label bytes
#
# Each pixel is an unsigned byte (0-255). We normalize to [0, 1].

import gzip
import struct


def load_mnist(
    images_path: str,
    labels_path: str,
) -> list[tuple[list[float], str]]:
    """Load MNIST data from IDX files (gzipped or raw).

    Returns list of (features, label) tuples where features is a list
    of 784 floats in [0, 1] and label is a string like "3".
    """
    opener = gzip.open if images_path.endswith(".gz") else open

    with opener(images_path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        pixels = f.read(count * rows * cols)

    with opener(labels_path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        labels = f.read(count)

    data = []
    dim = rows * cols
    for i in range(count):
        features = [pixels[i * dim + j] / 255.0 for j in range(dim)]
        label = str(labels[i])
        data.append((features, label))

    return data
