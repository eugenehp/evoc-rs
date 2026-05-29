#!/usr/bin/env python3
"""
Read raw float32 values from stdin and write NumPy argsort indices as little-endian int64 bytes.

This is used by Rust parity code to match NumPy's exact sort/tie behavior.
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    data = sys.stdin.buffer.read()
    if not data:
        return
    x = np.frombuffer(data, dtype=np.float32)
    order = np.argsort(x, kind="quicksort").astype(np.int64, copy=False)
    sys.stdout.buffer.write(order.astype("<i8", copy=False).tobytes(order="C"))


if __name__ == "__main__":
    main()

