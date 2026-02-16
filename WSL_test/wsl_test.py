#!/usr/bin/env python3

import sys
from pathlib import Path

def main():
    print("Hello from WSL and Python!")
    print("-" * 40)

    # Show Python version
    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version.split()[0]}")
    print("-" * 40)

    # Show current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")
    print("Files here:")
    for p in cwd.iterdir():
        print(f"  - {p.name}")

if __name__ == "__main__":
    main()
