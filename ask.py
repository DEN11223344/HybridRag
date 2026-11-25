#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.cli_ask import main

if __name__ == "__main__":
    main()


