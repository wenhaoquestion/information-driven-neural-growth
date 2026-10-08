#!/usr/bin/env python3
"""Optional future rerun in a new child directory; never overwrites frozen work."""
import re
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import compute_pilot as pilot

if len(sys.argv) != 2 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', sys.argv[1]):
    raise SystemExit('usage: reproduce_once.py NEW_CHILD_DIRECTORY_NAME')
destination = Path(__file__).resolve().parent / sys.argv[1]
destination.mkdir(exist_ok=False)
pilot.OUT = destination
pilot.run_compute()
pilot.run_compare()
