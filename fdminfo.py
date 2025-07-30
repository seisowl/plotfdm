#!/usr/bin/env python

"""
Python version of fdminfo binary - shows FDM file header information
Usage: python fdminfo.py <fdm_file>
"""

import sys
from fdm import FDM

def main():
    if len(sys.argv) != 2:
        print("Usage: python fdminfo.py <fdm_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        fdm = FDM(filename)
        fdm.read_header_only()
        fdm.print_header()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
