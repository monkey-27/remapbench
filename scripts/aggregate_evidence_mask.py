"""Compatibility wrapper for evidence-mask aggregation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_counterfactual_difference import main


if __name__ == "__main__":
    main()
