"""Run the reproducible member-A grid. Defaults to the coverage pilot only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.deployment_grid import main

if __name__ == "__main__":
    main()
