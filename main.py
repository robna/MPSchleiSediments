import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent
APP   = SCRIPT / "analysis"
sys.path.append(str(APP))

if __name__ == "__main__":
    # Stay in project root so data paths resolve correctly
    import app
    app.main()
