#!/usr/bin/env python3
"""
SOLAT Dashboard Launcher

Starts the Streamlit dashboard for monitoring the autonomous trading system.

Usage:
    python3 run_dashboard.py

This will:
1. Launch the Streamlit app server
2. Open the dashboard in your default browser
3. Auto-refresh every 30 seconds
4. Connect to the SQLite database in read-only mode
"""

import os
import subprocess
import sys
import time

def main():
    """Launch the Streamlit dashboard."""

    print("=" * 70)
    print("SOLAT DASHBOARD LAUNCHER")
    print("=" * 70)
    print()

    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✓ Streamlit {streamlit.__version__} found")
    except ImportError:
        print("✗ Streamlit not installed. Installing...")
        os.system("pip install streamlit streamlit-autorefresh")
        print()

    # Check if database exists
    db_path = "data/db/trading_engine.db"
    if not os.path.exists(db_path):
        print(f"⚠ Warning: Database not found at {db_path}")
        print("  Sentinel may not have run yet. Dashboard will wait for data...")
        print()
    else:
        print(f"✓ Database found at {db_path}")
        print()

    # Launch dashboard
    print("Launching Streamlit dashboard...")
    print("=" * 70)
    print()

    try:
        # Run streamlit
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--logger.level=info"],
            cwd=os.getcwd()
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Dashboard stopped by user")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
