"""
arm_path.py

Where the System 2 (arm) code and results are. This GRU folder sits INSIDE it:

    System2/
        arm_sim.py, arm_mbrl.py, ...      <- existing code (unchanged)
        results_arm_study2/               <- baseline results
        gru/                              <- this folder: the GRU files + results_arm_gru/

The GRU code imports the arm, task, models and learning loop from the
System2 folder, and the analysis reads results_arm_study2/ from there.
If your layout is different, change ARM_FOLDER (relative to this folder, or a full path).
"""

import os
import sys

ARM_FOLDER = '..'

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_DIR = os.path.normpath(ARM_FOLDER if os.path.isabs(ARM_FOLDER) else os.path.join(HERE, ARM_FOLDER))

if not os.path.isfile(os.path.join(ARM_DIR, 'arm_mbrl.py')):
    raise SystemExit(f"Could not find the arm code in {ARM_DIR}\n"
                     f"Open arm_path.py and set ARM_FOLDER to your System2 folder.")
if ARM_DIR not in sys.path:
    sys.path.insert(1, ARM_DIR)      # after this folder, so the GRU files come first
