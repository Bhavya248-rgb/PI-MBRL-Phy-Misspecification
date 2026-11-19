"""
studyb_path.py

Where the System 1 Study 2 folder is. This GRU folder sits NEXT TO it:

    System1/
        studyB/        <- existing code (models.py, model_based_rl.py, ...) and
                          results_study2/, results_study2b/, results_extra/
        gru/           <- this folder (only the new GRU files + results_gru/)

The GRU code imports the robot, task, models and learning loop from studyB
(nothing there is copied or changed) and the analysis reads the baseline
results from there. If your folder has a different name or place, change
STUDY_B below (a path relative to this folder, or a full path).
"""

import os
import sys

STUDY_B = os.path.join('..', 'studyB')

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_B_DIR = os.path.normpath(STUDY_B if os.path.isabs(STUDY_B) else os.path.join(HERE, STUDY_B))

if not os.path.isfile(os.path.join(STUDY_B_DIR, 'model_based_rl.py')):
    raise SystemExit(f"Could not find the Study 2 code in {STUDY_B_DIR}\n"
                     f"Open studyb_path.py and set STUDY_B to your studyB folder.")
if STUDY_B_DIR not in sys.path:
    sys.path.insert(1, STUDY_B_DIR)      # after this folder, so the GRU files come first
