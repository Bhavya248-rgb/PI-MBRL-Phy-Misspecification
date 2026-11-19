\# Sensitivity of Physics-Informed Model-Based RL to Parametric and Structural Model Errors



Capstone project, B.Tech CSE (AI \& ML), VIT-AP University.

Supervisor: Dr. Sibi Chakkaravarthy Sethuraman.



Studies when a learned residual network helps a physics model in model-based RL,

on two simulated robots: a differential-drive robot and a two-link arm.


## Folders
- System1        : differential-drive robot, Study 1 (wrong wheel radius)
- System1/studyB : robot Studies 2, 2b, 3 (motor lag, input fixes, both errors) and extra inputs
- System1/gru    : GRU (recurrent) residual for the robot
- System2        : two-link arm, Studies 1-3, robustness checks, task animation
- System2/gru    : GRU residual for the arm, offline data check


\## Requirements

Python 3, NumPy, PyTorch, Matplotlib



\## Run

python run\_study1.py        # in System1/studyB

python analyze\_study1.py    # tables, statistics, figures

python run\_arm\_study2.py    # in System2

python analyze\_arm.py 2

