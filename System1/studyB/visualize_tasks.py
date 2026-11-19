"""
visualize_tasks.py

Shows the robot driving the 6 new evaluation tasks with different models,
after each model has learned for 15 episodes (exactly as in the studies).

Default setting = the clearest Study 2 case: motor lag tau = 0.2 s, correct
wheel radius. Compared models:
  physics only               (blue)
  hybrid, pose only          (orange)  -- the one that gets confused
  hybrid, + wheel encoders   (purple)  -- the fix
Change CONFIG below to show another setting (e.g. a radius error, or the
'history' input).

Outputs (in the current folder):
  tasks_paths.png   still picture: the 6 tasks, each model's path
  tasks_robot.gif   animation: the robots driving all 6 tasks at once

Takes about 1-2 minutes. Usage:
    python visualize_tasks.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from diffdrive_sim import DiffDriveSimulator
from model_based_rl import run_model_based_rl, run_episode
from task import EVAL_TASKS, SUCCESS_RADIUS, MAX_STEPS

# =====================================================================
# CONFIG
# =====================================================================
TRUE_R, WHEELBASE, TRUE_SLIP, DT = 0.03, 0.15, 0.85, 0.1
MOTOR_TAU = 0.2            # motor lag of the real robot (s); 0 = Study 1 robot
RADIUS_ERROR_PCT = 0.0     # wrong radius in the physics prior (%)
SEED = 0
# (label, model type, what the model sees, colour)
MODELS = [
    ('physics only', 'physics_only', 'pose', '#2a78d6'),
    ('hybrid, pose only', 'hybrid', 'pose', '#eb6834'),
    ('hybrid, + wheel encoders', 'hybrid', 'wheel_speed', '#6250d6'),
]
GIF_FPS = 10               # 10 frames per second = real time (dt = 0.1 s)
# =====================================================================


def train_and_drive(model_type, obs_mode):
    """Learns exactly as in the studies (warmup + 15 episodes), then drives
    the 6 new tasks with learning off. Returns per-task paths and success."""
    assumed_r = TRUE_R * (1 - RADIUS_ERROR_PCT / 100)
    res = run_model_based_rl(model_type, TRUE_R, WHEELBASE, TRUE_SLIP, DT,
                             assumed_r=assumed_r, assumed_slip=TRUE_SLIP,
                             seed=SEED, motor_tau=MOTOR_TAU, obs_mode=obs_mode,
                             do_eval=False, verbose=False)
    model = res['model_object']
    sim = DiffDriveSimulator(TRUE_R, WHEELBASE, TRUE_SLIP, DT, motor_tau=MOTOR_TAU)
    np.random.seed(SEED + 10_000)          # same planner noise as the studies' eval
    out = []
    for start, goal in EVAL_TASKS:
        ep = run_episode(sim, model, start_pose=start, goal=goal, obs_mode=obs_mode)
        poses = [t[0][:3] for t in ep['transitions']]
        poses.append(ep['transitions'][-1][2][:3] if ep['transitions'] else np.array(start))
        out.append({'poses': np.array(poses), 'success': ep['success'],
                    'steps': ep['steps']})
    return out, res['mean_return']


def setup_axes(ax, k, start, goal):
    ax.set_xlim(-0.35, 1.0)
    ax.set_ylim(-0.35, 1.0)
    ax.set_aspect('equal')
    ax.add_patch(Circle(goal, SUCCESS_RADIUS, color='#1baf7a', alpha=0.18))
    ax.plot(*goal, marker='*', color='#1baf7a', ms=12)
    ax.plot(start[0], start[1], 'ks', ms=5)
    ax.annotate('', xy=(start[0] + 0.12 * np.cos(start[2]), start[1] + 0.12 * np.sin(start[2])),
                xytext=(start[0], start[1]), arrowprops=dict(arrowstyle='->', color='k', lw=1))
    ax.set_title(f'New task {k + 1}', fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)


def main():
    lag = f'motor lag tau = {MOTOR_TAU} s' if MOTOR_TAU > 0 else 'no motor lag'
    setting = f'{lag}, radius error {RADIUS_ERROR_PCT:+.0f}%, seed {SEED}'
    results = []
    for label, mtype, mode, color in MODELS:
        print(f'Training and driving: {label} ...')
        paths, ret = train_and_drive(mtype, mode)
        n_ok = sum(p['success'] for p in paths)
        print(f'   solved {n_ok}/6 new tasks')
        results.append((label, color, paths, n_ok))

    # ------------------------------------------------ still picture
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for k, (start, goal) in enumerate(EVAL_TASKS):
        ax = axes.flat[k]
        setup_axes(ax, k, start, goal)
        for label, color, paths, _ in results:
            p = paths[k]['poses']
            ax.plot(p[:, 0], p[:, 1], color=color, lw=2,
                    ls='-' if paths[k]['success'] else '--')
            ax.plot(p[-1, 0], p[-1, 1], 'o', color=color, ms=5)
    handles = [plt.Line2D([], [], color=c, lw=2, label=f'{l} ({n}/6 reached)')
               for l, c, _, n in results]
    handles += [plt.Line2D([], [], color='gray', lw=2, ls='--', label='dashed = goal not reached in 120 steps'),
                plt.Line2D([], [], color='#1baf7a', marker='*', ls='', ms=10, label='goal (shaded = 10 cm success zone)'),
                plt.Line2D([], [], color='k', marker='s', ls='', ms=5, label='start (arrow = initial heading)')]
    fig.suptitle(f'Robot paths on 6 unseen tasks after learning ({setting})', y=0.985, fontsize=12)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig('tasks_paths.png', dpi=140)
    plt.close(fig)
    print('Saved tasks_paths.png')

    # ------------------------------------------------ animation
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    artists = []
    for k, (start, goal) in enumerate(EVAL_TASKS):
        ax = axes.flat[k]
        setup_axes(ax, k, start, goal)
        row = []
        for label, color, paths, _ in results:
            trail, = ax.plot([], [], color=color, lw=1.8)
            body = Circle((0, 0), 0.035, color=color, alpha=0.85)
            ax.add_patch(body)
            head, = ax.plot([], [], color='k', lw=1.5)
            row.append((trail, body, head, paths[k]))
        artists.append(row)
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=3, label=f'{l} ({n}/6 reached)')
                        for l, c, _, n in results],
               loc='upper center', ncol=3, frameon=False, fontsize=10)
    clock = fig.text(0.5, 0.905, '', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    n_frames = max(len(p['poses']) for _, _, paths, _ in results for p in paths)

    def draw(f):
        clock.set_text(f'{setting}   |   t = {f * DT:4.1f} s')
        for row in artists:
            for trail, body, head, path in row:
                p = path['poses']
                i = min(f, len(p) - 1)
                trail.set_data(p[:i + 1, 0], p[:i + 1, 1])
                x, y, th = p[i]
                body.center = (x, y)
                head.set_data([x, x + 0.05 * np.cos(th)], [y, y + 0.05 * np.sin(th)])
        return []

    anim = FuncAnimation(fig, draw, frames=n_frames + 10, interval=1000 / GIF_FPS)
    anim.save('tasks_robot.gif', writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)
    print('Saved tasks_robot.gif')


if __name__ == '__main__':
    main()
