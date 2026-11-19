"""
visualize_arm_tasks.py

System 2 (arm) version of visualize_tasks.py: shows the arm reaching the 6
new evaluation tasks with different models, after each model has learned
for 15 episodes (exactly as in the studies).

Default setting = a strong Study 2 case on the arm: motor lag
tau = 0.3 s, correct link masses. Compared models:
  physics only               (blue)
  hybrid, state only         (orange)  -- the one that gets confused
  hybrid, + torque sensor    (purple)  -- the fix
Change CONFIG below to show another setting (e.g. a mass error, 'history'
or 'prev_action').

Outputs (in the current folder):
  arm_tasks_paths.png   still picture: the 6 tasks, each model's tip path
                        and the final arm pose
  arm_tasks.gif         animation: the three arms moving on all 6 tasks

Put this file in the System2 folder (next to arm_mbrl.py) and run:
    python visualize_arm_tasks.py
Takes a few minutes (each learned model trains as in the studies).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

from arm_sim import ArmSimulator, M1, M2, DT, L1, L2, tip_position
from arm_mbrl import run_model_based_rl, run_episode
from arm_task import EVAL_TASKS, SUCCESS_RADIUS, MAX_STEPS

# =====================================================================
# CONFIG
# =====================================================================
MOTOR_TAU = 0.3            # motor lag of the real arm (s); 0 = Study 1 arm
MASS_ERROR_PCT = 0.0       # wrong link masses in the physics prior (%)
SEED = 0
# (label, model type, what the model sees, colour)
MODELS = [
    ('physics only', 'physics_only', 'state', '#2a78d6'),
    ('hybrid, state only', 'hybrid', 'state', '#eb6834'),
    ('hybrid, + torque sensor', 'hybrid', 'torque', '#6250d6'),
]
GIF_FPS = 20               # 20 frames per second = real time (dt = 0.05 s)
FRAME_STRIDE = 2           # 1 = every step (larger GIF), 2 = every second step
# =====================================================================


def joints(q):
    """Shoulder, elbow and tip positions for joint angles q = (q1, q2)."""
    q1, q2 = q[0], q[1]
    elbow = np.array([L1 * np.cos(q1), L1 * np.sin(q1)])
    tip = elbow + np.array([L2 * np.cos(q1 + q2), L2 * np.sin(q1 + q2)])
    return np.array([[0.0, 0.0], elbow, tip])


def train_and_move(model_type, obs_mode):
    """Learns exactly as in the studies (warmup + 15 episodes), then runs
    the 6 new tasks with learning off. Returns per-task joint-angle paths."""
    res = run_model_based_rl(model_type, mass_error_pct=MASS_ERROR_PCT,
                             motor_tau=MOTOR_TAU, obs_mode=obs_mode, seed=SEED,
                             do_eval=False, verbose=False)
    model = res['model_object']
    sim = ArmSimulator(M1, M2, motor_tau=MOTOR_TAU)
    np.random.seed(SEED + 10_000)          # same planner noise as the studies' eval
    out = []
    for start, goal in EVAL_TASKS:
        ep = run_episode(sim, model, start=start, goal=goal, obs_mode=obs_mode)
        qs = [t[0][:2] for t in ep['transitions']]
        qs.append(ep['transitions'][-1][2][:2] if ep['transitions'] else np.array(start))
        out.append({'q': np.array(qs), 'success': ep['success'], 'steps': ep['steps'],
                    'final_dist': ep['final_dist']})
    return out


def setup_axes(ax, k, start, goal):
    reach = L1 + L2
    ax.set_xlim(-reach - 0.05, reach + 0.05)
    ax.set_ylim(-reach - 0.05, reach + 0.05)
    ax.set_aspect('equal')
    ax.add_patch(Circle((0, 0), reach, fill=False, ls=':', color='0.6', lw=0.8))
    ax.add_patch(Circle(goal, SUCCESS_RADIUS, color='#1baf7a', alpha=0.2))
    ax.plot(*goal, marker='*', color='#1baf7a', ms=12)
    sj = joints(start)
    ax.plot(sj[:, 0], sj[:, 1], color='0.7', lw=3, solid_capstyle='round', zorder=1)
    ax.plot(0, 0, 'ks', ms=6, zorder=5)
    ax.set_title(f'New task {k + 1}', fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=7)


def main():
    lag = f'motor lag tau = {MOTOR_TAU} s' if MOTOR_TAU > 0 else 'no motor lag'
    setting = f'{lag}, mass error {MASS_ERROR_PCT:+.0f}%, seed {SEED}'
    results = []
    for label, mtype, mode, color in MODELS:
        print(f'Training and running: {label} ...')
        paths = train_and_move(mtype, mode)
        n_ok = sum(p['success'] for p in paths)
        print(f'   solved {n_ok}/6 new tasks')
        results.append((label, color, paths, n_ok))

    # ------------------------------------------------ still picture
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for k, (start, goal) in enumerate(EVAL_TASKS):
        ax = axes.flat[k]
        setup_axes(ax, k, start, goal)
        for label, color, paths, _ in results:
            q = paths[k]['q']
            tips = np.array([joints(qi)[2] for qi in q])
            ax.plot(tips[:, 0], tips[:, 1], color=color, lw=1.6, alpha=0.9,
                    ls='-' if paths[k]['success'] else '--')
            fj = joints(q[-1])
            ax.plot(fj[:, 0], fj[:, 1], '-o', color=color, lw=2.5, ms=3)
    handles = [plt.Line2D([], [], color=c, lw=2, label=f'{l} ({n}/6 reached)')
               for l, c, _, n in results]
    handles += [plt.Line2D([], [], color='gray', lw=2, ls='--', label='dashed tip path = target not reached'),
                plt.Line2D([], [], color='#1baf7a', marker='*', ls='', ms=10, label='target (shaded = 5 cm zone)'),
                plt.Line2D([], [], color='0.7', lw=3, label='start pose (grey); coloured = final pose')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f'Arm on 6 unseen tasks after learning ({setting})', y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig('arm_tasks_paths.png', dpi=140)
    plt.close(fig)
    print('Saved arm_tasks_paths.png')

    # ------------------------------------------------ animation
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
    artists = []
    for k, (start, goal) in enumerate(EVAL_TASKS):
        ax = axes.flat[k]
        setup_axes(ax, k, start, goal)
        row = []
        for label, color, paths, _ in results:
            trail, = ax.plot([], [], color=color, lw=1.2, alpha=0.7)
            arm, = ax.plot([], [], '-o', color=color, lw=2.5, ms=3)
            row.append((trail, arm, paths[k]))
        artists.append(row)
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=3, label=f'{l} ({n}/6 reached)')
                        for l, c, _, n in results],
               loc='upper center', ncol=3, frameon=False, fontsize=10)
    clock = fig.text(0.5, 0.905, '', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    n_steps = max(len(p['q']) for _, _, paths, _ in results for p in paths)
    frames = list(range(0, n_steps, FRAME_STRIDE)) + [n_steps - 1] * 15

    def draw(f):
        clock.set_text(f'{setting}   |   t = {f * DT:4.2f} s')
        for row in artists:
            for trail, arm, path in row:
                q = path['q']
                i = min(f, len(q) - 1)
                tips = np.array([joints(qi)[2] for qi in q[:i + 1]])
                trail.set_data(tips[:, 0], tips[:, 1])
                j = joints(q[i])
                arm.set_data(j[:, 0], j[:, 1])
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 * FRAME_STRIDE / GIF_FPS)
    anim.save('arm_tasks.gif', writer=PillowWriter(fps=GIF_FPS // FRAME_STRIDE))
    plt.close(fig)
    print('Saved arm_tasks.gif')


if __name__ == '__main__':
    main()
