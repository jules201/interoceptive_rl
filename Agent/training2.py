"""Training + greedy evaluation + core diagnostics for the homeostatic modular agent.

Everything per-step that the old `log` list was used for is now accumulated per episode,
so memory stays small and nothing important is lost.
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter  # needs: pip install pillow

from agent import Agent
from env import GridWorld

SHOW = True       # call plt.show() at the very end (set False for sweeps / headless runs)
MAKE_GIF = True   # rollout.gif is slow; turn off for quick runs

config = {
    "env":   dict(grid_size=5, n_stats=2, seed=0, episode_length=3000),
    "agent": dict(grid_size=5, n_stats=2, seed=0, lr=0.1, gamma=0.99,
                  n_bins=20, h_min=-1, h_max=3, n=4, m=2,
                  stat_decrease_per_step=np.array([0.02, 0.02]),
                  stat_replenish_per_resource=np.array([0.4, 0.4]),
                  set_point=1.0,
                  epsilon=0.9,            # overwritten every episode below; has no effect
                  boundary_penalty=0.01,  # unused in agent.py at the moment
                  storable=[False, True], storable_reward=0.03),
    "train": dict(n_episodes=3000, eps_start=1.0, eps_end=0.05),
    "eval":  dict(n_eval_episodes=5, start_stats=[1.0, 1.0]),
}

env = GridWorld(**config["env"])
agent = Agent(env=env, **config["agent"])

n_episodes = config["train"]["n_episodes"]
eps_start = config["train"]["eps_start"]
eps_end = config["train"]["eps_end"]
n_stats, n_bins = agent.n_stats, agent.n_bins
stat_idx = np.arange(n_stats)

# ----------------------------------------------------------------------------
# run folder, config, git state
# ----------------------------------------------------------------------------
run_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir.mkdir(parents=True)

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

config["git_commit"] = git("rev-parse", "--short", "HEAD")
config["git_dirty"] = bool(git("status", "--porcelain"))   # True = uncommitted changes in this run

with open(run_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# ----------------------------------------------------------------------------
# helpers: binning, summaries, plots
# ----------------------------------------------------------------------------
def bin_index(h):
    """Same binning as agent.encode (used here only for occupancy counts)."""
    return np.clip(((h - agent.h_min) / (agent.h_max - agent.h_min) * n_bins).astype(int), 0, n_bins - 1)


def moving_avg(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def q_report(Q):
    rep = {}
    for i in range(Q.shape[0]):
        q = Q[i].ravel()
        rep[f"sub_agent_{i + 1}"] = dict(
            min=float(q.min()), max=float(q.max()), mean=float(q.mean()), std=float(q.std()),
            exactly_zero=float(np.mean(q == 0)), abs_lt_0p01=float(np.mean(np.abs(q) < 0.01)),
            gt_0p01=float(np.mean(q > 0.01)), lt_minus_0p01=float(np.mean(q < -0.01)))
    return rep


def summarize_training(w=200):
    for name, sl in (("first", slice(0, w)), ("last", slice(-w, None))):
        died = ep_died[sl]
        print(f"\n[{name} {w} episodes]")
        print("  mean steps survived:   ", ep_len[sl].mean())
        print("  fraction ending in death:", died.mean())
        print("  mean deviation per stat:", ep_dev[sl].mean(axis=0))
        print("  mean |TD error| per stat:", ep_td[sl].mean(axis=0))
        print("  min / max stat reached: ", ep_hmin[sl].min(axis=0), ep_hmax[sl].max(axis=0))
        occ = ep_bins[sl].sum(axis=0) / ep_len[sl].sum()           # (n_stats, n_bins)
        print("  fraction in lowest bin: ", occ[:, 0])
        print("  fraction in highest bin:", occ[:, -1])
        cause, on_res = ep_cause[sl][died], ep_last_on_res[sl][died]
        for i in range(n_stats):
            fatal = cause[:, i] != 0
            on_res_frac = on_res[fatal, i].mean() if fatal.any() else float("nan")
            print(f"  stat {i + 1}: deaths below h_min {np.mean(cause[:, i] == -1):.3f}, "
                  f"above h_max {np.mean(cause[:, i] == 1):.3f} (fractions of deaths); "
                  f"on a resource in the fatal step: {on_res_frac:.3f}")


def plot_learning_curves(w=50):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(w - 1, n_episodes)
    ax[0, 0].plot(x, moving_avg(ep_len, w))
    ax[0, 0].axhline(env.episode_length, ls="--", color="grey")
    ax[0, 0].set_title("steps survived")
    ax[0, 1].plot(x, moving_avg(ep_died.astype(float), w))
    ax[0, 1].set_title("fraction of episodes ending in death")
    for i in range(n_stats):
        ax[1, 0].plot(x, moving_avg(ep_dev[:, i], w), label=f"stat {i + 1}")
        ax[1, 1].plot(x, moving_avg(ep_td[:, i], w), label=f"stat {i + 1}")
    ax[1, 0].set_title("mean |h - set point| per episode")
    ax[1, 1].set_title("mean |TD error| per episode")
    ax[1, 1].set_yscale("log")
    ax[1, 0].legend()
    for a in ax.flat:
        a.set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(run_dir / "learning_curves.png", dpi=150)


def plot_value_maps(refs=((0.4, 1.0), (1.0, 0.4))):
    vms = [agent.value_map(np.array(h)) for h in refs]
    fig, axes = plt.subplots(n_stats, len(refs), figsize=(10, 8), squeeze=False)
    for i in range(n_stats):
        for j, (h, v) in enumerate(zip(refs, vms)):
            im = axes[i, j].imshow(v[i])
            axes[i, j].scatter(env.resource_pos[:, 1], env.resource_pos[:, 0],
                               marker="X", color="red", s=100, label="resources")
            axes[i, j].set_title(f"Sub-agent {i + 1}: stats {list(h)}")
            fig.colorbar(im, ax=axes[i, j])      # one colorbar per panel: each has its own scale
    axes[0, 0].legend(fontsize=8)
    fig.savefig(run_dir / "value_maps.png", dpi=150, bbox_inches="tight")


def plot_h_over_time(run, n=300):
    # stats[t] and on_resource[t] are AFTER action t
    stats, on_res = run["stats"][:n], run["on_resource"][:n]
    time = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(11, 4))
    for i in range(n_stats):
        ax.plot(time, stats[:, i], color=f"C{i}", label=f"stat {i + 1}")
        hit = np.flatnonzero(on_res[:, i])
        ax.scatter(hit, stats[hit, i], color=f"C{i}", s=25, zorder=3)
    ax.axhline(agent.set_point, color="k", lw=1, label="set point")
    ax.axhline(agent.h_min, color="r", ls="--", lw=1, label="death bounds")
    ax.axhline(agent.h_max, color="r", ls="--", lw=1)
    ax.set_ylabel("internal stat")
    ax.set_xlabel("time step")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "h_over_time.png", dpi=150, bbox_inches="tight")


def make_rollout_gif(run, n=300):
    # align: stats[t] is AFTER action t; locs[t + 1] is where action t landed
    stats, locs = run["stats"][:-1][:n], run["locs"][1:][:n]
    T, G = len(stats), agent.grid_size
    vmaps = np.array([agent.value_map(s) for s in stats])      # (T, n_stats, G, G)

    fig, (axg, axs, axq0, axq1) = plt.subplots(1, 4, figsize=(18, 4.5))
    axg.set_xlim(-0.5, G - 0.5); axg.set_ylim(G - 0.5, -0.5); axg.set_aspect("equal")
    axg.set_xticks(range(G)); axg.set_yticks(range(G)); axg.grid(True)
    for i, (r, c) in enumerate(env.resource_pos):
        axg.scatter(c, r, marker="X", s=250, color=f"C{i}", label=f"resource {i + 1}")
    axg.legend(loc="upper center", ncol=2, fontsize=8)
    for i in range(n_stats):
        axs.plot(np.arange(T), stats[:, i], color=f"C{i}", label=f"stat {i + 1}")
    axs.axhline(agent.set_point, color="k", lw=1)
    axs.axhline(agent.h_min, color="r", ls="--", lw=1)
    axs.axhline(agent.h_max, color="r", ls="--", lw=1)
    axs.set_xlabel("time step"); axs.legend(loc="upper right")

    dot, = axg.plot([], [], "o", color="black", markersize=14)
    vline = axs.axvline(0, color="gray")
    label = axg.text(0.5, -0.12, "", transform=axg.transAxes, ha="center")
    vmin, vmax = vmaps.min(), vmaps.max()                      # same color scale for both sub-agents
    im0 = axq0.imshow(vmaps[0, 0], vmin=vmin, vmax=vmax)
    im1 = axq1.imshow(vmaps[0, 1], vmin=vmin, vmax=vmax)
    dot0, = axq0.plot([], [], "wo"); dot1, = axq1.plot([], [], "wo")
    axq0.set_title("sub-agent 1 value"); axq1.set_title("sub-agent 2 value")
    fig.colorbar(im0, ax=[axq0, axq1])

    def update(frame):
        r, c = locs[frame]
        dot.set_data([c], [r])
        vline.set_xdata([frame, frame])
        label.set_text(f"t={frame}   stats = {np.round(stats[frame], 2)}")
        im0.set_data(vmaps[frame, 0]); im1.set_data(vmaps[frame, 1])
        dot0.set_data([c], [r]); dot1.set_data([c], [r])
        return dot, vline, label

    anim = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    anim.save(run_dir / "rollout.gif", writer=PillowWriter(fps=5))
    plt.close(fig)


# ----------------------------------------------------------------------------
# training: everything is recorded per episode
# ----------------------------------------------------------------------------
ep_len = np.zeros(n_episodes, dtype=int)                        # steps survived
ep_died = np.zeros(n_episodes, dtype=bool)                      # ended by death (not time limit)
ep_dev = np.zeros((n_episodes, n_stats))                        # mean |set_point - h| per stat
ep_td = np.zeros((n_episodes, n_stats))                         # mean |TD error| per stat
ep_hmin = np.zeros((n_episodes, n_stats))                       # lowest value each stat reached
ep_hmax = np.zeros((n_episodes, n_stats))                       # highest value each stat reached
ep_bins = np.zeros((n_episodes, n_stats, n_bins), dtype=int)    # steps spent in each bin
ep_cause = np.zeros((n_episodes, n_stats), dtype=int)           # at death: -1 below h_min, +1 above h_max
ep_last_on_res = np.zeros((n_episodes, n_stats), dtype=bool)    # on a resource in the fatal step?

for ep in range(n_episodes):
    env.reset()
    agent.make_stats()
    loc = env.agent_pos.copy()
    # decay epsilon linearly over the first half of training
    agent.epsilon = max(eps_end, eps_start - (eps_start - eps_end) * ep / (0.5 * n_episodes))

    dev_sum, td_sum = np.zeros(n_stats), np.zeros(n_stats)
    h_lo, h_hi = np.full(n_stats, np.inf), np.full(n_stats, -np.inf)

    for t in range(env.episode_length):
        a = agent.select_action(loc, agent.stats, training=True)
        loc_next, on_resource, done = env.step(a)
        h_prev, rewards, dead, dead_per_stat = agent.organism_step(on_resource)
        deltas = agent.update_value_map(location=loc, stats_old=h_prev, stats_new=agent.stats,
                                        action=a, rewards=rewards, location_new=loc_next, done=dead)

        dev_sum += np.abs(agent.set_point - agent.stats)        # includes the fatal step
        td_sum += np.abs(deltas)
        h_lo, h_hi = np.minimum(h_lo, agent.stats), np.maximum(h_hi, agent.stats)
        ep_bins[ep, stat_idx, bin_index(agent.stats)] += 1

        loc = loc_next
        if dead or done:
            break

    ep_len[ep], ep_died[ep] = t + 1, dead
    ep_dev[ep], ep_td[ep] = dev_sum / (t + 1), td_sum / (t + 1)
    ep_hmin[ep], ep_hmax[ep] = h_lo, h_hi
    if dead:
        ep_cause[ep] = np.where(agent.stats <= agent.h_min, -1, np.where(agent.stats >= agent.h_max, 1, 0))
        ep_last_on_res[ep] = np.asarray(on_resource, dtype=bool)

# consistency checks (nothing printed = all good)
assert ((ep_len == env.episode_length) | ep_died).all()         # episodes end only by death or time limit
assert (ep_bins.sum(axis=2) == ep_len[:, None]).all()           # bin counts add up to steps taken

np.savez(run_dir / "episode_metrics.npz", ep_len=ep_len, ep_died=ep_died, ep_dev=ep_dev, ep_td=ep_td,
         ep_hmin=ep_hmin, ep_hmax=ep_hmax, ep_bins=ep_bins, ep_cause=ep_cause,
         ep_last_on_res=ep_last_on_res)
np.save(run_dir / "Q.npy", agent.Q)

# ----------------------------------------------------------------------------
# training diagnostics: printed and saved
# ----------------------------------------------------------------------------
qrep = q_report(agent.Q)
print("Q overall: min", agent.Q.min(), "max", agent.Q.max(), "mean", agent.Q.mean(), "std", agent.Q.std())
for name, r in qrep.items():
    print(f"\n{name}:", {k: round(v, 4) for k, v in r.items()})
summarize_training()

# Q-values at the resource cells for a low-stat state (are the modules pulled toward their resource?)
h_probe = np.array([0.4, 1.0])
for i in range(n_stats):
    print(f"\nSub-agent {i + 1}")
    for j, pos in enumerate(env.resource_pos):
        q = agent.Q[i, agent.encode(pos, h_probe), :]
        print(f"  resource {j + 1} at {pos}: Q = {np.round(q, 3)}, state value = {q.max():.3f}")

plot_learning_curves()
plot_value_maps()

# ----------------------------------------------------------------------------
# greedy evaluation (no exploration, no learning)
# ----------------------------------------------------------------------------
eval_runs = []
for k in range(config["eval"]["n_eval_episodes"]):
    env.reset()
    agent.make_stats()
    agent.stats = np.array(config["eval"]["start_stats"], dtype=float)   # fixed start
    loc = env.agent_pos.copy()
    locs, stats_hist, actions, on_res_hist = [], [], [], []

    for t in range(env.episode_length):
        a = agent.select_action(loc, agent.stats, training=False)
        locs.append(loc.copy())                       # position the action was chosen in
        actions.append(a)
        loc_next, on_resource, done = env.step(a)
        h_prev, rewards, dead, dead_per_stat = agent.organism_step(on_resource)
        stats_hist.append(agent.stats.copy())         # stats after this step
        on_res_hist.append(np.asarray(on_resource, dtype=bool))
        loc = loc_next
        if dead or done:
            break

    eval_runs.append({"locs": np.array(locs), "stats": np.array(stats_hist), "actions": np.array(actions),
                      "on_resource": np.array(on_res_hist), "died": bool(dead)})

eval_summary = dict(
    steps_survived=[len(r["stats"]) for r in eval_runs],
    died=[r["died"] for r in eval_runs],
    deviation_per_stat=[np.abs(agent.set_point - r["stats"]).mean(axis=0).tolist() for r in eval_runs],
)
print("\nGreedy evaluation:", eval_summary)
with open(run_dir / "summary.json", "w") as f:
    json.dump({"q": qrep, "eval": eval_summary}, f, indent=2)
np.savez(run_dir / "eval_run0.npz", **eval_runs[0])

plot_h_over_time(eval_runs[0])
if MAKE_GIF and len(eval_runs[0]["stats"]) > 2:
    make_rollout_gif(eval_runs[0])

if SHOW:
    plt.show()
plt.close("all")