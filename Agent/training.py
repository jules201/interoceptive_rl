from env import GridWorld
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt



env = GridWorld(grid_size=5, n_stats=2, seed=0, episode_length=2000)
agent = Agent(grid_size=5, env=env, n_stats=2, seed=0)

n_episodes, eps_start, eps_end = 2000, 1.0, 0.05
rollout = 5
log = []

for ep in range(n_episodes):
    env.reset() #environment initialization
    agent.make_stats() # internal stats initialization
    loc = env.agent_pos.copy() # copy of the agent location
    # decay epsilon linearly over the first half of training
    agent.epsilon = max(eps_end, eps_start - (eps_start - eps_end) * ep / (0.5 * n_episodes)) 

    for t in range(env.episode_length):
        a = agent.select_action(loc, agent.stats, training=True) #select action based on agent location and internal stats

        loc_next, on_resource, done = env.step(a)

        h_prev, rewards, dead, dead_per_stat = agent.organism_step(on_resource)
        

        deltas = agent.update_value_map(location=loc, stats_old=h_prev, stats_new=agent.stats, action=a, rewards=rewards, location_new=loc_next, done=dead)

        log.append((ep, t, agent.stats.copy(), rewards.copy(), deltas.copy(), a))
        loc = loc_next
        if dead or done:
            break


print("Q min:", agent.Q.min())
print("Q max:", agent.Q.max())
print("Q mean:", agent.Q.mean())
print("Q std:", agent.Q.std())

for i in range(agent.n_stats):
    print(f"\nSub-agent {i + 1}")
    print("Min:", agent.Q[i].min())
    print("Max:", agent.Q[i].max())
    print("Mean:", agent.Q[i].mean())
    print("Std:", agent.Q[i].std())

for i in range(agent.n_stats):

    q = agent.Q[i].flatten()

    print(f"\nSub-agent {i + 1}")

    print("Exactly zero:", np.mean(q == 0))
    print("Abs(Q) < 0.001:", np.mean(np.abs(q) < 0.001))
    print("Abs(Q) < 0.01:", np.mean(np.abs(q) < 0.01))
    print("Q > 0.01:", np.mean(q > 0.01))
    print("Q < -0.01:", np.mean(q < -0.01))





v0 = agent.value_map(np.array([0.4, 1.0]))
v1 = agent.value_map(np.array([1.0, 0.4]))

resource_loc = env.resource_pos
agent_position = env.agent_pos

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Sub-agent 1: Stats [0.4, 1.0]
im0 = axes[0, 0].imshow(v0[0])
axes[0, 0].scatter(
    resource_loc[:, 1], resource_loc[:, 0],
    marker="X", color="red", s=100
)
axes[0,0].scatter(
    agent_position[1], agent_position[0],
    marker ="o", color="green", s = 100
)
axes[0, 0].set_title("Sub-agent 1: Stats [0.4, 1.0]")

# Sub-agent 1: Stats [1.0, 1.0]
im1 = axes[0, 1].imshow(v1[0])
axes[0, 1].scatter(
    resource_loc[:, 1], resource_loc[:, 0],
    marker="X", color="red", s=100, 
    label="Resources"
)
axes[0,1].scatter(
    agent_position[1], agent_position[0],
    marker ="o", color="green", s = 100,
    label="Agent"
)
axes[0, 1].set_title("Sub-agent 1: Stats [1.0, 0.4]")

# Sub-agent 2: Stats [0.4, 1.0]
im2 = axes[1, 0].imshow(v0[1])
axes[1,0].scatter(
    resource_loc[:, 1], resource_loc[:, 0],
    marker="X", color="red", s=100, 
    label="Resources"
)
axes[1,0].scatter(
    agent_position[1], agent_position[0],
    marker ="o", color="green", s = 100,
    label="Agent"
)
axes[1, 0].set_title("Sub-agent 2: Stats [0.4, 1.0]")

# Sub-agent 2: Stats [1.0, 0.4]
im3 = axes[1, 1].imshow(v1[1])
axes[1, 1].scatter(
    resource_loc[:, 1], resource_loc[:, 0],
    marker="X", color="red", s=100, 
    label="Resources"
)
axes[1,1].scatter(
    agent_position[1], agent_position[0],
    marker ="o", color="green", s = 100,
    label="Agent"
)
axes[1, 1].set_title("Sub-agent 2: Stats [1.0, 0.4]")

fig.colorbar(im3, ax=axes)

plt.show()


h = np.array([0.4, 1.0])

for i in range(agent.n_stats):

    print(f"\nSub-agent {i + 1}")

    for j, pos in enumerate(resource_loc):

        state = agent.encode(pos, h)
        q = agent.Q[i, state, :]

        print(f"Resource {j + 1} at {pos}")
        print("Q-values:", q)
        print("State value:", np.max(q))

H = np.array([entry[2] for entry in log])      # shape (n_steps, n_stats)
print("min per stat:", H.min(axis=0))
print("max per stat:", H.max(axis=0))

bins = np.clip(((H - agent.h_min) / (agent.h_max - agent.h_min) * agent.n_bins).astype(int),
               0, agent.n_bins - 1)

print("fraction in lowest bin: ", np.mean(bins == 0, axis=0))
print("fraction in highest bin:", np.mean(bins == agent.n_bins - 1, axis=0))

print("fraction below h_min:", np.mean(H < agent.h_min, axis=0))
print("fraction above h_max:", np.mean(H > agent.h_max, axis=0))

ep = np.array([e[0] for e in log])
late = ep >= n_episodes - 200
H_late = H[late]
print("late below h_min:", np.mean(H_late < agent.h_min, axis=0))
print("late above h_max:", np.mean(H_late > agent.h_max, axis=0))
print("late min/max:", H_late.min(axis=0), H_late.max(axis=0))

lengths = np.bincount([e[0] for e in log])
print("mean length, first 200 eps:", lengths[:200].mean())
print("mean length, last 200 eps:", lengths[-200:].mean())
print("episodes shorter than 2000:", np.mean(lengths < 2000))



n_eval_episodes = 5
eval_runs = []

for k in range(n_eval_episodes):          # k, not ep: your analysis code later reassigns `ep`
    env.reset()
    agent.make_stats()
    agent.stats = np.array([1.0, 1.0]) #to fix offset of stats at the beginning of the evaluation
    loc = env.agent_pos.copy()

    locs, stats_hist, actions, on_res_hist = [], [], [], []

    for t in range(env.episode_length):
        a = agent.select_action(loc, agent.stats, training=False)
        locs.append(loc.copy())            # position the action was chosen in
        actions.append(a)

        loc_next, on_resource, done = env.step(a)
        h_prev, rewards, dead, dead_per_stat = agent.organism_step(on_resource)

        stats_hist.append(agent.stats.copy())      # stats after this step
        on_res_hist.append(on_resource.copy())     # same step as the stats above

        loc = loc_next
        if dead or done:
            break

    eval_runs.append({
        "locs": np.array(locs),            # (T, 2)
        "stats": np.array(stats_hist),     # (T, n_stats)
        "actions": np.array(actions),      # (T,)
        "on_resource": np.array(on_res_hist),  # (T, n_stats)
        "died": bool(dead),
    })


run = eval_runs[0]
n = 300                                     # window; use len(run["stats"]) for everything

# align: stats[t] and on_resource[t] are AFTER action t, locs[t+1] is where action t landed
stats = run["stats"][:-1][:n]
on_res = run["on_resource"][:-1][:n]
locs = run["locs"][1:][:n]
time = np.arange(len(stats))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 6))

# top: internal stats
for i in range(agent.n_stats):
    ax1.plot(time, stats[:, i], color=f"C{i}", label=f"stat {i + 1}")
    hit = np.flatnonzero(on_res[:, i])      # steps spent on resource i
    ax1.scatter(hit, stats[hit, i], color=f"C{i}", s=25, zorder=3)
ax1.axhline(agent.set_point, color="k", lw=1, label="set point")
ax1.axhline(agent.h_min, color="r", ls="--", lw=1, label="death bounds")
ax1.axhline(agent.h_max, color="r", ls="--", lw=1)
ax1.set_ylabel("internal stat")
ax1.legend(loc="upper right", ncol=4, fontsize=8)

# bottom: location
ax2.plot(time, locs[:, 0], label="row")
ax2.plot(time, locs[:, 1], label="col")
ax2.set_ylabel("grid coordinate")
ax2.set_xlabel("time step")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()



from matplotlib.animation import FuncAnimation, PillowWriter   # needs: pip install pillow

run = eval_runs[0]
n = 300
stats = run["stats"][:-1][:n]
locs = run["locs"][1:][:n]
T = len(stats)
G = agent.grid_size

fig, (axg, axs) = plt.subplots(1, 2, figsize=(10, 4.5))

# static parts, drawn once
axg.set_xlim(-0.5, G - 0.5)
axg.set_ylim(G - 0.5, -0.5)                 # row 0 on top
axg.set_aspect("equal")
axg.set_xticks(range(G)); axg.set_yticks(range(G))
axg.grid(True)
for i, (r, c) in enumerate(env.resource_pos):
    axg.scatter(c, r, marker="X", s=250, color=f"C{i}", label=f"resource {i + 1}")
axg.legend(loc="upper center", ncol=2, fontsize=8)

for i in range(agent.n_stats):
    axs.plot(np.arange(T), stats[:, i], color=f"C{i}", label=f"stat {i + 1}")
axs.axhline(agent.set_point, color="k", lw=1)
axs.axhline(agent.h_min, color="r", ls="--", lw=1)
axs.axhline(agent.h_max, color="r", ls="--", lw=1)
axs.set_xlabel("time step")
axs.legend(loc="upper right")

# moving parts, created once
dot, = axg.plot([], [], "o", color="black", markersize=14)
vline = axs.axvline(0, color="gray")
label = axg.text(0.5, -0.12, "", transform=axg.transAxes, ha="center")

def update(frame):
    r, c = locs[frame]
    dot.set_data([c], [r])                  # set_data wants sequences
    vline.set_xdata([frame, frame])
    label.set_text(f"t={frame}   stats = {np.round(stats[frame], 2)}")
    return dot, vline, label

anim = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
anim.save("rollout.gif", writer=PillowWriter(fps=10))
plt.close(fig)