# Nautilus Onboarding — Task 5 Writeup (RL on Nautilus)

## What this task was

Task 5 was about running reinforcement learning algorithms on Nautilus. The full assignment had three parts: read the Gymnasium and Stable Baselines 3 documentation, train a baseline RL algorithm on the Car Racing Gymnasium environment with TensorBoard logging, and run CARLA Gym's `run.py` for comparison. The deliverable was a one-page writeup describing the results, plus plots showing the training curves and a link to a GitHub repo with the code.

I focused on the Car Racing portion, since CARLA Gym's `train.py` had hung in env initialization during Task 4 and the underlying RL skills are the same regardless of which environment trains on. The CARLA Gym attempt is documented in the Task 4 writeup.

## Background

Gymnasium is the maintained replacement for OpenAI Gym, which has been unsupported since 2022. It provides a standard interface for RL environments where every env exposes `reset()`, `step(action)`, `observation_space`, and `action_space`. Stable Baselines 3 (SB3) is the matching maintained replacement for the original Stable Baselines, offering implementations of standard RL algorithms (PPO, SAC, A2C, DQN) that you point at any Gymnasium env. Together they form the standard RL pipeline.

Car Racing is a Gymnasium environment where the agent controls a 2D car through a procedurally generated racetrack. Observations are 96x96 RGB images of the car's view. Actions are three continuous values: steering, gas, and brake. The reward is -0.1 per timestep (a small penalty for taking time) plus +1000/N per new track tile visited, where N is the total number of tiles. The episode ends when the agent visits all tiles, fails badly, or hits the 1000-step time limit.

## Training setup

I ran PPO with the CnnPolicy (a convolutional network since observations are images) for 300,000 timesteps on a Nautilus Job with a single GTX 1080 Ti GPU. The infrastructure pattern reused everything from Task 3 with one optimization: rather than pip installing dependencies at the start of every Job, I pre-installed them onto the PVC once using a separate installer pod and the `pip install --target=` flag. The Job then set `PYTHONPATH=/pvcvolume/python-packages` so the pre-installed dependencies were findable. This saved 2-3 minutes of pip install time per run and matches the pattern real research workflows use on shared clusters.

Training took 56 minutes wall time. The agent completed 300 full episodes over the 300K timesteps, with the script using a custom logging callback to write per-episode rewards to TensorBoard.

## Errors I hit and how I fixed them

**Missing `rollout/ep_rew_mean` in TensorBoard.** My first training run completed 94,000 timesteps without ever producing the canonical reward curve metric. Investigation showed that SB3's automatic logging of episode rewards requires the env to be wrapped in a `Monitor` wrapper before being passed to `DummyVecEnv`. Without it, the vectorized env was dropping the episode info dicts that SB3 needs to compute averages. Fixed by adding the Monitor wrapper to the env factory and also writing a custom callback that pulls episode rewards from the `infos` dict directly and logs them under a `custom/` prefix. This guaranteed reward data would appear in TensorBoard regardless of any quirks in SB3's automatic logging.

**Stale local TensorBoard cache.** Several times when I re-ran `kubectl cp` to refresh logs, the new local file was clearly different size than the previous version but TensorBoard kept showing old data. The fix was to fully delete the local logs directory (`rm -rf tb_logs`) before re-copying, then kill and restart TensorBoard before refreshing the browser. The Python TensorBoard process caches event file state and doesn't always pick up overwrites cleanly.

**`kubectl cp` failing on large transfers.** Copying the `models/` directory (about 290MB of checkpoint files) consistently errored out partway through with "read message: unexpected EOF." The workaround is to copy single files instead of the whole directory and pass `--retries=5`. For very large files, piping through `tar` via `kubectl exec` works when `kubectl cp` doesn't, since tar streaming is more resilient to transient connection drops.

**Completed pod has no exec.** Once training finished and the Job pod entered `Succeeded` state, I could no longer `kubectl cp` from it because the container had exited. The fix was to spin up a separate shell pod mounting the same PVC, then copy files out of that pod. Lesson: anything you want to preserve from a Job pod should be copied while the pod is still running, or you need to mount the PVC into a second pod afterward.

**GitHub authentication failure.** Pushing to GitHub failed with "Password authentication is not supported for Git operations." GitHub no longer accepts your account password for git pushes; you need a Personal Access Token from https://github.com/settings/tokens with the `repo` scope. Pasting the token as the password works fine.

## Results

The final 100-episode mean reward was **+139.70**, achieved at step 300,032 after 300 complete episodes. The training curve shows three distinct phases that line up cleanly with what theory predicts for PPO on Car Racing:

Phase 1 (0 to 100K steps) is the initial descent and plateau. The agent quickly learns that random actions yield large negative rewards from going off-track. It then converges to a stable but suboptimal policy of moving slowly in place, collecting the -0.1 per step penalty over the full 1000-step episode for a steady reward around -65. This is the classic "circle of safety" local minimum where the agent has learned to avoid punishment but not to gain reward.

Phase 2 (100K to 200K steps) is the gradual breakthrough. The reward curve climbs from -65 to about +25 as the agent starts to discover that driving forward on the track yields more reward than the time penalty costs. Episodes still hit the 1000-step time limit but the agent is now collecting meaningful tile rewards.

Phase 3 (200K to 300K steps) is rapid improvement. The reward accelerates from +25 to +140 as the agent develops genuine driving competence. At a final mean reward of +139.7, the agent is collecting roughly 60 tiles per episode out of about 280 total, or about 21% of the track on average.

The complete training run, with all event logs, plots, monitor CSV, model checkpoints, and Kubernetes manifests, is on GitHub at [https://github.com/ndontubo/car-racing-ppo-nautilus](https://github.com/ndontubo/car-racing-ppo-nautilus).

## What I learned

The most useful technical lesson was that the Monitor wrapper is not optional for RL training that you actually want to observe. SB3's automatic logging depends on it, and without it the canonical reward curves silently fail to appear. The defensive pattern of also writing a custom callback that pulls from `infos` directly is worth doing on any training script you care about, because it's the difference between "I trained for an hour and have no idea what happened" and "I have a clean reward curve to discuss."

The broader lesson is about how RL training actually unfolds. The curve I produced is not a steady upward climb. It's a long plateau followed by a sudden breakthrough. If I had stopped training at 100K timesteps because nothing was happening, I would have concluded PPO doesn't learn Car Racing. The truth is that PPO needs time to escape the local minimum, and the breakthrough comes well after the period when training looks dead. This argues for committing to longer training runs than feels comfortable and for not interpreting flat reward curves as failure too early.

The Option B PVC install pattern (`pip install --target=/pvcvolume/...`) is something I will reuse for every future Nautilus RL project. The first installer run was the same speed as a normal pip install, but every subsequent Job started in seconds instead of minutes, and the dependency state was reproducible across restarts of the Desktop GUI environment. This is the real workflow shape for iterative ML research on shared clusters, much closer to how production ML teams work than the per-job pip install I used in Task 3.

##CARLA Gym attempt
After completing the Car Racing baseline, I modified run.py (made a new file called run_nav.py) from carla-gym-env to train SAC from scratch instead of loading a pretrained model that didn't exist on the system. After two GPU OOM crashes (CARLA simulator + SAC competing for an 11GB shared 1080 Ti), I moved SAC to CPU. This succeeded: the training pipeline ran end-to-end, completing 4 episodes over 222 timesteps with the rollout/, train/, and time/ TensorBoard metrics all populating. At ~0.35 fps (the CPU bottleneck), the run was too slow to reach meaningful learning, but it demonstrated the full CARLA Gym + SB3 pipeline working. Logs from all three attempts are in carla_gym/carla_tb_logs/.
