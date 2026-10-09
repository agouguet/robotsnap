# RobotSNAP

**Robot Social Navigation Assessment Platform** - a Unity simulator in which a
mobile robot moves among pedestrians, and the Python package that drives it,
watches it and scores it.

RobotSNAP exists to answer one question: *how does a navigation policy actually
behave around people?* It gives one place to replay the same scenarios, measure
the same social metrics, and compare methods that have nothing else in common -
a policy you trained yourself, a classic method from the literature, or a
navigation stack running as a ROS2 node.

## Start here

Pick the line that matches what you want to do. Each one below says what to
install, the command to run, and where the detail lives.

| I want to... | install | start with |
| --- | --- | --- |
| **train a policy / study RL** | `pip install -e ".[env,train]"` | `robotsnap train` |
| **replay the full benchmark** | `pip install -e ".[env]"` | `robotsnap benchmark --suite basic` |
| **benchmark my own policy (RL)** | `pip install -e ".[env,train]"` | `robotsnap benchmark --load` |
| **benchmark my own stack (ROS2 or other)** | `pip install -e ".[env]"` + [`robotsnap-ros2`](https://github.com/agouguet/robotsnap-ros2) | `robotsnap benchmark --policy external` |
| **connect a ROS2 node** | [`robotsnap-ros2`](https://github.com/agouguet/robotsnap-ros2) | `robotsnap bridge --ros2` |
| **watch or script a session** | `pip install -e ".[viewer]"` | `robotsnap watch` |
| **write my own task or method** | `pip install -e ".[env,train]"` | `models/template.py` |

Everything runs against the Unity simulator, which lives in its own repository:
[`robotsnap-unity`](https://github.com/agouguet/robotsnap-unity). Start the
bridge first, then press Play in the application - the scene dials
`127.0.0.1:10000`.

```bash
python -m robotsnap bridge --port 10000     # then press Play in Unity
python -m robotsnap goal --episodes 3       # a sanity check: drive to the goal
```

One flag starts the application for you and closes it again when the run ends:
`python -m robotsnap train --launch --headless --viewer` opens Unity, trains,
and stops the simulation and the app on Ctrl-C. See
[docs/manual.md](docs/manual.md) for `--unity-app` and the discovery order.

## Train a policy (reinforcement learning)

One flag picks the learning rule, one command trains it, and the same
environment feeds all of them. Stable-Baselines3 brings PPO, A2C, SAC and DQN;
the package also ships its own small PyTorch loop.

```bash
# a continuous-action rule over the plain environment
python -m robotsnap train --algo ppo --timesteps 200000

# a named method of the literature: its own observation, actions and reward
python -m robotsnap train --method cadrl --episodes 2000

# each run prints the checkpoint it wrote; replay it later at real time
python -m robotsnap play --load policy/cadrl-20261006-181500.pt --episodes 3 --render
```

- Checkpoints go to `policy/`, named after the run and the moment, so two runs
  never overwrite each other. `--save <path>` picks your own, `--no-save`
  trains without writing one (see `policy/`).
- Hyperparameters come from a file: `--config ppo`, `--config cadrl` (see
  `configs/`). A whole ladder of scenarios is available with `--curriculum`.
- Training stops the simulation in Unity when it ends, exactly like the red
  stop button; `--no-stop` leaves it running.
- The full guide: [docs/manual.md](docs/manual.md).

## Use it as a Gymnasium environment

```python
from robotsnap.envs import RobotSNAPEnv

env = RobotSNAPEnv(scenario="python_env_demo")
observation, info = env.reset(seed=0)
done = False
while not done:
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

One episode is one navigation problem. Everything the simulator does lives in
the base class; everything the *task* does - what the robot sees, what it is
paid, when the episode ends - is a method you can replace in a subclass.

## Replay the full benchmark

`benchmark` plays a policy over a suite of scenarios, averages the metrics Unity
reports, and writes the campaign to a file. `--compare` turns two of those files
into one table, so a method can be put next to any other.

```bash
# the six basic social-navigation scenarios, five episodes each
python -m robotsnap benchmark --suite basic --episodes 5 --out results/goal.json

# the same suite against the reference methods
python -m robotsnap benchmark --suite basic --load cadrl.pt --out results/cadrl.json
python -m robotsnap benchmark --suite basic --load sarl.pt  --out results/sarl.json

# one table
python -m robotsnap benchmark --compare results/goal.json results/cadrl.json results/sarl.json
```

A campaign file keeps every episode as well as the averages, so a policy that
wins one scenario and fails the others is visible as such.

## Benchmark your own policy

Point `--load` at a checkpoint and the run reads back how it was trained -
Stable-Baselines3, this package's own policy, or a named method - so `--algo`
and `--method` are only needed to override the file.

```bash
python -m robotsnap benchmark --suite basic --load my_policy.zip --out results/mine.json
python -m robotsnap benchmark --compare results/goal.json results/mine.json
```

## Benchmark a stack that is not Python

The robot does not have to be driven by this package. With `--policy external`,
RobotSNAP still resets the scenarios, paces the world and records the same
metrics, while the velocity comes from an outside process - typically a ROS2
node, possibly in a container.

```bash
pip install "RobotSNAP[env]" robotsnap-ros2

python -m robotsnap bridge --ros2            # Unity + the ROS2 graph
# ... then start your navigation node, and let it publish /cmd_vel
python -m robotsnap benchmark --policy external --suite basic --out results/ros2.json
```

No PyTorch, no display: this profile needs `gymnasium` and nothing else.
The wiring is in
[`robotsnap-ros2/docs/ros2-with-a-docker-method.md`](https://github.com/agouguet/robotsnap-ros2/blob/main/docs/ros2-with-a-docker-method.md).

## Connect ROS2

`robotsnap-ros2` is the two-way bridge between a running session and a ROS2
graph: it publishes what Unity sends and writes back what the graph publishes.
The core never imports ROS2 - it loads the transport through an entry point - so
a machine without ROS installs and runs everything else unharmed.

```bash
pip install "RobotSNAP[env]" robotsnap-ros2
source /opt/ros/humble/setup.bash

python -m robotsnap bridge --ros2             # serve Unity and mirror it on the graph
python -m robotsnap watch --ros2              # the viewer, reading the graph
python -m robotsnap play --transport ros2 --load cadrl.pt --episodes 3
```

## Watch or script a session

```python
from robotsnap.client import RobotSNAPClient

with RobotSNAPClient(port=10000) as client:   # owns the bridge, stops it on exit
    client.wait_until_ready()
    client.play()
    print(client.snapshot()["sim_time_seconds"], client.humans(), client.agents())
```

`watch` draws the live 2D view; the client facade reads the session, sends
velocity, drives the crowd and writes scenarios. The reference for every topic
and body is
[docs/robotsnap-unity-contract.md](docs/robotsnap-unity-contract.md).

## The methods that ship

Each method is one file under `robotsnap/models/`, holding everything that makes
it that method: its observation, its action set, its reward and its policy.

| method | what it is |
| --- | --- |
| `cadrl` | Chen et al., 2017 - the classic crowd-aware value agent |
| `sarl` | Chen et al., 2019 - the same learner with attention over the crowd |
| `ga3c_cadrl` | Everett et al., 2018 - the actor-critic successor |
| `rgl` | Chen et al., 2020 - a relational graph and a multi-step lookahead |
| `template` | a small worked example to copy when writing your own |

## Write your own

- **A new task or reward**: subclass the environment and override the handful of
  methods listed in the guide.
- **A new method**: copy `models/template.py`, keep the parts it shares with the
  others, and register one name.
- **An existing stack**: leave it where it is and benchmark it as an external
  policy, as above.

## Install profiles

The bridge and the client only need a pure-Python codec, so a session can be
driven without ROS, PyTorch or a display. Nothing else is pulled in unless you
ask for it.

```bash
git clone https://github.com/agouguet/robotsnap
cd robotsnap
python -m venv .venv && source .venv/bin/activate

pip install -e .             # bridge and client: drive or read a session
pip install -e ".[viewer]"   # + the pygame window
pip install -e ".[env]"      # + the Gymnasium environment (run an episode)
pip install -e ".[train]"    # + PyTorch, for training and inference
pip install -e ".[sb3]"      # + Stable-Baselines3 (PPO, DQN, ...)
```

## Documentation

- [docs/manual.md](docs/manual.md) - the full guide: commands, the clock and
  pacing, hyperparameters, curriculum, the environment, the metrics.
- [docs/robotsnap-unity-contract.md](docs/robotsnap-unity-contract.md) - the
  Unity-side reference: the topics, their bodies and the commands.
- `examples/` - a scenario driven end to end, the environment used three ways,
  and the benchmark suite written as plain library calls.

## Tests

No Unity and no ROS are needed: the suite ships a fake peer that speaks the wire
protocol over a local socket.

```bash
python -m pytest -q
```
