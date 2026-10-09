# RobotSNAP

Robot Social Navigation Assessment Platform: a Unity simulator of a mobile
robot moving among pedestrians, and the Python companion that drives it and
reads it back.

Unity runs `com.unity.robotics.ros-tcp-connector` and dials a TCP *server*, so
the Python side is what listens. What Unity dials can be a ROS2
`ros_tcp_endpoint` or this package's bridge: the wire protocol is the same, and
the Python bridge needs no ROS installation.

## Install

```bash
pip install -e .
```

The bridge and the client import `rosbags` only, a pure-Python CDR codec;
`pygame` is imported lazily by the viewer's entry point, so a headless session
never touches a display.

## One entry point

Every run the repository has lives behind one command: `python -m robotsnap
<command>` is a single door onto the environment, the viewer and the scenario
round trip, and `pip install -e .` puts the same entry point on the path as
`robotsnap`. The `examples/` scripts keep their own command lines and call the
same functions, so the two cannot drift apart.

| command | what it runs |
| --- | --- |
| `viewer` | the live pygame view of a session |
| `episode` | one random-action episode of the goal-navigation environment |
| `goal` | the social task with a scripted go-to-goal controller |
| `train` | the REINFORCE loop (needs `torch`) |
| `play` | play a saved policy back, without training (needs `torch`) |
| `scenario` | write a scenario, launch it, drive it, freeze it |
| `bench` | time N steps: simulated seconds against wall seconds |

Every command also takes `--launch`: it starts the RobotSNAP application for
the run and stops it when the run ends, so one command covers the whole stack
(see *One command that starts everything* below).

```bash
python -m robotsnap episode --port 10000 --observations "pose,goal"
python -m robotsnap goal --episodes 3 --observation-structure dict
python -m robotsnap bench --steps 200 --time-scale 5
```

`python -m robotsnap <command> --help` lists what one run takes. The
environment's observation is chosen with `--observations` (a comma-separated
list of components, or a JSON object that also carries their parameters), tuned
with `--observation-params` (a JSON object, for example
`'{"lidar": {"bins": 24}}'`), and handed out either flat or as a dict of named
arrays with `--observation-structure`.

## One command that starts everything

Unity is normally opened by hand and driven by a command that dials it. On a
machine where the application is installed, one flag collapses the two steps:
`--launch` starts the RobotSNAP player, points the bridge at it, and stops both
when the run ends.

```bash
python -m robotsnap train --launch --headless --viewer --algo ppo \
    --timesteps 200000 --time-scale 100 --save policy/ppo.zip
```

The player is discovered from `--unity-app` first, then `$ROBOTSNAP_UNITY_APP` -
both name the player itself or a directory that holds it - then the layout the
workspace installer writes (`$ROBOTSNAP_HOME/app/robotsnap-unity/<version>`, and
the `~/robotsnap-workspace` clone beside the home directory), then the usual
install prefixes (`~/.local/share/robotsnap-unity`,
`~/.local/bin/robotsnap-unity`, `/opt/robotsnap-unity`, ...), then the `PATH`,
and finally the source project's own `Builds/`. A path named on the command line
is an instruction: when it holds no player the run stops with that path in the
message instead of quietly starting another install.

An installed application reads its scenarios from the data folder beside its
executable, not from the source project, so a run started with `--launch` reads
and writes them there: the scenarios the application ships are used as they
stand rather than written over, and `--unity-project` still overrides the folder
when the caller knows which one the session reads. An installation that cannot
be written to - a system-wide one under `/opt`, say - can only play the
scenarios it already ships, and a run that asks for another name is refused
before the session opens, with the folder that refused it; name one of the
scenarios it ships (`--scenario default`) or install the application under your
home, where the folder belongs to you, and every scenario works.

That is also why an edit made in the editor does not show up in a run started
with `--launch`: the editor reads the project's `Assets/StreamingAssets`, and a
built player reads the copy baked into its own data folder. Refreshing the
build (`setup-workspace.sh --build-unity`) is what puts the edit in the copy the
player reads; until then the run says so before it starts, as
`scenario note : the player reads <the player's file>, not the project's
<the project's file>`.

`--headless` starts the player with `-batchmode -nographics` : no window and
much faster, and still driveable over ssh. Pair it with `--viewer` to watch the
run in the package's own 2D window, and with `--unity-log PATH` to say where the
player writes its log - a temporary file by default when `--headless`. A player
that exits while starting is reported right away, with the last lines of that
log, rather than after the run's wait timeout.

Stopping the command with Ctrl-C (or a `SIGTERM`) stops the run, puts the
simulation back in standby, and closes the application it started. Every
sub-command takes `--launch`, so the same flag drives `play`, `bench` and
`benchmark` too.

## Load a policy

A training run writes a checkpoint that carries everything needed to play it
back: the weights, the width of the network, and the statistics the observations
were normalised by. Those statistics travel in the file, so a run that replays
it does not have to guess them. `play` reads the checkpoint and drives the
session without training:

```bash
python -m robotsnap train --episodes 20
# weights written to policy/reinforce-20261006-181500.pt

python -m robotsnap play --load policy/reinforce-20261006-181500.pt --episodes 3 --render
```

`train` writes into `policy/` when the line does not say where, under a name
that carries the run and the moment, so two runs of the same command never
overwrite each other and every run prints the file it wrote. A path still wins:
`--save runs/today.zip` writes exactly there, and `--save policy/` generates a
name in that directory. `--no-save` trains without writing a checkpoint. The
suffix is the writer's own - `.pt` for this package's trainers, `.zip` for
Stable-Baselines3 - and `policy/README.md` has the whole rule.

The action is the mean of the policy unless `--sample` asks for a draw from it.
`play` and `train` stop the simulation in Unity once they end, exactly as the
application's red stop button does: the clock is paused, the agents leave the
scene and the application is left in standby (`Ready`), the map and the scenario
kept. Pass `--no-stop` to leave the simulation running instead.

## Run a session without ROS

```bash
python -m robotsnap.bridge --port 10000
```

Start that first, then press Play in Unity: the scene connects to
`127.0.0.1:10000`. The runner prints one summary line per tick, announces each
topic as it appears and lists the inventory on exit; Ctrl+C stops it cleanly.

## Drive the session from Python

```python
from robotsnap.client import RobotSNAPClient

with RobotSNAPClient(port=10000) as client:   # owns the bridge, stops it on exit
    client.wait_until_ready()                 # Unity is connected and has registered its topics
    client.play()                             # one method per command
    print(client.snapshot()["sim_time_seconds"], client.humans(), client.agents())
```

`snapshot()`, `humans()`, `agents()` and `robot()` read the session,
`state(topic)` returns the last decoded message of any topic (`/odom`, `/scan`,
`/map`, ...), `send_cmd_vel()` writes `/cmd_vel`, and `set_human_velocities()`,
`stop_human()` and `stop_humans()` drive the crowd. The reference for every
topic and body is [robotsnap-unity-contract.md](robotsnap-unity-contract.md).

`stop_simulation()` is the application's red stop button: the clock is paused,
every agent is taken out of the scene and the application goes back to standby
(`Ready`), the map and the scenario kept. `stop_scenario()` only freezes the
session where it stands - it pauses, parks every robot of the roster and stops
the crowd, but keeps every agent in the scene.

## Watch the session

```bash
python -m robotsnap.viewer --port 10000 --rate 30
```

The viewer starts the bridge itself and draws the occupancy grid, the robot and
its heading, the crowd, the lidar returns and the robot goal, with a HUD for the
run state, the scenario, the simulated time, the time scale and the agent count.
It says what it waits for until Unity connects; Esc, `q` or the window close
button quits.

## Write, launch and stop a scenario from Python

Unity reads a scenario from `Assets/StreamingAssets/Scenarios/<name>.yaml` and
names it after that file, so creating one is writing that file.
`robotsnap.scenario` builds and writes it without a session running, and the
client's `create_scenario` is the two steps in one call:

```python
from robotsnap import scenario
from robotsnap.client import RobotSNAPClient

with RobotSNAPClient(port=10000) as client:
    client.wait_until_ready()
    client.create_scenario(
        "python_demo",
        map_name="basic/crowd",
        robots=[scenario.robot("robot_1", "jackal", (-7.0, 0.0, 0.0), (7.0, 0.0))],
        humans=[
            scenario.crowd(
                "pedestrians", 5,
                spawn=scenario.spawn_zone(0.0, -5.0, 3.0, 3.0),
                goal=scenario.spawn_zone(0.0, 5.0, 3.0, 3.0),
            )
        ],
        launch=True,
    )
    print(client.robots(), client.scenario_time_seconds)
    client.stop_scenario()
```

A robot's start, goal and waypoints are `(x, z)` or `(x, z, yaw)`, in the Unity
world axes the scenario editor writes rather than the ROS frame the streams
publish, and each one becomes a named entry of `points`. `map_name` is either a
map such as `basic/crowd` or the name of a collection such as `basic`, which
draws a random map of it. `robot()` needs one of the type ids of the project's
catalog: `bibus`, `freight`, `ginger`, `jackal`, `kuri`.

`launch_scenario` sends `load_scenario` and then waits for the world to be
built - the command alone only means the request was accepted - and
`stop_scenario` pauses the session, parks every robot of the roster and stops
the crowd, leaving the world in the scene. `examples/scenario_control.py` is
this round trip as a script; it starts the bridge, waits for Unity, writes a
scenario, drives the robot and freezes it again.

```bash
python examples/scenario_control.py --port 10000
```

It listens before Unity dials, so start it first and press Play afterwards.

### The clock is not the episode

`snapshot()["sim_time_seconds"]` is the simulator's own clock, and it belongs to
the Unity session rather than to the scenario: the `Clock` component is
`[ExecuteAlways]` and takes its origin when it is enabled - in the Editor, when
the scene was loaded, unless entering Play mode reloads the domain - and a
scenario load does not put it back to zero. Measured live: a scenario applied
just after Play mode started reported `sim_time_seconds` at 610 s while the
scenario itself was 0 s old.

`client.scenario_time_seconds` is that clock minus the value it had when the
scenario was applied, the mark `launch_scenario` and `wait_for_scenario` take;
after a bare `reset()`, whose acknowledgement arrives before the world is
rebuilt, call `mark_scenario_start()` yourself. Both are the scaled clock of the
simulator: with `set_time_scale(2)`, one second of them is two seconds of the
wall.

### Running the world faster than the wall

The two pacing modes answer two different questions, and the environment passes
the choice through: `pacing="free"` (the default of the demos) lets the world run
at its time scale while the policy thinks, and `pacing="lockstep"` (the default
of `train` and `play`) has Unity spend exactly one control period per step and
stop in between.

Free is what makes a session look alive in a viewer; lockstep is what makes a
control period mean something. The difference is not cosmetic: a policy that
takes fifty milliseconds to answer while the world runs at ten times speed has
let half a second of the world go by on the previous command - two and a half
control periods - so the same code trains a different task at every skill level.
In lockstep a slow model costs wall time and nothing else, and an episode cannot
run past the world time it was given.

Two ceilings decide how fast a lockstep session can actually run, and both are
measured rather than guessed:

* **The scale a control period can hold.** Unity advances the world on the drawn
  frame, and a frame carries at least one physics step, so a period of `P`
  seconds cannot hold more than `P / fixedDeltaTime` times speed. Measured live:
  a 0.2 s period at scale 100 on a 0.02 s step spent 5.000 s of simulation in
  the single frame that followed a release - twenty-five periods - and the
  episode's budget was gone before the pacing gate was looked at again. The gate
  now holds the scale at that ceiling and says so in its answer:

  ```bash
  python -m robotsnap train --time-scale 100 --control-period 0.2
  # lockstep, 0.2 s of simulation per release at scale 10 (asked 100; a frame of
  # 0.02 s physics cannot spend less than 0.2 s of simulation, so 0.2 s of period
  # holds 10 times speed - raise the control period to raise the scale)
  ```

* **The frame rate.** In lockstep the world only moves inside a released period,
  and the gate stops it at the next frame boundary, so a step costs about two
  frames whatever the time scale is. The speed a session delivers is therefore
  about `frame_rate * control_period / 2` seconds of simulation per wall second,
  and the control period - not the time scale - is the knob. Measured on one
  editor session rendering at 1.7 fps: a 0.2 s period delivered 0.19x real and a
  2 s period 1.89x, both inside their budget. At a focused editor's 30 fps the
  same two periods are worth about 3x and 30x.

Free pacing is not bound by either ceiling, because a frame may catch up on all
the wall time it owes - up to a ceiling the router keeps at ten seconds of
simulation - which is why it reaches a higher wall speed; what it does not give
is a control period that survives a slow policy. The two recipes that follow
from this:

```bash
# Train as fast as the machine allows, with a control period that is still a
# period: every step is 2 s of world time, and the episode budget is exact.
python -m robotsnap train --episodes 200 --control-period 2 --pacing lockstep \
    --time-scale 100 --seconds 120 --save policy/fast.pt

# Watch the result at real time.
python -m robotsnap play --load policy/fast.pt --control-period 0.2 --time-scale 1 \
    --pacing lockstep --render
```

A heavier model - SARL or CADRL, which plan over the neighbours' future
trajectories before they emit one action - costs wall time per step and nothing
else, provided the pacing is lockstep: the delay lands *between* two releases, so
the world is stopped while the model thinks. What it does change is the
simulation seconds a training run covers per hour, in proportion to the extra
thinking time; a 30 ms decision under a 2 s control period is 1.5% of the step,
and the same decision under a 0.2 s period is 15% of it. A model that needs to
run *inside* the physics loop (a learned collision checker consulted every
physics step, say) is a different matter: it is then part of the per-step cost
and multiplies with the number of physics steps a run asks for, which is what
`--control-period` and the scale decide.

### Headless, which is where the speed is

Measured on this machine, and the numbers are the argument: the same scene in
play mode runs its frame loop at **68 fps with the editor window and 4459 fps
with `-batchmode -nographics`**. Rendering is not what the world does - the
physics, the lidar raycasts and the publishers are - and it is the largest thing
between a training run and the wall clock. Start the editor headless before a
long run:

```bash
/home/adam/Unity/Hub/Editor/6000.4.4f1/Editor/Unity -batchmode -nographics \
    -projectPath /home/adam/robotsnap-workspace/robotsnap-unity \
    -logFile /tmp/unity_headless.log &

# then, once `unity status` reports it ready, drive it exactly as usual
python -m robotsnap train --algo ppo --timesteps 200000 --control-period 2 \
    --time-scale 100 --pacing lockstep --save policy/ppo.zip
```

The same session can also be opened normally and driven from Python; the two
differ in speed, not in what they are. Two measured episodes of 60 s of world
time, headless, both with the budget held:

| what | delivered | wall |
| --- | --- | --- |
| `free`, x10, 0.2 s period | 10.0x real | 6.0 s |
| `lockstep`, x20, 2 s period | 6.0x real | 10.2 s |

Free paces the world at whatever it manages and lets it drift while the policy
thinks; lockstep holds the control period exactly. What neither escapes is the
physics: some two thousand physics steps per second is what this scene costs
here, so the simulated seconds a run covers per hour are bounded by that, and
the levers are the physics step (`--control-period` and the scene's fixed
timestep), the size of the crowd, and the number of streams the client asks for.

## Algorithms

One flag picks the algorithm, one command trains it, and the same environment
feeds all of them:

| `--algo` | what it is | needs |
| --- | --- | --- |
| `reinforce` | the loop of this package, a small PyTorch policy | `pip install -e ".[train]"` |
| `ppo`, `a2c`, `sac`, `dqn` | stable-baselines3 over the same environment | `pip install -e ".[sb3]"` |
| `cadrl`, `sarl`, `ga3c_cadrl`, `rgl` | the social-navigation methods of `robotsnap.models`, on their own axis: `--method cadrl` | `pip install -e ".[train]"` |

```bash
# a continuous-action algorithm as it stands
python -m robotsnap train --algo ppo --timesteps 200000 --save policy/ppo.zip
# a value-based one, which needs the discrete command set
python -m robotsnap train --algo dqn --timesteps 50000 --save policy/dqn.zip
# a named method: its own observation, action set, reward and policy, all in
# robotsnap/models/<method>.py - including the `template` a new one is copied from
python -m robotsnap train --method cadrl --episodes 2000 --save policy/cadrl.pt
python -m robotsnap train --method sarl  --episodes 2000 --save policy/sarl.pt
python -m robotsnap train --method ga3c_cadrl --episodes 2000 --save policy/ga3c_cadrl.pt
python -m robotsnap train --method rgl   --episodes 2000 --save policy/rgl.pt
# nothing named: the run writes policy/template-<stamp>.pt and says so
python -m robotsnap train --method template --episodes 200

# replay, whichever of the three wrote the file: the checkpoint says so itself
python -m robotsnap play --load policy/ppo.zip --episodes 3
python -m robotsnap play --load policy/cadrl.pt --episodes 3
```

`--observations` chooses what the policy sees, so the same algorithm is trained
on `pose,goal`, `pose,goal,lidar` or `pose,goal,agents` without touching code
(`python -m robotsnap train --help` lists the components); `examples/my_env.py`
shows the four-line subclass that changes the reward instead.

CADRL, SARL, GA3C-CADRL and RGL are the methods of the social-navigation
literature this task is modelled on: a discrete set of eighty `(speed, heading)`
commands, a
hand-written cost that pays progress and charges for collisions and for entering
a pedestrian's personal space, and a value network over the robot, its goal and
a varying number of neighbours. CADRL pools the neighbours with an LSTM, SARL
with multi-head self-attention, RGL with a relational graph; the task, the
action set and the reward are shared, so a comparison between them is a
comparison of the encoders. GA3C-CADRL shares the crowd encoder with CADRL but
learns with actor-critic returns instead of a value network and a lookahead, so
it is the one method here that is *not* a "deep V-learning" one.

Each method is one file under `robotsnap/models/`, holding *all* of its parts -
its environment, its reward constants, its policy and its agent - so what a
method is can be read in one screen. What the methods share (the command grid,
the observation layout, the social cost, the task, the state-value base, the
lookahead learner and the DQN) is a single definition in
`robotsnap/models/social.py`, not a copy per
method, and nothing under `robotsnap/` outside that package names a method at
all. `robotsnap/models/template.py` is a complete worked example to copy: a
reduced observation, three commands, a two-term reward and a policy of its own,
with the four axes a method can change marked where they are changed.

### Hyperparameters come from a file

The values a run starts from live under `configs/`, one file per learning rule
and one per method, and `--config` reads one of them by name or by path:

```bash
python -m robotsnap train --config ppo                 # configs/algorithms/ppo.yaml
python -m robotsnap train --config cadrl --episodes 5000
python -m robotsnap train --config my_study.yaml --gamma 0.5   # the line still wins
```

Precedence is defaults in the code, then the file, then what was typed: a file
supplies what the command line left out and never overrides it. A key the
sub-command does not define is refused with both names in the message, so a typo
cannot half-configure a run. `configs/README.md` lists the two levels and how a
name resolves; adding your own file is copying the closest one.

### Curriculum learning

`--curriculum` gives a training run a ladder of scenarios, and moves up it either
after a fixed number of episodes or as soon as the success rate over a trailing
window clears a threshold - whichever comes first:

```bash
python -m robotsnap train --algo ppo --timesteps 300000 \
    --curriculum social_navigation --save policy/ppo_social.zip
```

`configs/curriculum/social_navigation.yaml` is the worked ladder: an empty map,
then one person, a doorway, a crossing, and a crowd. Each stage names the
scenario it runs and its own promotion rule, and a stage may carry `options` -
so a curriculum is also where a scenario's `count_range`, speed range or
departure window is randomised per episode. The run prints the stage it is at
each time it moves, and a stage with neither an episode budget nor a threshold
simply stays where it is, which is how a fixed final stage is written.

## Benchmark a suite

`benchmark` plays a policy over a suite of scenarios, N episodes each, averages
the metrics Unity reports and writes the campaign to a file - and `--compare`
reads two of those files back into one table:

```bash
# the six basic navigation scenarios, five episodes each, no checkpoint needed
python -m robotsnap benchmark --suite basic --episodes 5 --out results/goal.json
# the same suite against a trained policy, and against two at once
python -m robotsnap benchmark --suite basic --load ppo_social.zip --out results/ppo.json
python -m robotsnap benchmark --load cadrl.pt --out results/cadrl.json
python -m robotsnap benchmark --compare results/goal.json results/ppo.json
```

Without `--load` the robot is driven by this package's go-to-goal controller
(`--policy scripted`) or by random actions (`--policy random`): a benchmark needs
floors to compare against. A checkpoint names its own reader - Stable-Baselines3,
the small policy of this package, or a named method, which brings its own
observation and action space - so `--load` is enough and `--algo`/`--method` only
override what the file already says. The campaign keeps every episode per
scenario as well as the average over the whole run, because a policy that fails
one scenario and wins the others is not the same as one that is mediocre
everywhere.

## A Gymnasium environment: one episode is one navigation problem

`robotsnap.envs` is an extra rather than a requirement - `pip install -e
".[env]"` brings `gymnasium`, and nothing in the bridge, the client or the
viewer imports it.

```python
from robotsnap.envs import RobotSNAPEnv

env = RobotSNAPEnv(scenario="python_demo", scenario_fields=dict(map_name="basic/crowd", ...))
observation, info = env.reset(seed=0)
while True:
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

`reset()` makes sure the scenario runs - launching the configured one when the
session has none, re-applying the current one otherwise, which is the cheap
reset that gives every episode the same start - puts the robot under ROS
control, and anchors the episode clock. `step()` writes `/cmd_vel`, holds for
`control_period` seconds, and reads the world back. The reward pays progress
toward the goal and arrival, and charges for time, a crash and leaving the map.

### Inherit from it

Everything the simulator does lives in the base class; everything the *task*
does is a method to replace:

| method | what it decides |
| --- | --- |
| `observation(world, task)` | what the agent sees |
| `robot_features`, `goal_features`, `agent_features`, `lidar_features` | the parts of the default vector |
| `reward(previous, current, action, world)` | what it is paid |
| `action_to_command(action)` | what an action means on the wire |
| `terminated(task)`, `truncated(task)` | when the episode ends |
| `task(world)` | the quantities all of the above read |
| `make_observation_space()`, `make_action_space()` | the spaces those two have to match |

The observation is usually chosen rather than overridden - see
[the components below](#the-observation) - so a subclass that only wants
different *features* registers a component or declares one on itself, and a
subclass that wants a different *task* replaces `reward`, `terminated` or
`task` alone.

Each of them receives the `World` of that step: poses, the laser, the agents,
the occupancy grid - no JSON, no sockets. A custom environment is therefore a
few lines:

```python
class SocialNavEnv(RobotSNAPEnv):
    def reward(self, previous, current, action, world):
        value = super().reward(previous, current, action, world)
        if current.nearest_agent_distance is not None and current.nearest_agent_distance < 1.2:
            value -= 2.0
        return value
```

`examples/my_env.py` is a full one - a polar observation, a social reward, and a
go-to-goal controller run against it, which is the other half of the point:
benchmarking a method you already have, not only training one.
`examples/env_random_episode.py` is the smallest complete loop, and
`examples/env_train_reinforce.py` plugs a PyTorch policy into the same two calls.

### The observation

An observation is a list of **named components**, and the default one is the
four the environment has always stacked: `robot`, `goal`, `agents` and `lidar`.
Ask for another set and you get exactly those, in the order you give them - the
space is derived from the same components, so it always matches what comes out:

```python
RobotSNAPEnv(observations="pose,goal")                       # no lidar, no crowd
RobotSNAPEnv(observations=("robot", "goal", "agents"))       # no lidar
RobotSNAPEnv(observations="pose,goal,lidar_stats")           # no agents
RobotSNAPEnv(observations="none")                            # reward only
RobotSNAPEnv(observations={"lidar": {"bins": 24}, "pose": {}})
RobotSNAPEnv(observations="pose,goal", observation_structure="dict")
```

| component | size | what it is |
| --- | --- | --- |
| `robot` | 6 | pose and body velocity: x, y, cos yaw, sin yaw, v, w |
| `pose` | 4 | pose alone |
| `velocity` | 2 | body velocity alone |
| `goal` | 5 | the goal in the robot frame: dx, dy, distance, cos bearing, sin bearing |
| `goal_distance` | 1 | the distance alone, or -1 when the session publishes no goal |
| `agents` | 5 × `max` | the neighbours the robot's detector reports, nearest first, with a visibility flag |
| `nearest_agent` | 5 | the closest of them alone: distance, cos bearing, sin bearing, speed, present |
| `humans` | 5 × `max` | the simulated humans of the scenario, ground truth rather than perception |
| `lidar` | `bins` | the sweep resampled, normalised to `[0, 1]` |
| `lidar_stats` | 3 | closest range, mean range, fraction of free beams |
| `occupancy` | `size`² | a world-aligned patch of `/map` around the robot, rows along +y |
| `status` | 3 | goal reached, collision, out of bounds |
| `time` | 1 | the fraction of the episode budget spent |

A component's own parameters - `max` for `agents` and `humans`, `bins` for
`lidar`, `size` for `occupancy`, `threshold` for the wall value - size the
environment before the component is built, so nothing can declare one shape and
produce another. Whatever you ask for, `info["world"]` still carries the whole
read, so a callback or a logger is never blind to what the observation leaves
out. A collision is the cell under the robot holding an obstacle, read from
`/map`; without a grid the lidar is the fallback.

A project with a feature of its own registers it once and asks for it by name:

```python
from robotsnap.envs import ObservationPart, register_observation_part

register_observation_part("battery", ObservationPart(
    name="battery", size=1,
    low=np.zeros(1, dtype=np.float32), high=np.ones(1, dtype=np.float32),
    build=lambda context: np.array([context.world.linear_velocity / 10.0]),
))
RobotSNAPEnv(observations=("pose", "goal", "battery"))
```

A task that does not want to register anything globally declares its components
on itself (`RobotSNAPEnv.extra_observation_parts`), names them in its own
default spec, and a caller can still override the whole list - which is what
`examples/my_env.py` does.

### Three things to know before a long run

This environment is **real-time coupled**. Unity has no "advance exactly N fixed
steps" primitive, so a step holds until the world has moved for `control_period`
seconds of simulated time, which is `control_period / time_scale` seconds of wall
time. `max_episode_seconds` counts that simulated time, so `set_time_scale(5)`
makes an episode five times shorter in wall time while each step still covers the
same amount of the world.

The **scene has to be in Play mode**: the environment listens on the bridge port
and waits for Unity to dial in, so start the script first and press Play after,
as with the other examples.

**One session per environment.** Each environment owns a bridge port, so a
vector of them needs one Unity instance each - the topic prefix of the contract
(`EnvROS.Prefix`) is the hook for that, and it is not exercised yet. Two
environments on one port is a bind error, which is the port doing its job.

**What a higher scale costs.** The pacing above is a wall-clock wait for a slice
of simulated time, so what the interpreter spends waking up and reading the
streams comes out of the slice. The environment measures that cost and takes it
off the next wait, so one step holds the slice it was asked for: measured on the
crowd scenario, 0.2000 s of simulated time at x1, 0.1995 s at x5 and 0.2045 s at
x10. What does not shrink with the wait is what the *session* can deliver: at x20
a step lasts 17 ms of wall time because that is how often Unity can hand over a
coherent state and odometry pair, it advances 0.35 s of simulated time, and
`info["fresh"]` is false most of the time. Training between x5 and x10 and
evaluating at x1 is the combination that holds.

## Run the tests

```bash
python -m pytest
```

The suite talks to a fake Unity peer over a local socket, so it needs neither
Unity nor ROS.

## Command reference

```bash
# install, once, into the virtualenv the repository already carries
venv_robotSNAP/bin/pip install -e .
# or into an environment you use already
pip install -e .
# and, for the Gymnasium environment, its extra (gymnasium, a few megabytes)
pip install -e ".[env]"
# and, for the training example, PyTorch as well
pip install -e ".[env,train]"

# a session without ROS: this listens first, then press Play in Unity
python -m robotsnap.bridge --port 10000
python -m robotsnap.bridge --host 0.0.0.0 --port 10000   # another machine dials in
# the same bridge, with the session mirrored on a ROS2 graph for a ROS2 method
python -m robotsnap bridge --ros2 --port 10000

# hyperparameters from a file, and a ladder of scenarios to train on
python -m robotsnap train --config ppo --timesteps 200000 --save policy/ppo.zip
python -m robotsnap train --algo ppo --timesteps 300000 \
    --curriculum social_navigation --save policy/ppo_social.zip

# a suite of scenarios, N episodes each, and the comparison of two campaigns
python -m robotsnap benchmark --suite basic --episodes 5 --out results/goal.json
python -m robotsnap benchmark --load ppo_social.zip --out results/ppo.json
python -m robotsnap benchmark --compare results/goal.json results/ppo.json
# what `bench` answers instead: simulated seconds against wall seconds
python -m robotsnap bench --steps 100 --time-scale 10

# watch a session nobody drives: `watch` starts a bridge of its own and draws it
python -m robotsnap watch --port 10000 --rate 30
# ...or watch while a command drives, which is the same window as an option
python -m robotsnap train --viewer --algo ppo --timesteps 20000
python -m robotsnap bridge --ros2 --viewer      # the bridge draws what it serves

# the scenario round trip: write it, launch it, drive it, freeze it
python examples/scenario_control.py --port 10000
python examples/scenario_control.py --scenario demo --seconds 10 --keep
python examples/scenario_control.py --unity-project /path/to/Project --no-drive

# the Gymnasium environment: one random episode, one custom task, one training loop
python examples/env_random_episode.py --port 10000
python examples/env_random_episode.py --port 10000 --time-scale 5   # same episode, five times faster in wall time
python examples/my_env.py --episodes 3 --render          # scripted go-to-goal
python examples/my_env.py --episodes 3 --random          # the baseline it beats
python examples/env_train_reinforce.py --episodes 20     # needs torch
python examples/env_train_reinforce.py --episodes 20 --save policy.pt
# an interpreter that already has torch works too, with the source tree on the path
PYTHONPATH=src /usr/bin/python3 examples/env_train_reinforce.py --episodes 20
# where a scenario is written, and what the project already holds
python -c "from robotsnap import scenario; print(scenario.scenarios_dir()); print(scenario.list_names())"
# another Unity project than the checkout beside the home
export ROBOTSNAP_UNITY_PROJECT=/path/to/Project

# one command at a time, against a bridge already listening somewhere else
python -c "from robotsnap.client import RobotSNAPClient; c=RobotSNAPClient(port=10000); print(c.snapshot())"

# a ROS2 graph instead of the Python bridge: Unity dials this the same way
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=10000
# ...and the same watcher, reading that graph rather than owning the socket
python -m robotsnap watch --ros2
# ...and the same training or inference run, riding the graph instead of the socket
python -m robotsnap train --transport ros2 --algo ppo --timesteps 20000 --keep
python -m robotsnap play  --transport ros2 --load policy.zip --time-scale 1
# (or skip ros_tcp_endpoint entirely: `robotsnap bridge --ros2` serves Unity and
# mirrors the session on the graph in one process)
# the whole wiring - Unity + viewer + a navigation method in a container - is in
# the sibling repo: robotsnap-ros2/docs/ros2-with-a-docker-method.md

# the tests: no Unity, no ROS, a fake peer over a local socket
python -m pytest
python -m pytest tests/robotsnap/bridge        # wire protocol, codecs, state model
python -m pytest tests/robotsnap/client        # the command facade
python -m pytest tests/robotsnap/envs          # the world model and the environment
python -m pytest tests/robotsnap/test_scenario.py tests/robotsnap/viewer
python -m pytest -k scenario -v
```

## Layout

- `src/robotsnap/topics.py` - the ten topic names and their types, the one place
  the Python side names a stream.
- `src/robotsnap/bridge/` - how a session is reached. `server.py` is the TCP
  endpoint, `protocol.py` and `codec.py` the wire format, `state.py` the decoded
  snapshot, and `transport.py` the one door to an optional transport: the core
  knows the entry-point group `robotsnap.transports`, not ROS2. The `ros2`
  transport itself - the mirror that puts the session on a graph beside the
  socket, and the client that reads a graph instead of a socket - lives in the
  sibling repository `robotsnap-ros2`; its `Ros2Client.bridge` answers the
  bridge's whole surface, so the client, the environment and the viewer take
  either transport unchanged.
- `src/robotsnap/cli/` - `python -m robotsnap <command>`: one entry point for
  every run, the bridge included, with `--transport {tcp,ros2}` where a session
  is reached. One module per command under `cli/commands/` - read
  `commands/episode.py` first, it is the shortest complete one - the shared
  options in `cli/options.py`, and the list of commands in
  `cli/commands/__init__.py`.
- `src/robotsnap/client.py` - the script-facing facade: commands, state reads,
  crowd control.
- `src/robotsnap/config/` - reading a run's hyperparameters from a file and
  turning them into the defaults of a sub-command. `configs/` holds the files
  themselves, one per learning rule, per method, per curriculum and per suite.
- `src/robotsnap/scenario.py` - the YAML of a scenario: authoring it, writing it
  into the Unity project, listing and removing it.
- `src/robotsnap/analysis/` - reading back what a session produced:
  `metrics.py` knows the shape of an episode document and turns a list of them
  into the averages a campaign compares, `suites.py` names the scenarios a
  benchmark runs, and `campaign.py` runs the suite, saves it and compares two.
- `src/robotsnap/rl/` - the learning plumbing, method-agnostic: `policy.py` is
  the PyTorch policy and the checkpoint it travels in, `sb3.py` is the
  Stable-Baselines3 adapter, `curriculum.py` is the ladder of scenarios and
  `curriculum_env.py` the wrapper that applies it to each reset. None of them
  imports torch, numpy or gymnasium until it is used.
- `src/robotsnap/runs/` - what each command actually does, one module per
  responsibility: `session.py` opens and closes a session, `presets.py` holds
  the default scenarios, `episodes.py`, `training_*.py`, `inference_*.py`,
  `scenario_control.py` hold the runs themselves, `benchmark.py` times a session
  where `campaign.py` scores one, and `spec.py` says in one line what a run
  resolved to. Read `runs/episodes.py` first.
- `src/robotsnap/envs/` - the Gymnasium environment: `world.py` reads a session
  into poses, a scan, agents and a grid; `base.py` is the environment to inherit
  from, and the only place that launches, paces and reads.
- `src/robotsnap/viewer/` - the pygame view: `window.py` draws, `session.py`
  turns a client's messages into what is drawn, `geometry.py` converts frames,
  `colours.py` holds the palette, `run.py` is the `watch` command behind
  `--viewer`.
- `examples/scenario_control.py` - a scenario written, launched and stopped from
  a script, end to end.
- `examples/env_random_episode.py`, `examples/my_env.py` and
  `examples/env_train_reinforce.py` - the environment used three ways: a random
  episode, a task of your own with a scripted controller, and a training loop.
- `examples/check_mission_time.py` - one episode measured against the clock
  Unity displays, for when an episode must not overrun its budget.
- `examples/benchmark_suite.py` - the same suite, campaign and comparison as
  `python -m robotsnap benchmark`, written as the four library calls a script of
  your own would make.
- `docs/robotsnap-unity-contract.md` - the Unity-side reference: the ten topics,
  the JSON bodies and the commands.
- `legacy/` - the human trajectory prediction research code (datasets, trained
  models, Trajectron++), kept for reference and not part of the package.
