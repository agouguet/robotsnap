# RobotSNAP <-> Python contract

Reference for the Python-side bridge. Everything below is read from the Unity
project at `/home/adam/robotsnap-workspace/robotsnap-unity` (branch `main`).
Component names point at the Unity sources.

## Topology

Unity runs `com.unity.robotics.ros-tcp-connector` and opens a TCP connection as
a *client*: Unity is the peer that dials, and whatever it dials is the server.
That peer can be either

- `ros_tcp_endpoint`, when a ROS2 install is available and the session should
  sit on the ROS graph, or
- `robotsnap.bridge`, the pure-Python endpoint of this package, which speaks the
  same wire protocol and needs no ROS at all.

Unity cannot tell the two apart, so one scene runs in both setups. The Python
bridge listens on port 10000 by default, so it must be listening before the
Unity scene starts; a reconnect means a new handshake and a fresh set of
registrations.

The project compiles with `ROS2` defined
(`ProjectSettings/ProjectSettings.asset`:
`scriptingDefineSymbols: Standalone: ROS2;...`), which changes the string
encoding on the wire. See below.

### The surface: ten topics

Ten topics replaced nineteen. The rule is one topic per kind of information, and
the standard ROS message type wherever one exists, so a client needs no custom
`.msg` - which is exactly what makes the "Unity + Python without ROS" case free
of dependencies.

| Topic | Type | Direction | Meaning |
| --- | --- | --- | --- |
| `/clock` | `rosgraph_msgs/Clock` | out | simulation clock |
| `/odom` | `nav_msgs/Odometry` | out | robot pose and twist |
| `/scan` | `sensor_msgs/LaserScan` | out | lidar |
| `/map` | `nav_msgs/OccupancyGrid` | out | occupancy grid, walls 100 / free 0, image order |
| `/cmd_vel` | `geometry_msgs/Twist` | in | robot velocity command |
| `/simulation/state` | `std_msgs/String`, JSON | out | the session snapshot |
| `/simulation/agents` | `std_msgs/String`, JSON | out | every agent in the robot frame |
| `/simulation/control` | `std_msgs/String`, JSON | in | every command |
| `/simulation/control_result` | `std_msgs/String`, JSON | out | the answer |
| `/reset_done` | `std_msgs/Bool` | out | the world is ready |

The names live in exactly one place on each side, and the two lists are meant to
be read side by side:

- `Assets/Scripts/RobotSNAP/ROS/RobotSNAPTopics.cs` - the constants and the
  `/prefix/topic` join every component builds its name with. No component writes
  a topic of its own any more.
- `src/robotsnap/topics.py` - the same ten names and their types, which the
  bridge, the client and the viewer import instead of carrying strings.

Renaming a stream is one edit per side, and `test_topics.py` fails if a topic
literal appears anywhere else in the Python package.

One caveat about `/clock`: the component exists and is enabled in code, but the
`ROSClock` object of `Assets/Scenes/default.unity` carries `m_Enabled: 0`, so a
run of that scene publishes nine of the ten topics. The clock is still readable
from `sim_time_seconds` in `/simulation/state`.

Gone with this revision: the `simulation_msgs/*` types on the wire, `/agents`,
`/agents/global`, `/agents/pose`, `/human_pose`, `/robot_pose`, `/robot_odom`,
`/global_path`, `/local_goal_from_robot`, `/local_goal_from_map`,
`/simulation/humans/control`, and the `/unity/reset` and `/unity/play`
services. Nothing a client needs was lost: the goal travels in the state
snapshot, and the crowd commands travel in the `humans` command of
`/simulation/control`.

## The four setups

There is one Unity transport, not four. Unity is a plain TCP client, so the four
ways of using RobotSNAP are decided by **what listens on the other end**, not by
anything compiled into the scene:

| Setup | Peer | What you need |
| --- | --- | --- |
| Unity + Python | `python -m robotsnap.bridge` | nothing but this package |
| Unity + ROS2 | `ros_tcp_endpoint` | a ROS2 install |
| Unity + Python + Gymnasium / PyTorch | `robotsnap.client` (owns the bridge) | this package, plus `gymnasium`/`torch` for your own loop |
| Unity + ROS2 + Python + Gymnasium | `ros_tcp_endpoint`, with your Python node on the ROS graph | both |

The first and the third rows need **no ROS at all**: the bridge speaks the same
wire protocol as `ros_tcp_endpoint`, so Unity cannot tell the difference. The
only dependency of the Python side is `rosbags`, which is a pure-Python package
used to encode and decode CDR - it does not pull in `rclpy` or a ROS
distribution.

Two consequences worth knowing:

- The topic prefix (`EnvROS.Prefix`, empty by default) is the only thing that
  changes between a single session and several. Everything else is identical.
- `ros_tcp_endpoint` renames nothing, so a ROS2 user gets `/odom`, `/scan`,
  `/cmd_vel`, `/map` and `/simulation/*` unchanged, and a Python user gets the
  same graph on a socket instead of on DDS.

## Wire protocol

Little-endian unsigned int32 for lengths. Strings are UTF-8, length-prefixed.

```
frame: [int32 len(dest)][dest][int32 size][size bytes payload]
```

The same layout is used in both directions. A system command is simply a frame
whose destination is the command name and whose payload is that command's JSON
body. There is no extra nesting.

In ROS2 mode `MessageSerializer.Write(string)` emits
`[int32 len+1][utf8 bytes][0x00]`, so **declared string lengths include a
trailing NUL byte**. Readers trust the declared length, so a producer that omits
the NUL is still understood; a reader must strip trailing NULs, which
`protocol.read_string` does.

Unity's reader builds the destination with
`Encoding.ASCII.GetString(scratch, 0, topicLength)` and does not strip the NUL.
For topic names that is harmless: dispatch uses `topic.StartsWith("__")` and a
topic lookup. For the handshake it is not: `ReaderThread` compares the
destination to `"__handshake"` with exact string equality, so a handshake reply
must **not** carry a trailing NUL in the destination.

System commands (destination starts with `__`):

| command | payload |
| --- | --- |
| `__handshake` | `{"version": str, "metadata": str}` |
| `__publish` | `{"topic": str, "message_name": str}` |
| `__subscribe` | `{"topic": str, "message_name": str}` |
| `__topic_list` | request for the topic inventory |
| `__response` / `__request` | service plumbing, carries `srv_id` |
| `__log` / `__warn` / `__error` | text for the Unity console |

### Session opening

The Python endpoint sends `__handshake` as the **first** bytes on the socket,
before reading anything. `ROSConnection.ReaderThread` reads exactly one message
and requires it to be the handshake, otherwise it logs
`Invalid ROS-TCP-Endpoint version detected: 0.6.0 or older`.

The payload is `{"version": str, "metadata": str}` where `metadata` is itself a
JSON string. Unity applies two checks (`ROSConnection.cs`):

- `version.StartsWith("v0.7.")`, with `k_Version = "v0.7.0"` and
  `k_CompatibleVersionPrefix = "v0.7."`
- `metadata.protocol == "ROS2"`, because the project is compiled with `ROS2`
  defined. `ROS1` raises
  `Incompatible protocol: ROS-TCP-Endpoint is using ROS1, but Unity is in ROS2 mode`.

A command written before Unity has read and accepted that handshake is consumed
as the handshake and lost, and a command written before the components of the
scene have subscribed goes to a topic nobody listens to yet: the connector drops
it. Both windows are why `RobotSNAPClient.wait_until_ready` waits for the
`__publish` and `__subscribe` registrations rather than for the socket, and why
it reads `RobotSNAPBridge.announced_topics()` - what the peer said - instead of
`topic_types()`, which the bridge seeds before anything connects and which would
make every freshly opened socket look ready.

## Streams published by Unity

| topic | type | producer |
| --- | --- | --- |
| `/clock` | `rosgraph_msgs/Clock` | `ROSClockPublisher.cs` |
| `/odom` | `nav_msgs/Odometry` | `OdometryPublisher.cs` |
| `/scan` | `sensor_msgs/LaserScan` | `LaserScanPublisher.cs` |
| `/map` | `nav_msgs/OccupancyGrid` | `SimulationStatePublisher.cs` |
| `/simulation/state` | `std_msgs/String` (JSON) | `SimulationStatePublisher.cs` |
| `/simulation/agents` | `std_msgs/String` (JSON) | `AgentDetectorROS.cs` |
| `/simulation/control_result` | `std_msgs/String` (JSON) | `SimulationControlBridge.cs` |
| `/reset_done` | `std_msgs/Bool` | `SimulationStatePublisher.cs` |

The state snapshot, the laser and the agent detection publish at 10 Hz by
default (`_publishInterval`, `publishFrequencyHz = 10f`) and are driven by
`Update`, so the real rate is bounded by the frame rate. The grid changes only
when a scenario is applied, so it is sampled once, published at most once per
second (`mapIntervalSeconds`, floored at one second), and `/reset_done` tells a
client the world is ready again after a reset or a scenario load.

### The occupancy grid on `/map`

`nav_msgs/OccupancyGrid`: `data` carries `100` for a wall, `0` for free space
and `-1` for unknown; `info.resolution` is metres per cell and `info.origin` the
pose of cell `(0, 0)` in the ROS frame. The grid is published in the ROS order -
`index = row * width + column`, a column stepping along +x and a row along +y,
`info.origin` being the (min x, min y) corner - so a client draws it with +x to
the right and +y upwards without reordering anything.

### The state snapshot on `/simulation/state`

One JSON object per snapshot, with the keys `simulation_state` (`idle`, `ready`,
`running`, `paused`), `playing`, `paused`, `stopped`, `scenario_applied`,
`sim_time_seconds`, `time_scale`, `scenario_id`, `scenario_name`, `environment`,
`map_name`, `map_width`, `map_height`, `map_resolution`, `map_origin_x`,
`map_origin_y`, `human_count`, `humans`, `robot`, `robot_has_goal`,
`robot_goal`, `camera_focus`, `camera_focus_is_robot`, `camera_focus_is_human`,
`camera_focus_agent_id`, `camera_tool`, `camera_mode`, `camera_pose` and
`refreshed_at`.

A `humans` entry is:

```json
{"id": 3, "x": 1.0, "y": 2.0, "z": 0.0, "vx": 0.1, "vy": 0.0, "vz": 0.0,
 "speed": 1.2, "goal": {"x": 5.0, "y": 2.0}, "group": "G1",
 "controller": "sfm", "end_behavior": "loop"}
```

`goal` and `group` may be `null`, `controller` is `sfm`, `external`, `manual`
or `replay`, and `end_behavior` is what the agent does when its route runs out:
`stay`, `disappear` or `loop`. `manual` is a pedestrian driven by the keyboard
of the machine running the simulator: an operator takes the pedestrian the
camera follows with the grab key, and its velocity then travels the same
channel as `external` - so the two are one contract with a different driver,
and a reader can tell them apart. `replay` is set from the Analysis tab, which
re-applies a past episode's scenario and walks its humans along the recorded
motion instead of the social forces; there is no command for it, because the
tracks travel with the episode rather than on the wire.

Poses are in the ROS frame (x forward, y left, z up), yaw in radians; the
**world-axis** `x`/`z` of the scenario editor is not what these publish.
`std_msgs/String` has no header, so the snapshot carries no simulator timestamp
either: `sim_time_seconds` and `refreshed_at` are the clock and the wall-clock
time of the publisher.

`sim_time_seconds` is `Clock.Instance.CurrentTimeSeconds`, and that clock
belongs to the **session**, not to the scenario. `Clock` is `[ExecuteAlways]` and
sets its origin when the component is enabled: in the Editor that is the moment
the scene was loaded, unless entering Play mode reloads the domain, in which case
it is the moment Play started. Either way it is before a scenario is chosen, and
`load_scenario` does not put it back to zero. Only `reset` does, and even there
the answer comes back before the world is rebuilt. Measured on a live session
entered without a domain reload: a scenario applied just after Play mode started
reported `sim_time_seconds` at 610 s while the scenario itself was 0 s old.

So a caller that measures an episode takes a mark of its own.
`RobotSNAPClient.scenario_time_seconds` is that mark subtracted from the clock,
anchored when a launch settles, and `mark_scenario_start()` is there for a
`reset`. In real-time mode the clock is wall-clock seconds since that origin; in
simulation mode it is the scaled clock, which `set_time_scale` speeds up or
slows down.

### The agent stream on `/simulation/agents`

One JSON object in a `std_msgs/String`, every active agent in the robot frame:

```json
{"agents": [{"id": "<id>", "x": 0.0, "y": 0.0, "z": 0.0,
             "vx": 0.0, "vy": 0.0, "vz": 0.0, "visible": true}],
 "frame": "robot"}
```

Two things separate this stream from the `humans` of the state snapshot, and
both matter to a client that draws:

- The poses are built with `transform.InverseTransformPoint`, so they are in the
  **robot frame**, not the world frame. Placing them on a map means applying the
  robot pose first.
- `visible` is the lidar-visibility flag of that agent, so a viewer can show the
  crowd the way the robot sensor sees it.

Like the state snapshot, this is a `std_msgs/String`: no header, so no simulator
timestamp, only the arrival time.

## Streams consumed by Unity

| topic | type | consumer |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/Twist` | `RobotInputController.cs`, `cmdVelTopic` |
| `/simulation/control` | `std_msgs/String` (JSON) | `SimulationControlBridge.cs` |

Details that matter for a control loop (`RobotInputController.cs`):

- The subscription is created only when `controlMode` is `ROS` or `Hybrid`. The
  mode is settable over the bridge (`set_control_mode`), but the scenarios still
  ship without one, so a fresh session has to ask for `ros` before `/cmd_vel` is
  read.
- `_targetAngular = -msg.angular.z`: the angular sign is flipped relative to
  ROS.
- Commands are clamped to `maxLinearSpeed` (2 m/s) and `maxAngularSpeed`
  (2 rad/s).
- In `Hybrid` mode the controller falls back to the keyboard after 0.5 s
  without a command (`rosCommandTimeout`), which would fight an intermittent
  Python client. Use `ROS` mode for control.
- `Update` stops the wheels while `Supervisor.IsPaused` and `FixedUpdate` skips
  applying velocities, so commands have no effect while paused.

## Inbound control surface

Unity is extended with a JSON control surface that runs over the same socket.
Both topics are `std_msgs/String` whose body is a JSON object, and message
bodies are CDR with the 4-byte ROS2 encapsulation header that
`MessageDeserializer.InitWithBuffer` skips.

| topic | direction | body |
| --- | --- | --- |
| `/simulation/control` | Python -> Unity | one command, session or crowd |
| `/simulation/control_result` | Unity -> Python | one acknowledgement per command |

Unity answers every command with a single message on
`/simulation/control_result`:

```json
{"command": "play", "ok": true, "message": "", "sim_time_seconds": 12.5}
```

`robotsnap.client.RobotSNAPClient` wraps this surface: one method per command,
each one sending its body to Unity and returning the parsed acknowledgement, or
`None` with `last_error` set when nothing arrives inside the timeout.

### Commands on `/simulation/control`

The body always carries a `command` key plus the keys that command needs.

| command | extra keys |
| --- | --- |
| `play` | none |
| `pause` | none |
| `toggle_pause` | none |
| `reset` | `scenario` (str, optional), `seed` (int, optional) |
| `load_scenario` | `scenario` (str), `start` (bool), `apply` (bool), `seed` (int, optional) |
| `set_time_scale` | `time_scale` (float) |
| `set_random_seed` | `seed` (int) |
| `set_robot_goal` | `x` (float), `z` (float), `y` (float, optional) |
| `clear_robot_goal` | none |
| `stop_robot` | none |
| `set_control_mode` | `mode`: `keyboard`, `ros`, `hybrid` or `scenario` |
| `set_agent_controller` | `mode`: `sfm`, `external` or `manual` |
| `humans` | `commands`: one entry per human, see below |

The optional keys are omitted from the body when Python has no value for them,
so Unity keeps its own default.

### The `humans` command

One message carries a whole batch, so a crowd moves in a single frame:

```json
{"command": "humans",
 "commands": [{"id": 3, "vx": 1.0, "vz": 0.0}, {"id": 4, "stop": true}]}
```

Each entry is either `{"id": int, "vx": float, "vz": float}` or
`{"id": int, "stop": true}`. The velocities are in metres per second along the
Unity world x and z axes: this is the one body of the contract whose numbers are
not in the ROS frame, because the crowd controller behind it works on Unity's
own axes.

The acknowledgement carries the usual keys plus `unknown_ids`, the ids the
simulator does not know. Note that the wire contract only carries an id per
entry: stopping *every* human means enumerating the ids of the last
`/simulation/state` snapshot, which is what `RobotSNAPClient.stop_humans()` does.

### Scenarios on disk

`load_scenario` names a scenario by the stem of its file under
`StreamingAssets/Scenarios`; the router resolves that name against the directory
listing case-insensitively, so `crowd` and `Crowd` reach the same file, and
`GetAvailableScenarios()` re-reads the directory on every call. A file written
there is therefore loadable without restarting the session, which is what makes
a scenario authorable from Python: `src/robotsnap/scenario.py` writes exactly
this file and `RobotSNAPClient.create_scenario` writes and loads it in one call.

The `map` key names a map under `StreamingAssets/Dataset`: `basic/crowd` is
`Dataset/basic/png/crowd.png` next to `Dataset/basic/json/crowd.json`, while a
bare collection name such as `basic` draws a random map of that collection. The
points of a scenario are in the **Unity world** axes the editor writes - x and z
on the ground plane, yaw in degrees - which is not the ROS frame the streams
publish.

Two limits are worth knowing:

- The loader caches a scenario under the name it was loaded with, so writing a
  different document over a name a session has used does not change what that
  name loads. Give a new file name, or write before the session reads it.
- `ScenarioData.IsValid` refuses a scenario with no robot, and one whose robot
  has no start or no goal; those are checked before a file is written rather
  than at load time.

### Running a session without ROS

No `rclpy`, no ROS2 graph and no launch file are involved: Python listens and
Unity dials in.

```bash
python -m robotsnap.bridge --host 0.0.0.0 --port 10000   # from /home/adam/robotSNAP_ws
```

Unity then connects to `127.0.0.1:10000`. In a script, the same session is:

```python
from robotsnap.client import RobotSNAPClient

with RobotSNAPClient(port=10000) as client:
    client.wait_until_ready()
    client.play()
    client.set_human_velocities([(3, 1.0, 0.0)])
```

Nothing has to be registered first: every body above is a `std_msgs/String`, so
no Python-side message type has to exist for the session to run.

## Services

There are none on this surface. `/unity/reset`, `/unity/play` and the
`simulation_msgs/Reset` and `simulation_msgs/PausePlay` types they needed are
gone: reset and play are `reset`/`load_scenario` and `play`/`pause` on
`/simulation/control`, run by the same router, so no second surface can diverge
from the command topic.

## Prefixing and multi-environment

`EnvROS` builds every topic name from `_prefix` (`BuildTopicName`, private:
`/{prefix}/{topic}`, or `/{topic}` when the prefix is empty), and the public
`UpdatePrefix` exists explicitly "for multi-environment". `SimulationConfig`
carries `_environmentCount` and `_environmentSpacing`, and `_inferenceMode` for
a single optimized environment. Parallel environments are anticipated on the
Unity side but there is no evidence they are exercised.

## Execution control

- `SimulationConfig.ApplyTimeSettings` is the only place that sets
  `Time.timeScale` and `Time.fixedDeltaTime` (defaults 1.0 and 0.02).
- `Clock` is a display clock: it tracks real or simulated wall-clock time for
  lighting, in `timeScale` and `_simulatedTimeSeconds`. It does not drive
  physics.
- Pausing (`Supervisor.Pause`, the `pause` command) does not freeze `Time`; it
  stops agents at the component level, each script checking
  `Supervisor.IsPaused` before applying motion.

Consequence: there is no deterministic "advance N fixed steps" primitive. A
Python step loop is real-time coupled unless a stepping service is added on the
Unity side.

## Open questions for the next lot

1. **Deterministic stepping versus real-time coupling**, and at what control
   rate. This is the one real gap left for a Gymnasium environment: pausing stops
   the agents at the component level and `set_time_scale` changes how fast the
   frame loop runs, but nothing advances the world by exactly N fixed steps and
   hands the frame back, so a `step()` is wall-clock coupled.
2. Which `/cmd_vel` consumer configuration ships with the scenarios. The mode can
   be set over the bridge (`set_control_mode`), but the scenarios still ship
   without one, so a fresh session has to ask for `ros` before `/cmd_vel` is
   read.
### Frames, as measured

Every pose Unity publishes - the robot of `/odom`, the `robot`, `humans`, `goal`,
`start_pose` and `target_pose` of `/simulation/state` - is in the ROS frame,
`x` = Unity `z`, `y` = the negated Unity `x`, `z` = Unity `y`, which is what
`To<FLU>()` produces. The scan of `/scan` follows: its bearings and its beam
order are mirrored, so beam `i` of the message sits at
`angle_min + i * angle_increment`. The occupancy grid of `/map` follows too:
`index = row * width + column`, a column counting along +x from `info.origin`
and a row along +y, `info.origin` being the pose of cell (0, 0).

The two mismatches this section used to list as open were measured against a
running scene rather than inferred: with the robot standing 0.8 m from a wall,
152 of its 167 finite lidar beams ended on a cell the map marks as a wall once
the bearings were mirrored, and 6 without, so the scan was mirrored; and the
grid was transposed, its columns counting along +y and its rows along +x.

Two more were measured the same way, and neither is fixed yet:

- **A positive angular command turns the robot clockwise** in the ROS frame the
  poses are published in. Holding ``/cmd_vel`` at ``angular.z = +0.5`` for two
  seconds took the published yaw of ``/odom`` from 0.00 to -0.63 rad. So a
  controller written against the published yaw negates its heading error.
- **The twist of ``/odom`` disagrees with the pose of ``/odom``**: for that same
  turn, ``twist.angular.z`` read +0.30 rad/s while the yaw in the same message
  was decreasing. A reader that needs a yaw rate consistent with the published
  poses has to difference those poses instead.

Resolved since the previous revision, kept here so the list is not silently
shortened: the control mode is settable over the bridge, the reset and play
services are gone with their `simulation_msgs` types, and the crowd commands are
folded into `/simulation/control`. The wire format, the handshake, the topic
names and the CDR bodies have all been observed on a live socket against Unity in
Play mode.

## Several robots

The application runs a roster of robots, and the two sides split the work
unevenly: the per-robot streams follow the roster, the session streams do not.

### What already follows the roster

`RobotIdentity` names every stream of one robot:

- every robot answers on `/robot_<id>/<topic>`, so `/robot_2/scan`;
- the primary robot (`robot_1`) *also* keeps the name the project has always
  published, so `/scan` reaches it and nothing older breaks.

`OdometryPublisher`, `LaserScanPublisher`, `RobotInputController` and
`AgentDetectorROS` all take their names from it, so `/odom`, `/scan`,
`/cmd_vel` and `/simulation/agents` are already per-robot. The agent snapshot is
taken in the frame of the robot that carries the detector, so
`/robot_2/simulation/agents` is what robot 2 can see.

Two consequences a client has to respect:

- the legacy name reaches the **primary** robot only; addressing any other robot
  means its `/robot_<id>/...` name;
- the frame of a pose is the one the naming implies. A pose read on
  `/robot_2/odom` is robot 2, not "the robot".

### What does not, and is what this section fixes

The session snapshot and the command router still resolve a single robot with
`FindAnyObjectByType<Robot>()`:

- `/simulation/state` carried one `robot`, plus `robot_has_goal`,
  `robot_goal`, `robot_start_pose` and `robot_target_pose`;
- `set_robot_goal`, `clear_robot_goal`, `stop_robot` and `set_control_mode` had
  no way to say *which* robot;
- the Python client mirrored that exactly, so `client.robot()`,
  `client.send_cmd_vel()` and the robot commands reached one arbitrary robot.

### The unified contract

`/simulation/state` gains a roster view and keeps its old keys:

| key | meaning |
| --- | --- |
| `robots` | ordered array of live robots, one entry each |
| `robot_count` | length of `robots` |
| `robot`, `robot_has_goal`, `robot_goal` | unchanged, the **primary** robot |

One `robots` entry, every pose in the ROS frame and yaw in radians:

```
{
  "id": "robot_2",          string, the id a command names
  "type": "jackal",         string, the RobotProfile id
  "is_primary": false,      bool, true for the robot legacy names reach
  "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0,
  "has_goal": false,        bool, true while it drives towards a goal
  "goal": null,             {x, y, z} or null
  "start_pose": null,       what the scenario authored {x, y, z, yaw}, or null
  "target_pose": null       what the scenario authored {x, y, z, yaw}, or null
}
```

`robots` is empty, not null, when the scene has no robot. The order is the
scenario's order, which is what `RobotRoster.FillRobots` walks.

`/simulation/control` gains an optional `"robot"` key on the four robot
commands, holding an id such as `"robot_2"`:

| command | keys |
| --- | --- |
| `set_robot_goal` | `x`, `y?`, `z`, `robot?` |
| `clear_robot_goal` | `robot?` |
| `stop_robot` | `robot?` |
| `set_control_mode` | `mode`, `robot?` |

Rules, which are what keeps one contract serving both a one-robot scene and a
fleet:

- `"robot"` absent, null or blank selects the roster's primary robot, which is
  exactly the robot those commands reached before, so an existing client is
  unaffected;
- an id the roster does not carry is refused, and the message names the ids it
  does carry;
- the acknowledgement echoes the id it acted on as `"robot"`, so a caller can
  confirm which robot answered without re-reading the snapshot.

### Naming a stream from an id, in Python

```
/robot_<id>/<topic>     always valid
/<topic>                the primary robot only
```

`robotsnap.topics.robot_topic(id, topic)` builds the first form, and
`robotsnap.client.RobotSNAPClient.send_cmd_vel(..., robot="robot_2")` routes
through it.
