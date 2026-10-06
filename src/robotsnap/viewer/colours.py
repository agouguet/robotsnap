"""The viewer's palette and its tunable constants.

Every number a reader may want to change lives here and nowhere else: the
window size, the redraw rate, the marker sizes, and one RGB triple per thing
the window draws. The frame names and the lidar-placement budgets are here too,
because they are tuning values rather than logic.
"""

#: Window size, in pixels.
WINDOW_SIZE = (1000, 800)
#: Redraws per second when ``--rate`` is not given.
DEFAULT_RATE = 30.0
#: Metres visible across the window while no map has arrived to fit.
DEFAULT_VIEW_SPAN = 24.0
#: Fraction of the window left free around a fitted map.
MAP_MARGIN = 0.04
#: Radius of the robot marker, in pixels.
ROBOT_RADIUS = 9
#: Height of the status band, in pixels.
HUD_HEIGHT = 62
#: Frame names the agent stream can announce.
ROBOT_FRAME = "robot"
WORLD_FRAME = "world"

#: How many stamped robot poses are kept to place a scan at the instant it was measured. Ten seconds of a
#: ten hertz odometry stream, which is far more than the longest gap between two messages.
POSE_HISTORY = 100

#: How long a scan waits for the odometry sample that comes after its stamp, in seconds of wall time. One
#: odometry period is the natural bound: the sample that closes the interval arrives within one of them, and
#: the floor is what keeps a stream that stopped publishing odometry from holing the backlog for ever.
SCAN_HOLD_SECONDS = 0.6

#: Scans kept while they wait for the pose of their own instant. Two speeds of stream are never more than a
#: handful of scans apart, so this is a runaway guard rather than a working size.
PENDING_SCANS = 32

#: Palette, RGB.
BACKGROUND = (24, 26, 31)
GRID = (44, 48, 56)
HUD_BACKGROUND = (16, 18, 23)
MAP_FREE = (238, 240, 243)
MAP_WALL = (40, 43, 50)
MAP_UNKNOWN = (150, 154, 162)
ROBOT_BODY = (0, 160, 255)
ROBOT_OUTLINE = (255, 255, 255)
#: Body colours for the non-primary robots, the same weight as :data:`ROBOT_BODY`.
ROBOT_PALETTE = (
    (0, 200, 160),
    (170, 120, 255),
    (255, 170, 60),
    (120, 200, 90),
    (235, 100, 140),
    (110, 160, 255),
)
LASER = (0, 230, 118)
AGENT_VISIBLE = (255, 64, 208)
AGENT_HIDDEN = (122, 44, 112)
AGENT_OUTLINE = (255, 255, 255)
GOAL = (255, 72, 72)
TEXT = (238, 240, 243)
WAITING = (255, 190, 80)
