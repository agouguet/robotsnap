"""The scenario document the demo commands write."""

from __future__ import annotations

from robotsnap import scenario


def scenario_document(name: str = "python_demo") -> dict:
    """One robot crossing a small crowd, on the crowd map of the basic dataset.

    The points are in the Unity world axes the scenario editor writes, which is
    what this file is authored in; the poses read back from the session are in
    the ROS frame the streams publish.
    """
    return scenario.build(
        name,
        map_name="basic/crowd",
        description="A robot crossing a crowd, written from Python",
        tags=["Python", "Demo"],
        robots=[
            scenario.robot(
                "robot_1",
                "jackal",
                (-7.0, 0.0, 0.0),
                (7.0, 0.0),
                speed=1.0,
            ),
        ],
        humans=[
            scenario.crowd(
                "pedestrians",
                5,
                spawn=scenario.spawn_zone(0.0, -5.0, 3.0, 3.0),
                goal=scenario.spawn_zone(0.0, 5.0, 3.0, 3.0),
                speed=1.0,
                end_behavior="loop",
            ),
        ],
    )
