"""The default world of each command.

One helper per run, so a command that is not pointed at a scenario still has a
world to write and the command line stays short.
"""

from __future__ import annotations

from typing import Any

from robotsnap import scenario


def random_episode_fields() -> dict[str, Any]:
    """The world of the random episode: one robot crossing a small crowd."""
    return dict(
        map_name="basic/crowd",
        description="Goal navigation, written from Python",
        robots=[
            scenario.robot("robot_1", "jackal", (-7.0, 0.0, 0.0), (7.0, 0.0), speed=1.5),
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


def social_fields() -> dict[str, Any]:
    """The world of the social task: a crowd the robot walks beside."""
    return dict(
        map_name="basic/crowd",
        description="Social goal navigation",
        robots=[
            scenario.robot("robot_1", "jackal", (-7.0, 2.0, 0.0), (7.0, 2.0), speed=1.5),
        ],
        humans=[
            scenario.crowd(
                "pedestrians",
                6,
                spawn=scenario.spawn_zone(0.0, -2.0, 4.0, 2.0),
                goal=scenario.spawn_zone(0.0, 4.0, 4.0, 2.0),
                speed=1.0,
                end_behavior="loop",
            ),
        ],
    )


def training_fields() -> dict[str, Any]:
    """The world of the training run: one robot and a small crowd."""
    return dict(
        map_name="basic/crowd",
        description="Goal navigation for a training loop",
        robots=[
            scenario.robot("robot_1", "jackal", (-7.0, 0.0, 0.0), (7.0, 0.0), speed=1.5),
        ],
        humans=[
            scenario.crowd(
                "pedestrians",
                4,
                spawn=scenario.spawn_zone(0.0, -5.0, 3.0, 3.0),
                goal=scenario.spawn_zone(0.0, 5.0, 3.0, 3.0),
                speed=1.0,
                end_behavior="loop",
            ),
        ],
    )


def bench_fields() -> dict[str, Any]:
    """The world of the timed run: the same crossing, timed instead of scored."""
    return dict(
        map_name="basic/crowd",
        description="Goal navigation, timed from Python",
        robots=[
            scenario.robot("robot_1", "jackal", (-7.0, 0.0, 0.0), (7.0, 0.0), speed=1.5),
        ],
        humans=[
            scenario.crowd(
                "pedestrians",
                4,
                spawn=scenario.spawn_zone(0.0, -5.0, 3.0, 3.0),
                goal=scenario.spawn_zone(0.0, 5.0, 3.0, 3.0),
                speed=1.0,
                end_behavior="loop",
            ),
        ],
    )
