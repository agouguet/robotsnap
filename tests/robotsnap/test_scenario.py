"""Tests for ``robotsnap.scenario``: what it writes, and where."""

import pytest

from robotsnap import scenario


def _document(name: str = "demo"):
    return scenario.build(
        name,
        map_name="basic/crowd",
        description="Two robots and a crowd",
        tags=["Python", "Demo"],
        robots=[
            scenario.robot("robot_1", "jackal", (2.0, -6.0, 90), (20.0, -6.0), speed=1.5),
            scenario.robot("robot_2", "kuri", "start_2", "goal_2"),
        ],
        points={
            "start_2": scenario.point(-6.0, 3.0),
            "goal_2": scenario.zone(0.0, 3.0, 2.0, 2.0),
        },
        humans=[
            scenario.crowd(
                "pedestrians",
                6,
                spawn=scenario.spawn_zone(11.0, 0.0, 6.0, 6.0),
                goal=scenario.goal_point(11.0, -8.0, radius=1.0),
                end_behavior="loop",
            )
        ],
    )


# -- the document -----------------------------------------------------------=


def test_a_loose_start_and_goal_become_named_points():
    document = _document()

    robot = document["robots"][0]
    assert robot["start"] == "robot_1_start"
    assert robot["goal"] == "robot_1_goal"
    assert document["points"]["robot_1_start"] == {
        "x": 2.0,
        "y": 0.0,
        "z": -6.0,
        "yaw": 90.0,
        "center": None,
        "size": None,
    }
    assert document["points"]["robot_1_goal"]["z"] == -6.0


def test_a_named_point_is_left_alone():
    document = _document()

    assert document["robots"][1]["start"] == "start_2"
    assert document["points"]["start_2"]["x"] == -6.0


def test_a_caller_point_is_not_overwritten_by_a_generated_one():
    document = scenario.build(
        "demo",
        map_name="basic",
        robots=[scenario.robot("robot_1", "jackal", (1.0, 1.0), (2.0, 2.0))],
        points={"robot_1_start": scenario.point(9.0, 9.0)},
    )

    assert document["points"]["robot_1_start"]["x"] == 9.0


def test_the_metadata_names_the_scenario_and_its_map():
    info = _document("my run / but not a path")["scenario_info"]

    assert info["name"] == "my run / but not a path"
    assert info["map"] == "basic/crowd"
    assert info["version"] == "1.0"
    assert info["tags"] == ["Python", "Demo"]
    # The label is a display string, the type of the robot list is what spawns.
    assert info["robot_type"] == "jackal"


def test_a_scenario_needs_a_robot_with_a_start_and_a_goal():
    with pytest.raises(ValueError):
        scenario.build("empty", map_name="basic", robots=[])
    with pytest.raises(ValueError):
        scenario.build(
            "no start",
            map_name="basic",
            robots=[scenario.robot("robot_1", "jackal", "", (2.0, 2.0))],
        )
    with pytest.raises(TypeError):
        scenario.build(
            "a list is not a point",
            map_name="basic",
            robots=[scenario.robot("robot_1", "jackal", [1, 2, 3, 4], (2.0, 2.0))],
        )


def test_a_waypoint_is_named_and_kept_in_order():
    document = scenario.build(
        "route",
        map_name="basic",
        robots=[
            scenario.robot(
                "robot_1",
                "jackal",
                (0.0, 0.0),
                (4.0, 0.0),
                waypoints=[(1.0, 1.0), "known", (3.0, -1.0)],
            )
        ],
        points={"known": scenario.point(2.0, 2.0)},
    )

    assert document["robots"][0]["waypoints"] == [
        "robot_1_waypoint_0",
        "known",
        "robot_1_waypoint_2",
    ]
    assert document["points"]["robot_1_waypoint_2"]["x"] == 3.0


# -- the YAML ---------------------------------------------------------------


def test_the_yaml_reads_back_as_the_document():
    """The emitter is checked against a parser, not against itself."""
    yaml = pytest.importorskip("yaml")

    document = _document()
    assert yaml.safe_load(scenario.to_yaml(document)) == document


def test_a_string_that_looks_like_another_type_stays_a_string():
    yaml = pytest.importorskip("yaml")

    document = scenario.build(
        "demo",
        map_name="basic",
        robots=[scenario.robot("robot_1", "jackal", (0.0, 0.0), (1.0, 1.0))],
        tags=["yes", "off", "1.0", "null", ""],
    )
    parsed = yaml.safe_load(scenario.to_yaml(document))

    assert parsed["scenario_info"]["tags"] == ["yes", "off", "1.0", "null", ""]


def test_a_text_with_a_colon_or_a_quote_is_quoted():
    yaml = pytest.importorskip("yaml")

    document = scenario.build(
        "demo",
        map_name="basic",
        robots=[scenario.robot("robot_1", "jackal", (0.0, 0.0), (1.0, 1.0))],
        description="it's a run: with punctuation",
    )
    parsed = yaml.safe_load(scenario.to_yaml(document))

    assert parsed["scenario_info"]["description"] == "it's a run: with punctuation"


# -- the file ---------------------------------------------------------------


def test_write_names_the_file_after_the_scenario(tmp_path):
    path = scenario.write(_document("python demo"), directory=tmp_path)

    assert path == tmp_path / "python demo.yaml"
    assert "scenario_info:" in path.read_text(encoding="utf-8")
    assert scenario.list_names(tmp_path) == ["python demo"]


def test_write_takes_a_name_and_refuses_a_path(tmp_path):
    path = scenario.write(_document(), name="other name", directory=tmp_path)
    assert path.name == "other name.yaml"

    with pytest.raises(ValueError):
        scenario.write(_document(), name="../escape", directory=tmp_path)


def test_write_can_refuse_to_overwrite(tmp_path):
    scenario.write(_document(), name="demo", directory=tmp_path)

    with pytest.raises(FileExistsError):
        scenario.write(_document(), name="demo", directory=tmp_path, overwrite=False)


def test_delete_removes_the_file_and_says_so(tmp_path):
    scenario.write(_document(), name="demo", directory=tmp_path)

    assert scenario.delete("demo", directory=tmp_path) is True
    assert scenario.delete("demo", directory=tmp_path) is False
    assert scenario.list_names(tmp_path) == []


def test_find_names_the_file_an_id_is_stored_in(tmp_path):
    """An id and a file name are two spellings of one thing."""
    scenario.write(_document(), name="demo", directory=tmp_path)

    found = scenario.find("demo", tmp_path)
    assert found == tmp_path / "demo.yaml"
    assert found.is_file()
    assert scenario.find("demo.yaml", tmp_path) == found
    assert scenario.find("absent", tmp_path) is None


def test_list_names_reads_both_extensions_and_sorts(tmp_path):
    (tmp_path / "b.yml").write_text("", encoding="utf-8")
    (tmp_path / "a.yaml").write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")

    assert scenario.list_names(tmp_path) == ["a", "b"]
    assert scenario.list_names(tmp_path / "missing") == []


# -- ranges -----------------------------------------------------------------


def _ranged_document():
    """A scenario that lets the run draw the four values a study randomizes."""
    return scenario.build(
        "ranged",
        map_name="basic/crowd",
        robots=[
            scenario.robot(
                "robot_1",
                "jackal",
                (2.0, -6.0, 90),
                (20.0, -6.0),
                speed=1.5,
                speed_range=(1.0, 2.0),
                start_yaw_range=scenario.value_range(-30.0, 30.0),
            )
        ],
        humans=[
            scenario.crowd(
                "pedestrians",
                3,
                spawn=scenario.spawn_zone(11.0, 0.0, 6.0, 6.0, spacing=1.0, spacing_range=(0.6, 1.4)),
                goal=scenario.goal_point(11.0, -8.0, radius=1.0),
                speed=1.1,
                speed_range=(0.8, 1.6),
                count_range=(2, 6),
                spawn_window=2.0,
                spawn_window_range=(0.0, 6.0),
            )
        ],
    )


def test_a_range_is_written_beside_the_value_it_replaces():
    document = _ranged_document()

    robot = document["robots"][0]
    assert robot["speed"] == 1.5  # the fixed value stays as the nominal one
    assert robot["speed_range"] == {"min": 1.0, "max": 2.0}
    assert robot["start_yaw_range"] == {"min": -30.0, "max": 30.0}

    human = document["humans"][0]
    assert human["count_range"] == {"min": 2.0, "max": 6.0}
    assert human["speed_range"] == {"min": 0.8, "max": 1.6}
    assert human["spawn_window_range"] == {"min": 0.0, "max": 6.0}
    assert human["spawn"]["spacing_range"] == {"min": 0.6, "max": 1.4}


def test_the_yaml_of_a_range_reads_back_as_the_document():
    """The range keys are checked against a real parser, like the rest of the emitter."""
    yaml = pytest.importorskip("yaml")

    document = _ranged_document()
    assert yaml.safe_load(scenario.to_yaml(document)) == document


def test_a_scenario_that_randomizes_nothing_carries_no_range():
    """The old shape is the absence of every range, so an old file is what a fixed one writes."""
    document = _document()

    assert "speed_range" not in document["robots"][0]
    assert "start_yaw_range" not in document["robots"][0]
    human = document["humans"][0]
    assert "count_range" not in human
    assert "speed_range" not in human
    assert "spawn_window_range" not in human
    assert "spacing_range" not in human["spawn"]


def test_value_range_orders_its_bounds():
    assert scenario.value_range(6, 2) == {"min": 2.0, "max": 6.0}


def test_read_range_reads_every_shape_the_vocabulary_can_carry():
    assert scenario.read_range(None) is None
    assert scenario.read_range(3) == (3.0, 3.0)
    assert scenario.read_range((0.8, 1.6)) == (0.8, 1.6)
    assert scenario.read_range({"min": 2, "max": 6}) == (2.0, 6.0)
    assert scenario.read_range({"max": 6, "min": 2}) == (2.0, 6.0)


# -- where the project is ---------------------------------------------------


def test_the_project_is_read_from_the_environment(monkeypatch, tmp_path):
    (tmp_path / "Assets" / "StreamingAssets").mkdir(parents=True)
    monkeypatch.setenv(scenario.UNITY_PROJECT_ENV, str(tmp_path))

    assert scenario.unity_project() == tmp_path.resolve()
    assert scenario.scenarios_dir() == tmp_path.resolve() / "Assets" / "StreamingAssets" / "Scenarios"


def test_an_explicit_project_wins_over_the_environment(monkeypatch, tmp_path):
    other = tmp_path / "other"
    (other / "Assets" / "StreamingAssets").mkdir(parents=True)
    monkeypatch.setenv(scenario.UNITY_PROJECT_ENV, str(tmp_path))

    assert scenario.unity_project(other) == other.resolve()


def test_an_installed_applications_data_folder_is_a_project(tmp_path):
    """A packaged player holds ``StreamingAssets`` itself, without ``Assets/``.

    That data folder is what an installed application reads its scenarios and
    maps from, so it is the folder a run has to be pointed at when the
    application ships as a build rather than as the project.
    """
    data = tmp_path / "robotsnap-unity_Data"
    (data / "StreamingAssets" / "Scenarios").mkdir(parents=True)

    assert scenario.unity_project(data) == data.resolve()
    assert scenario.scenarios_dir(data) == data.resolve() / "StreamingAssets" / "Scenarios"


def test_a_directory_that_is_not_a_project_is_refused(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.delenv(scenario.UNITY_PROJECT_ENV, raising=False)
    monkeypatch.setattr(scenario, "_FALLBACK_PROJECTS", (tmp_path / "absent",))
    monkeypatch.chdir(empty)

    with pytest.raises(FileNotFoundError):
        scenario.unity_project()
