"""Every shipped configuration must load and apply to the ``train`` parser.

The list of files is read from the folders at collection time, so dropping a
new configuration in ``configs/algorithms/`` or ``configs/methods/`` adds it to
this test rather than hiding from it.
"""

import pytest

from robotsnap import config
from robotsnap.cli.main import build_parser

#: The two trees a run names on the command line.
_TREES = ("algorithms", "methods")


def _shipped_configurations():
    files = []
    for tree in _TREES:
        directory = config.CONFIG_DIRECTORY / tree
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.yaml")))
    return files


@pytest.mark.parametrize(
    "path", _shipped_configurations(), ids=lambda path: f"{path.parent.name}/{path.name}"
)
def test_every_shipped_configuration_loads_and_applies(path):
    document = config.load(path)
    assert document, "a shipped configuration cannot be empty"
    if path.parent.name == "methods":
        assert "method" in document
    else:
        assert "algo" in document

    parser = build_parser()._robotsnap_commands["train"]
    applied = config.apply_to(parser, "train", document)
    assert applied == sorted(document)

    namespace = parser.parse_args([])
    for key in applied:
        assert hasattr(namespace, key)
