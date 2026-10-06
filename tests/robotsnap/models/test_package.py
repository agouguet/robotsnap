"""The catalogue's surface, and the two directions the split has to hold in.

Three things are worth a test here and each is a subprocess, because both of
them are statements about what an *import* does rather than about what a call
returns: the catalogue loads without a framework, the core modules load without
the catalogue at all, and the path the methods used to live at still resolves.
"""

import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

from robotsnap.models import ALGORITHM_CLASSES, ALGORITHMS, METHODS, METHOD_LEARNER, MODELS
from robotsnap.models.cadrl import CadrlAgent, CadrlEnv
from robotsnap.models.ga3c_cadrl import Ga3cAgent, Ga3cCadrlEnv
from robotsnap.models.rgl import RglAgent, RglEnv
from robotsnap.models.sarl import SarlAgent, SarlEnv
from robotsnap.models.template import TemplateAgent, TemplateEnv

#: A program that makes ``torch`` fail and then imports the catalogue. If any
#: method reached for the framework while its module was being imported, this is
#: where it would show up, as an ImportError rather than as a slow import.
_BLOCKED_TORCH = """
import sys


class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not available in this interpreter")
        return None


sys.meta_path.insert(0, _BlockTorch())

from robotsnap.models import ALGORITHMS, METHODS
import robotsnap.models.template as template

assert "torch" not in sys.modules, "reading the catalogue pulled torch in"
assert set(ALGORITHMS) == {"cadrl", "sarl", "ga3c_cadrl", "rgl", "template"}
assert set(ALGORITHMS) == set(METHODS)
assert template.TEMPLATE_OBSERVATION_SIZE > 0
"""

#: A program that makes the whole catalogue unimportable and then imports the
#: core. A core module that reached for ``robotsnap.models`` - even lazily, at
#: module import - would fail here instead of quietly coupling the two halves.
_BLOCKED_MODELS = """
import sys


class _BlockModels:
    def find_spec(self, name, path=None, target=None):
        if name == "robotsnap.models" or name.startswith("robotsnap.models."):
            raise ImportError("the core must not import the models")
        return None


sys.meta_path.insert(0, _BlockModels())

import robotsnap.cli
import robotsnap.rl.policy
import robotsnap.runs
import robotsnap.rl.sb3
import robotsnap.scenario
import robotsnap.runs.spec
import robotsnap.topics
import robotsnap.envs.base

assert not any(name.startswith("robotsnap.models") for name in sys.modules)
assert "torch" not in sys.modules, "the core pulled torch in"
"""


def _run(source: str) -> subprocess.CompletedProcess:
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=environment,
    )


def test_the_registry_pairs_each_method_with_its_classes():
    assert set(ALGORITHMS) == {"cadrl", "sarl", "ga3c_cadrl", "rgl", "template"}
    assert ALGORITHMS["cadrl"] == (CadrlEnv, CadrlAgent)
    assert ALGORITHMS["sarl"] == (SarlEnv, SarlAgent)
    assert ALGORITHMS["ga3c_cadrl"] == (Ga3cCadrlEnv, Ga3cAgent)
    assert ALGORITHMS["rgl"] == (RglEnv, RglAgent)
    assert ALGORITHMS["template"] == (TemplateEnv, TemplateAgent)
    assert ALGORITHM_CLASSES == {
        "cadrl": CadrlAgent,
        "sarl": SarlAgent,
        "ga3c_cadrl": Ga3cAgent,
        "rgl": RglAgent,
        "template": TemplateAgent,
    }


def test_the_registry_and_the_descriptions_are_the_same_names():
    """The table a command line prints and the classes a run builds cannot drift."""
    assert set(MODELS) == set(METHODS) == set(METHOD_LEARNER) == set(ALGORITHMS)
    for name, description in METHODS.items():
        assert description.strip(), f"{name} has no description to show in --help"
        assert METHOD_LEARNER[name]


def test_importing_the_catalogue_does_not_need_torch():
    result = _run(_BLOCKED_TORCH)
    assert result.returncode == 0, result.stderr


def test_the_core_does_not_import_the_models():
    result = _run(_BLOCKED_MODELS)
    assert result.returncode == 0, result.stderr


def test_the_old_social_package_still_reexports_under_a_warning():
    import robotsnap.social as shim

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim = importlib.reload(shim)

    assert any(issubclass(entry.category, DeprecationWarning) for entry in caught), (
        "the compatibility shim has to say that it is one"
    )
    assert shim.ALGORITHMS is ALGORITHMS
    assert shim.CadrlEnv is CadrlEnv
    assert shim.SarlAgent is SarlAgent


def test_the_old_submodule_paths_still_resolve():
    """The submodules the package was split into are gone; their names are not."""
    from robotsnap.social.actions import SocialActionSpace
    from robotsnap.social.agents import CadrlAgent as OldCadrlAgent
    from robotsnap.social.env import SocialNavEnv
    from robotsnap.social.models import EGO_SIZE
    from robotsnap.social.reward import SocialReward
    from robotsnap.social.sarl import SarlEnv as OldSarlEnv

    assert OldCadrlAgent is CadrlAgent
    assert OldSarlEnv is SarlEnv
    assert EGO_SIZE > 0
    assert SocialActionSpace.__module__ == "robotsnap.models.social"
    assert SocialReward.__module__ == "robotsnap.models.social"
    assert SocialNavEnv.__module__ == "robotsnap.models.social"
