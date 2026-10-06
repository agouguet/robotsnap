"""Tests for the two state-value networks, one per method module of the catalogue.

CADRL and SARL train ``V(s)``, not ``Q(s, a)``: each network turns a state - the
ego vector, the neighbour rows and their mask - into one scalar, and the command
is the lookahead's business. The tests below therefore check the state-value
contract the two encoders share: one value per state, padding that cannot change
it, and a mask honoured wherever it sits.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robotsnap.models.cadrl import cadrl_class, cadrl_value  # noqa: E402
from robotsnap.models.sarl import sarl_class, sarl_value  # noqa: E402
from robotsnap.models.social import (  # noqa: E402 - after the torch skip
    EGO_SIZE,
    GOAL_SIZE,
    NEIGHBOUR_SIZE,
)

FACTORIES = [cadrl_value, sarl_value]


def _state(batch: int = 3, count: int = 3, seed: int = 0):
    """A batch of states and its neighbour block."""
    rng = np.random.default_rng(seed)
    ego = rng.normal(size=(batch, EGO_SIZE + GOAL_SIZE)).astype(np.float32)
    neighbours = rng.normal(size=(batch, count, NEIGHBOUR_SIZE)).astype(np.float32)
    mask = np.ones((batch, count), dtype=bool)
    return ego, neighbours, mask


def _model(factory, ego_features: int):
    return factory(
        ego_features=ego_features,
        neighbour_features=NEIGHBOUR_SIZE,
        hidden=8,
    )


@pytest.mark.parametrize("factory", FACTORIES)
def test_the_value_is_one_scalar_per_state(factory):
    ego, neighbours, mask = _state(batch=2, count=4)
    model = _model(factory, ego.shape[1])
    values = model(ego, neighbours, mask)
    assert tuple(values.shape) == (2,)
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("factory", FACTORIES)
def test_padding_a_state_to_a_larger_neighbourhood_changes_nothing(factory):
    ego, neighbours, mask = _state(batch=3, count=3, seed=1)
    model = _model(factory, ego.shape[1])
    small = model(ego, neighbours, mask)

    absurd = np.full((3, 5, NEIGHBOUR_SIZE), 1e6, dtype=np.float32)
    padded = np.concatenate([neighbours, absurd], axis=1)
    padded_mask = np.concatenate([mask, np.zeros((3, 5), dtype=bool)], axis=1)
    large = model(ego, padded, padded_mask)

    assert torch.allclose(small, large, atol=1e-6)


@pytest.mark.parametrize("factory", FACTORIES)
def test_a_padded_neighbour_is_ignored_whatever_it_holds(factory):
    ego, neighbours, mask = _state(batch=2, count=4, seed=2)
    mask = mask.copy()
    mask[:, 2:] = False
    model = _model(factory, ego.shape[1])
    quiet = model(ego, neighbours, mask)

    changed = neighbours.copy()
    changed[:, 2:] = -12345.0
    noisy = model(ego, changed, mask)

    assert torch.allclose(quiet, noisy, atol=1e-6)


@pytest.mark.parametrize("factory", FACTORIES)
def test_a_state_with_no_neighbour_ignores_the_padded_rows(factory):
    ego, neighbours, _ = _state(batch=2, count=4, seed=3)
    model = _model(factory, ego.shape[1])
    empty = np.zeros((2, 4), dtype=bool)
    values = model(ego, neighbours, empty)
    other = model(ego, np.full_like(neighbours, -7.0), empty)
    assert torch.isfinite(values).all()
    assert torch.allclose(values, other, atol=1e-6)


@pytest.mark.parametrize("factory", FACTORIES)
def test_a_mask_that_is_not_packed_to_the_front_is_respected(factory):
    ego, partial, mask = _state(batch=3, count=3, seed=4)
    model = _model(factory, ego.shape[1])
    small = model(ego, partial, mask)

    trailing = np.concatenate([np.ones_like(partial), partial], axis=1)
    trailing_mask = np.concatenate([np.zeros((3, 3), dtype=bool), mask], axis=1)
    padded_front = model(ego, trailing, trailing_mask)

    assert torch.allclose(small, padded_front, atol=1e-6)


@pytest.mark.parametrize("factory", FACTORIES)
def test_a_float_mask_is_accepted(factory):
    ego, neighbours, mask = _state(batch=2, count=3, seed=6)
    model = _model(factory, ego.shape[1])
    as_float = mask.astype(np.float32)
    assert torch.allclose(model(ego, neighbours, mask), model(ego, neighbours, as_float))


@pytest.mark.parametrize("factory", FACTORIES)
def test_the_network_scores_states_and_holds_no_action_table(factory):
    """The whole point of the rewrite: no candidate actions enter the network."""
    ego, neighbours, mask = _state(batch=2, count=3, seed=7)
    model = _model(factory, ego.shape[1])
    assert not hasattr(model, "action_table")
    values = model(ego, neighbours, mask)
    # The same state scored twice is the same value, and no third argument - no
    # action block - is needed for it.
    assert torch.allclose(values, model(ego, neighbours, mask))


def test_the_two_models_are_lazily_built_and_are_different():
    assert cadrl_class() is not sarl_class()
    assert isinstance(cadrl_value(ego_features=4, neighbour_features=2), torch.nn.Module)
    assert isinstance(sarl_value(ego_features=4, neighbour_features=2), torch.nn.Module)
