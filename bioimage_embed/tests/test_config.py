from .. import config
import pytest
from hydra.utils import instantiate

schema_map = config.__schemas__
schemas = list(schema_map.values())


@pytest.mark.parametrize("Schema", schemas)
def test_schema(Schema):
    Schema()


@pytest.mark.parametrize("Schema", schemas)
def test_instantiate(Schema):
    schema = config.resolve_schema(Schema())
    obj = instantiate(schema)
    assert obj is not None, "obj should not be None"


def test_config(cfg):
    assert cfg is not None, "Config should not be None"


def test_config_instantiate(cfg):
    assert instantiate(cfg) is not None, "Config should not be None"


def test_resolve(bie):
    assert bie.resolve()


def test_model_check(bie):
    bie.model_check()


def test_train_check(bie):
    bie.trainer_check()


def test_bie_train(bie):
    bie.train()
