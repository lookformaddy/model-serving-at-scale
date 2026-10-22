import pytest
from model_serving.registry import build_demo_registry
from model_serving.service import ModelServingService

@pytest.fixture()
def registry():
    return build_demo_registry()

@pytest.fixture()
def service(registry):
    return ModelServingService(registry)
