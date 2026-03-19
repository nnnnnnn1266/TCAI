import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "temp.py"


spec = importlib.util.spec_from_file_location("temp_app", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)

sys.modules["pandas"] = types.ModuleType("pandas")
sys.modules["streamlit"] = types.ModuleType("streamlit")
sys.modules["ollama"] = types.ModuleType("ollama")
sys.modules["chromadb"] = types.SimpleNamespace(PersistentClient=object)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


class DummyModel:
    def __init__(self, name=None, model=None):
        self.name = name
        self.model = model


class DummyListResponse:
    def __init__(self, models):
        self.models = models


def test_extract_models_from_dict_response():
    resp = {"models": [{"name": "mxbai-embed-large"}, {"model": "llama3.1:latest"}]}
    names = module._extract_installed_model_names(module._extract_models_from_list_response(resp))
    assert "mxbai-embed-large" in names
    assert "llama3.1:latest" in names
    assert "llama3.1" in names


def test_extract_models_from_object_response():
    resp = DummyListResponse([DummyModel(name="mxbai-embed-large"), DummyModel(model="llama3.1:latest")])
    names = module._extract_installed_model_names(module._extract_models_from_list_response(resp))
    assert "mxbai-embed-large" in names
    assert "llama3.1" in names


def test_verify_ollama_ready_accepts_object_style_response():
    sys.modules["ollama"].list = lambda: DummyListResponse(
        [DummyModel(name="mxbai-embed-large"), DummyModel(model="llama3.1:latest")]
    )
    module.ollama = sys.modules["ollama"]
    module.verify_ollama_ready()
