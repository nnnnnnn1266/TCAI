import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "所有指標.py"


def load_metrics_module():
    """Load 所有指標.py with lightweight stubs for optional heavy deps."""
    # numpy stub
    np_mod = types.ModuleType("numpy")
    np_mod.mean = lambda arr: sum(arr) / len(arr) if arr else 0.0
    sys.modules["numpy"] = np_mod

    # pandas stub (only needed for import/typing in these tests)
    pd_mod = types.ModuleType("pandas")
    pd_mod.Series = list
    pd_mod.DataFrame = dict
    sys.modules["pandas"] = pd_mod

    # bert_score stub
    bert_score_mod = types.ModuleType("bert_score")
    bert_score_mod.score = lambda *args, **kwargs: (None, None, types.SimpleNamespace(mean=lambda: types.SimpleNamespace(item=lambda: 0.0)))
    sys.modules["bert_score"] = bert_score_mod

    # nltk BLEU stubs
    nltk_bleu_mod = types.ModuleType("nltk.translate.bleu_score")

    class DummySmooth:
        method4 = None

    nltk_bleu_mod.SmoothingFunction = lambda: DummySmooth()
    nltk_bleu_mod.sentence_bleu = lambda *args, **kwargs: 0.0
    sys.modules["nltk"] = types.ModuleType("nltk")
    sys.modules["nltk.translate"] = types.ModuleType("nltk.translate")
    sys.modules["nltk.translate.bleu_score"] = nltk_bleu_mod

    # rouge stubs
    rouge_mod = types.ModuleType("rouge_score")
    rouge_scorer_mod = types.ModuleType("rouge_score.rouge_scorer")

    class DummyRougeScorer:
        def __init__(self, *args, **kwargs):
            pass

        def score(self, *args, **kwargs):
            return {"rougeL": types.SimpleNamespace(fmeasure=0.0)}

    rouge_scorer_mod.RougeScorer = DummyRougeScorer
    rouge_mod.rouge_scorer = rouge_scorer_mod
    sys.modules["rouge_score"] = rouge_mod
    sys.modules["rouge_score.rouge_scorer"] = rouge_scorer_mod

    # sentence-transformers stub
    sent_mod = types.ModuleType("sentence_transformers")
    sent_mod.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sent_mod

    # sklearn stubs
    sklearn_mod = types.ModuleType("sklearn")
    sklearn_metrics_mod = types.ModuleType("sklearn.metrics")
    sklearn_metrics_mod.f1_score = lambda *args, **kwargs: 1.0
    sklearn_metrics_mod.recall_score = lambda *args, **kwargs: 1.0
    sklearn_mod.metrics = sklearn_metrics_mod
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.metrics"] = sklearn_metrics_mod

    spec = importlib.util.spec_from_file_location("metrics_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detect_column_case_insensitive():
    module = load_metrics_module()
    columns = ["question", "truth answer", "model_a"]
    assert module.detect_column(columns, ["Truth Answer"]) == "truth answer"


def test_simple_acc_trims_spaces():
    module = load_metrics_module()
    assert module.simple_acc(" turtle ", "turtle") == 1
    assert module.simple_acc("turtle", "rabbit") == 0


def test_char_level_metrics_returns_valid_range():
    module = load_metrics_module()
    f1, recall = module.char_level_metrics(["abc"], ["abc"])
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= recall <= 1.0
