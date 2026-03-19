import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "turtle_llama3_1_(8b).py"


spec = importlib.util.spec_from_file_location("training_cli", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_detect_column_matches_dataset_aliases():
    columns = ["Question", "Complex_CoT", "Response"]
    assert module.detect_column(columns, module.QUESTION_ALIASES) == "Question"
    assert module.detect_column(columns, module.ANSWER_ALIASES) == "Response"
    assert module.detect_column(columns, module.REASONING_ALIASES) == "Complex_CoT"


def test_build_prompt_uses_traditional_chinese_template():
    prompt = module.build_prompt("烏龜吃什麼？", "以常見飼養情境回答", "可吃葉菜與配方飼料。", "<eos>")
    assert "你是 TCAI" in prompt
    assert "### 問題：\n烏龜吃什麼？" in prompt
    assert prompt.endswith("<eos>")


def test_resolve_columns_accepts_explicit_overrides():
    class DummyArgs:
        question_column = "題目"
        answer_column = "答案"
        reasoning_column = "補充"

    class DummyFrame:
        columns = ["題目", "補充", "答案"]

    mapping = module.resolve_columns(DummyFrame(), DummyArgs())
    assert mapping.question == "題目"
    assert mapping.answer == "答案"
    assert mapping.reasoning == "補充"
