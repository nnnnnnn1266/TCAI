# -*- coding: utf-8 -*-
"""評估多個模型回答品質的腳本。

使用方式：
    python 所有指標.py --input turtle1QA.xlsx --output model_scores.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, recall_score

# 可能出現在資料集裡的標準答案與問題欄位名稱
TRUTH_COL_CANDIDATES = ["Truth Answer", "標準答案", "answer", "Answer"]
QUESTION_COL_CANDIDATES = ["Question", "問題", "question"]


def parse_args() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="計算多模型問答評估指標")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("turtle1QA.xlsx"),
        help="輸入資料檔（xlsx/csv），需包含標準答案與模型輸出欄位",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_scores.xlsx"),
        help="輸出彙整結果檔（xlsx）",
    )
    parser.add_argument(
        "--sbert-model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer 模型名稱",
    )
    return parser.parse_args()


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    """用候選名稱清單，從資料欄位中找出實際欄名（忽略大小寫）。"""
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lowered:
            return lowered[key]
    return None


def load_dataframe(file_path: Path) -> pd.DataFrame:
    """依副檔名讀取 xlsx/csv，並移除常見 Unnamed 欄位。"""
    if not file_path.exists():
        raise FileNotFoundError(f"找不到輸入檔案：{file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    return df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")].copy()


def simple_acc(truth: str, pred: str) -> int:
    """完全比對（字串去除前後空白後）是否一致。"""
    return int(str(truth).strip() == str(pred).strip())


def bleu_score(truth: str, pred: str) -> float:
    """以字元為單位計算 BLEU 分數（較適合中文）。"""
    smoothie = SmoothingFunction().method4
    truth_tokens = list(str(truth))
    pred_tokens = list(str(pred))
    return sentence_bleu([truth_tokens], pred_tokens, smoothing_function=smoothie)


def rouge_l_score(truth: str, pred: str) -> float:
    """計算 ROUGE-L F1。"""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(str(truth), str(pred))["rougeL"].fmeasure


def char_level_metrics(truth_list: pd.Series, pred_list: pd.Series) -> tuple[float, float]:
    """以字元集合的二元向量方式，計算平均 F1/Recall。"""
    y_true_all, y_pred_all = [], []
    for t, p in zip(truth_list, pred_list):
        y_true = list(str(t).strip())
        y_pred = list(str(p).strip())

        # 將「出現過的字元」轉成 0/1 向量，做多標籤比對
        all_tokens = set(y_true) | set(y_pred)
        y_true_bin = [1 if tk in y_true else 0 for tk in all_tokens]
        y_pred_bin = [1 if tk in y_pred else 0 for tk in all_tokens]
        y_true_all.append(y_true_bin)
        y_pred_all.append(y_pred_bin)

    f1_scores = [f1_score(y_true, y_pred, zero_division=0) for y_true, y_pred in zip(y_true_all, y_pred_all)]
    recall_scores = [
        recall_score(y_true, y_pred, zero_division=0) for y_true, y_pred in zip(y_true_all, y_pred_all)
    ]
    return float(np.mean(f1_scores)), float(np.mean(recall_scores))


def semantic_similarity(
    truth_list: pd.Series,
    pred_list: pd.Series,
    sbert_model: SentenceTransformer,
) -> float:
    """使用 SBERT 向量計算平均 cosine similarity。"""
    truth_embeds = sbert_model.encode(list(map(str, truth_list)), convert_to_tensor=True, show_progress_bar=False)
    pred_embeds = sbert_model.encode(list(map(str, pred_list)), convert_to_tensor=True, show_progress_bar=False)
    sim_scores = (truth_embeds * pred_embeds).sum(axis=1) / (truth_embeds.norm(dim=1) * pred_embeds.norm(dim=1))
    return float(sim_scores.cpu().numpy().mean())


def bertscore_metric(truth_list: pd.Series, pred_list: pd.Series) -> float:
    """以中文語系設定計算 BERTScore F1。"""
    _, _, f1 = bert_score(
        [str(p) for p in pred_list],
        [str(t) for t in truth_list],
        lang="zh",
        rescale_with_baseline=True,
    )
    return float(f1.mean().item())


def compute_scores(
    truth_list: pd.Series,
    pred_list: pd.Series,
    sbert_model: SentenceTransformer,
) -> dict[str, float]:
    """對單一模型輸出欄位計算完整評估指標。"""
    exact_match_scores, bleu_scores, rouge_scores = [], [], []
    f1, recall = char_level_metrics(truth_list, pred_list)

    # 逐筆計算可在單句評估的指標
    for t, p in zip(truth_list, pred_list):
        exact_match_scores.append(simple_acc(t, p))
        bleu_scores.append(bleu_score(t, p))
        rouge_scores.append(rouge_l_score(t, p))

    # 需要整批資料向量化的指標
    sem_sim = semantic_similarity(truth_list, pred_list, sbert_model)
    bert_score_val = bertscore_metric(truth_list, pred_list)

    return {
        "Accuracy": float(np.mean(exact_match_scores)),
        "BLEU": float(np.mean(bleu_scores)),
        "ROUGE-L": float(np.mean(rouge_scores)),
        "F1": f1,
        "Recall": recall,
        "SemSim": sem_sim,
        "BERTScore": bert_score_val,
    }


def main() -> None:
    """主流程：讀檔、欄位辨識、計算各模型分數並輸出結果。"""
    args = parse_args()
    df = load_dataframe(args.input)

    columns = [str(c) for c in df.columns]
    truth_col = detect_column(columns, TRUTH_COL_CANDIDATES)
    question_col = detect_column(columns, QUESTION_COL_CANDIDATES)

    if truth_col is None:
        raise ValueError(f"找不到標準答案欄位，請確認欄位名稱包含：{TRUTH_COL_CANDIDATES}")

    # 問題欄與標準答案欄不列入模型評估欄位
    ignored_cols = {truth_col}
    if question_col is not None:
        ignored_cols.add(question_col)

    model_cols = [col for col in columns if col not in ignored_cols]
    if not model_cols:
        raise ValueError("找不到模型輸出欄位，請確認資料格式是否正確")

    # 載入語意模型（僅初始化一次，供多欄位共用）
    sbert_model = SentenceTransformer(args.sbert_model)

    results = []
    for model_col in model_cols:
        scores = compute_scores(df[truth_col], df[model_col], sbert_model)
        scores["Model"] = model_col
        results.append(scores)

    results_df = pd.DataFrame(results)
    results_df = results_df[["Model", "Accuracy", "BLEU", "ROUGE-L", "F1", "Recall", "SemSim", "BERTScore"]]
    results_df = results_df.round(4)

    sorted_df = results_df.sort_values("ROUGE-L", ascending=False)
    print("模型分數比較：")
    print(sorted_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sorted_df.to_excel(args.output, index=False)
    print(f"\n已輸出結果：{args.output}")


if __name__ == "__main__":
    main()
