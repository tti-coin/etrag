import os

import torch

from .score_utils import run_semeval_evaluation

RUN_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl"
)
try:
    import datasets

    def evaluate_dataset(
        dataset: datasets.Dataset | list[dict[str, str]],
        labels: list[str],
        gold_tmp_file: str = ".tmp_gold",
        pred_tmp_file: str = ".tmp_pred",
        without_df: bool = False,
    ):
        gold_lines = []
        pred_lines = []
        for i, (dat, lbl) in enumerate(zip(dataset, labels)):
            pred_lines.append(f"{i}\t{lbl}")
            gold_lines.append(f"{i}\t{dat['label']}")
        with open(gold_tmp_file, "w") as f:
            f.write("\n".join(gold_lines) + "\n")
        with open(pred_tmp_file, "w") as f:
            f.write("\n".join(pred_lines) + "\n")
        return run_semeval_evaluation(RUN_FILE, pred_tmp_file, gold_tmp_file, without_df=without_df)

except ImportError:
    pass


def evaluate(
    golds: list[str],
    preds: list[str],
    pred_tmp_file: str = ".tmp_pred",
    gold_tmp_file: str = ".tmp_gold",
    without_df: bool = False,
):
    pred_lines = [f"{i}\t{lbl}" for i, lbl in enumerate(preds)]
    gold_lines = [f"{i}\t{lbl}" for i, lbl in enumerate(golds)]
    with open(gold_tmp_file, "w") as f:
        f.write("\n".join(gold_lines) + "\n")
    with open(pred_tmp_file, "w") as f:
        f.write("\n".join(pred_lines) + "\n")
    return run_semeval_evaluation(RUN_FILE, pred_tmp_file, gold_tmp_file, without_df=without_df)
