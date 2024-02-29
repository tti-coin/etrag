import os
import re
import subprocess
from typing import Any, Optional

import pandas as pd

from .data_utils import label2str

OTHER_KEY = "_O_"
XDIRX_KEY = "xDIRx"
SKIP_KEY = "skip"
RUN_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl"
)


def labels_to_official(
    data: list[dict[str, Any]], labels: list[int], rel2id: dict[str, int], undirected: bool = False
) -> str:
    id2rel = {v: k for k, v in rel2id.items()}
    lines = []
    for i, (dat, lbl) in enumerate(zip(data, labels), start=1):
        if undirected:
            # if undirected, add correct direction
            lbl_str = label2str(lbl, id2rel, gold_rev=dat["is_reverse"])
            lines.append(f"{dat['index']}\t{lbl_str}")
        else:
            lines.append(f"{dat['index']}\t{id2rel[lbl]}")
    return "\n".join(lines) + "\n"


def _read_confusion_matrix_line(line: str) -> tuple[Optional[str], Optional[list[int]], Optional[int], Optional[int]]:
    if line.strip().startswith("+-"):
        return None, None, None, None
    cols = line.split()
    # 0: row name
    label = cols[0]
    # 1: ignore frame
    cnt = 0
    for c in cols[2:]:
        if c == "|":
            break
        cnt += 1
    # 2-2+cnt-1: counts
    counts = list(map(int, cols[2 : 2 + cnt]))
    # -2: skip
    skip = int(cols[-2])
    # -3: xdirx
    xdirx = None
    if len(cols[2 + cnt + 1 :]) == 4:
        xdirx = int(cols[-3])
    # ignore remains: frame, summary that can be computed from counts
    return label, counts, skip, xdirx


def _parse_confusion_matrix_block(lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 0: Confusion matrix: (header)
    # 1: column names
    col_labels = lines[1].split()[:-3]
    # 2: ignore frame
    # 3-3+len(col_labels): matrix
    labels, counts, skips, xdirx = zip(*list(map(_read_confusion_matrix_line, lines[3 : 3 + len(col_labels)])))
    exclude_none = lambda x: [e for e in x if e is not None]
    labels = exclude_none(labels)
    counts = exclude_none(counts)
    skips = exclude_none(skips)
    xdirx = exclude_none(xdirx)
    confusion_matrix = pd.DataFrame(counts, index=labels, columns=col_labels)
    other_data = [skips, xdirx] if xdirx else [skips]
    other_labels = [SKIP_KEY, XDIRX_KEY] if xdirx else [SKIP_KEY]
    other_df = pd.DataFrame(other_data, index=other_labels, columns=labels)
    # ignore remains
    return confusion_matrix, other_df


def _parse_scores_block(lines: list[str]) -> dict[str, Any]:
    report = dict()
    # 0: coverage
    # 1: Accuracy (calculated for the above confusion matrix)
    # 2: Accuracy (considering all skipped examples as Wrong)
    # 3: Accuracy (considering all skipped examples as Other)
    report["coverage"] = float(lines[0].split()[-1][:-1])
    report["accuracy"] = float(lines[1].split()[-1][:-1])
    report["accuracy_skipped_as_wrong"] = float(lines[2].split()[-1][:-1])
    report["accuracy_skipped_as_other"] = float(lines[3].split()[-1][:-1])
    return report


def _compute_scores_from_confusion_matrix(confusion_matrix: pd.DataFrame, other_df: pd.DataFrame) -> dict[str, float]:
    scores = dict()
    # # precision
    # scores["precision"] = confusion_matrix.values.diagonal().sum() / (confusion_matrix.sum(axis=0) + 1e-10) * 100
    # # recall
    # scores["recall"] = confusion_matrix.values.diagonal().sum() / (confusion_matrix.sum(axis=1) + 1e-10) * 100
    # # F1
    # scores["f1"] = 2 * scores["precision"] * scores["recall"] / (scores["precision"] + scores["recall"] + 1e-10) * 100
    # support
    scores["support"] = confusion_matrix.sum(axis=1)
    # coverage
    scores["coverage"] = (
        (confusion_matrix.values.sum() + other_df[XDIRX_KEY:XDIRX_KEY].values.sum())
        / (confusion_matrix.values.sum() + other_df[SKIP_KEY:SKIP_KEY].values.sum())
        * 100
    )
    # accuracy
    scores["accuracy"] = confusion_matrix.values.diagonal().sum() / confusion_matrix.values.sum() * 100
    # accuracy (considering all skipped examples as Wrong)
    scores["accuracy_skipped_as_wrong"] = (
        confusion_matrix.values.diagonal().sum()
        / (confusion_matrix.values.sum() + other_df[SKIP_KEY:SKIP_KEY].values.sum())
        * 100
    )
    # accuracy (considering all skipped examples as Other)
    scores["accuracy_skipped_as_other"] = (
        (confusion_matrix.values.diagonal().sum() + other_df[SKIP_KEY:SKIP_KEY][OTHER_KEY].sum())
        / (confusion_matrix.values.sum() + other_df[SKIP_KEY:SKIP_KEY].values.sum())
        * 100
    )

    return scores


def _read_class_score_line(line: str) -> tuple[str, dict[str, float]]:
    scores = dict()
    cols = line.split()
    # 0: class name
    class_name = cols[0]
    # extract values ending with %
    values = [float(c[:-1]) for c in cols if c.endswith("%")]

    scores["precision"] = values[0]
    scores["recall"] = values[1]
    scores["f1"] = values[2]
    return class_name, scores


def _parse_class_scores_block(lines: list[str]) -> pd.DataFrame:
    # 0: Results for the individual relations:
    # 1-end: score for each class
    class_scores = dict()
    class_names, class_scores = zip(*list(map(_read_class_score_line, lines[1:])))
    return pd.DataFrame(class_scores, index=class_names)


def _read_block(lines: list[str], without_df: bool = False) -> dict[str, Any]:
    report = dict()
    # blocks is devided by blank lines
    blank_lines = [i for i, line in enumerate(lines) if line == ""]

    # 1st block: header
    # 0: header <<< (2*9+1)-WAY EVALUATION (USING DIRECTIONALITY)>>>:
    # 2nd block: confusion matrix
    confusion_matrix_lines = lines[blank_lines[0] + 1 : blank_lines[1]]
    confusion_matrix, other_df = _parse_confusion_matrix_block(confusion_matrix_lines)
    report["confusion_matrix"] = confusion_matrix.to_dict() if without_df else confusion_matrix
    report["other"] = other_df.to_dict() if without_df else other_df

    # 3rd block: scores based on the confusion matrix
    scores_lines = lines[blank_lines[1] + 1 : blank_lines[2]]
    scores = _parse_scores_block(scores_lines)
    report.update(scores)

    # assertation of scores
    computed_scores = _compute_scores_from_confusion_matrix(confusion_matrix, other_df)
    assert (report["coverage"] - computed_scores["coverage"]) < 0.01
    assert (report["accuracy"] - computed_scores["accuracy"]) < 0.01
    assert (report["accuracy_skipped_as_wrong"] - computed_scores["accuracy_skipped_as_wrong"]) < 0.01
    assert (report["accuracy_skipped_as_other"] - computed_scores["accuracy_skipped_as_other"]) < 0.01

    # 4th block: scores for each class
    class_scores_lines = lines[blank_lines[2] + 1 : blank_lines[3]]
    class_scores = _parse_class_scores_block(class_scores_lines)
    report["class_scores"] = class_scores.to_dict() if without_df else class_scores

    # 5th block: micro-F1
    micro_f1_lines = lines[blank_lines[3] + 1 : blank_lines[4]]
    # 0: micro-F1 header
    # 1: micro-F1
    vals_after_equal = []
    sp = micro_f1_lines[1].split()
    for i, t in enumerate(sp[1:], start=1):
        if sp[i - 1] == "=":
            vals_after_equal.append(t)
    report["micro-precision"] = float(vals_after_equal[1][:-1])
    report["micro-recall"] = float(vals_after_equal[3][:-1])
    report["micro-f1"] = float(vals_after_equal[-1][:-1])

    # 6th block: macro-F1
    macro_f1_lines = lines[blank_lines[4] + 1 : blank_lines[5]]
    # 0: macro-F1 header
    # 1: macro-F1
    vals_after_equal = []
    sp = macro_f1_lines[1].split()
    for i, t in enumerate(sp[1:], start=1):
        if sp[i - 1] == "=":
            vals_after_equal.append(t)
    report["macro-precision"] = float(vals_after_equal[0][:-1])
    report["macro-recall"] = float(vals_after_equal[1][:-1])
    report["macro-f1"] = float(vals_after_equal[2][:-1])
    # report["macro-precision"] = float(macro_f1_lines[1].split()[2][:-1])
    # report["macro-recall"] = float(macro_f1_lines[1].split()[5][:-1])
    # report["macro-f1"] = float(macro_f1_lines[1].split()[8][:-1])

    return report


def parse_eval(report_text: str, without_df: bool = False) -> dict[str, Any]:
    report = dict()
    report_text = re.sub(r" +", " ", report_text)
    lines = list(map(str.strip, report_text.split("\n")))
    # blocks is devided by header starting with <<<
    header_lines = [i for i, line in enumerate(lines) if line.startswith("<<<")]
    blocks: list[list[str]] = []
    for i in range(len(header_lines) - 1):
        blocks.append(lines[header_lines[i] : header_lines[i + 1]])
    reports = list(map(lambda x: _read_block(x, without_df=without_df), blocks))
    report["directed"] = reports[0]
    report["undirected"] = reports[1]
    report["official"] = reports[2]
    return report


def run_semeval_evaluation(run_file: str, pred_file: str, gold_file: str, without_df: bool = False) -> dict[str, Any]:
    pipe = subprocess.Popen(["perl", run_file, pred_file, gold_file], stdout=subprocess.PIPE)
    eval_out, _ = pipe.communicate()
    score = parse_eval(eval_out.decode("utf-8"), without_df=without_df)
    return score


def evaluate_semeval(
    prediction_string: str,
    gold_file: str,
    temp_pred_file: str = ".tmp_pred",
    run_file: str = "SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl",
    without_df: bool = False,
) -> dict[str, Any]:
    prediction_file = os.path.join(temp_pred_file)
    with open(prediction_file, "w") as f:
        f.write(prediction_string)
    score = run_semeval_evaluation(run_file, prediction_file, gold_file, without_df=without_df)
    return score
