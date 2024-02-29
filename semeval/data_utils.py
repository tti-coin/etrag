import json
import re
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

SEMEVAL_E1_START = "<e1>"
SEMEVAL_E1_END = "</e1>"
SEMEVAL_E2_START = "<e2>"
SEMEVAL_E2_END = "</e2>"


def parse_semeval_label(label_line: str, ignore_direction: bool = False) -> tuple[str, Optional[bool]]:
    label_match_wo_dir = re.match("^(.*(?=\()|Other)", label_line)
    assert label_match_wo_dir is not None
    label = label_match_wo_dir.group()
    if label == "Other":
        is_reverse = None
    else:
        is_reverse = re.match(".*e1.*e2", label_line) is not None
    if ignore_direction:
        return label, is_reverse
    else:
        if label != "Other":
            if is_reverse:
                label = label + "(e1,e2)"
            else:
                label = label + "(e2,e1)"
        return label, is_reverse


def process_semeval_sent(sent: str) -> tuple[str, list[list[tuple[int, int]]]]:
    # strip unecesery characters
    sent = sent.strip().strip('"')
    e1_start = sent.find(SEMEVAL_E1_START)
    e1_end = sent.find(SEMEVAL_E1_END)
    e2_start = sent.find(SEMEVAL_E2_START)
    e2_end = sent.find(SEMEVAL_E2_END)
    assert e1_start != -1 and e1_end != -1 and e2_start != -1 and e2_end != -1
    assert e1_start < e1_end and e2_start < e2_end
    sent_wotag = (
        sent.replace(SEMEVAL_E1_START, "")
        .replace(SEMEVAL_E1_END, "")
        .replace(SEMEVAL_E2_START, "")
        .replace(SEMEVAL_E2_END, "")
    )
    sent_wotag = re.sub(" +", " ", sent_wotag.strip())

    # adjust entity position after removing tags
    if e1_start < e2_start:
        e1_end -= len(SEMEVAL_E1_START)
        e2_start -= len(SEMEVAL_E1_START) + len(SEMEVAL_E1_END)
        e2_end -= len(SEMEVAL_E1_START) + len(SEMEVAL_E1_END) + len(SEMEVAL_E2_START)
    else:
        e2_end -= len(SEMEVAL_E2_START)
        e1_start -= len(SEMEVAL_E2_START) + len(SEMEVAL_E2_END)
        e1_start -= len(SEMEVAL_E2_START) + len(SEMEVAL_E2_END) + len(SEMEVAL_E1_START)

    return sent_wotag, [[(e1_start, e1_end)], [(e2_start, e2_end)]]


def read_semeval(
    file_in: str,
    rel2id: dict[str, int],
    without_label: bool = False,
    undirected: bool = False,
) -> list[dict]:
    features = []
    fp = open(file_in)
    buf = {}
    for l, line in enumerate(fp):
        if not line:
            continue
        if without_label or l % 4 == 0:
            # 2 columns: index, sentence
            cells = line.strip().split("\t")
            idx, sentence = int(cells[0]), str(cells[1])
            buf["index"] = idx
            buf["sentence"] = sentence
        elif l % 4 == 1:
            # label
            line = line.strip()
            label_str, is_reverse = parse_semeval_label(line, ignore_direction=undirected)
            buf["label"] = label_str
            buf["is_reverse"] = is_reverse
        else:
            if not ("sentence" in buf and "label" in buf and "index" in buf):
                buf = {}
                continue
            sentence, entity_pos = process_semeval_sent(buf["sentence"])
            feature = {
                "index": buf["index"],
                "sentence": sentence,
                "entity_pos": entity_pos,
                "e1": sentence[entity_pos[0][0][0] : entity_pos[0][0][1]],
                "e2": sentence[entity_pos[1][0][0] : entity_pos[1][0][1]],
                "labels": [rel2id[buf["label"]]],
                "label": buf["label"],
                "title": "SemEval_{}".format(buf["index"]),
                "is_reverse": buf["is_reverse"],
            }
            features.append(feature)

            buf = {}
    fp.close()
    return features


def label2str(label: int, id2rel: dict[int, str], gold_rev: Optional[bool] = None) -> str:
    rel_label = id2rel[label]
    if gold_rev is None:
        rev = False
        if rel_label.endswith("_reverse") or rel_label.endswith("(e2,e1)"):
            rev = True
            rel_label = rel_label[:-8]
    else:
        rev = gold_rev
    if rel_label != "Other":
        if rev:
            rel_label = rel_label + "(e2,e1)"
        else:
            rel_label = rel_label + "(e1,e2)"
    return rel_label
