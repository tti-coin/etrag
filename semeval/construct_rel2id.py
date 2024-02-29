import json
import os
import re
from argparse import ArgumentParser
from typing import Optional

from eval import parse_semeval_label


def read_semeval_rel(
    file_in: str,
    rel2id: Optional[dict[str, int]] = None,
    ignore_direction: bool = False,
) -> dict[str, int]:
    if rel2id is None:
        rel2id = {"Other": 0}
    fp = open(file_in)
    for l, line in enumerate(fp):
        if l % 4 == 1:
            # label
            line = line.strip()
            label_str, is_reverse = parse_semeval_label(line, ignore_direction=ignore_direction)
            if not label_str in rel2id:
                rel2id[label_str] = len(rel2id)
    fp.close()
    return rel2id


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--ignore_direction", action="store_true")
    args = parser.parse_args()
    rel2id = read_semeval_rel(
        os.path.join(args.data_dir, "train.txt"), ignore_direction=args.ignore_direction, rel2id=None
    )
    rel2id = read_semeval_rel(
        os.path.join(args.data_dir, "dev.txt"), ignore_direction=args.ignore_direction, rel2id=rel2id
    )
    rel2id = read_semeval_rel(
        os.path.join(args.data_dir, "test.txt"), ignore_direction=args.ignore_direction, rel2id=rel2id
    )
    print(json.dumps(rel2id, indent=4))
