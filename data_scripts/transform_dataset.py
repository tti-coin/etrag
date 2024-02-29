import json
import os
import re
from argparse import ArgumentParser

import datasets
from tqdm import tqdm


def transform_dataset(
    dataset: datasets.Dataset | datasets.DatasetDict,
    template_file_path,
    aug_note=False,
    aug_type=False,  # This option will not be used since entity types are not given.
    aug_tag=False,  # This option will be ignored since tags are already in the sentence.
    replace_basket_tags=False,
    neg_copy=False,
):
    # loading template file
    with open(template_file_path) as f:
        rel2temp = json.load(f)

    def _transform(each):
        sentence = each["sentence"]
        relation = each["relation"]

        template = rel2temp.get(relation, "")

        # Extracting entities based on <e1> and <e2> tags
        subj_match = re.search(r"<e1>(.*?)</e1>", sentence)
        obj_match = re.search(r"<e2>(.*?)</e2>", sentence)

        if subj_match and obj_match:
            subj = subj_match.group(1)
            obj = obj_match.group(1)
        else:
            subj = ""
            obj = ""

        if aug_note:
            head_sentence = f"The head entity is {subj} . "
            tail_sentence = f"The tail entity is {obj} . "
            sentence = head_sentence + tail_sentence + sentence

        if replace_basket_tags:
            sentence = sentence.replace("-LRB-", "(").replace("-RRB-", ")")
            sentence = sentence.replace("-LSB-", "[").replace("-RSB-", "]")
            subj = subj.replace("-LRB-", "(").replace("-RRB-", ")")
            subj = subj.replace("-LSB-", "[").replace("-RSB-", "]")
            obj = obj.replace("-LRB-", "(").replace("-RRB-", ")")
            obj = obj.replace("-LSB-", "[").replace("-RSB-", "]")

        target = template.format(subj=subj, obj=obj) if template else sentence

        if neg_copy and relation == "no_relation":
            target = sentence

        record = {
            "text": sentence,
            "target": target,
            "subj": subj,
            "obj": obj,
            "relation": relation,
        }
        return record

    # for debug
    a = _transform(dataset["train"][0])

    transformed_dataset = dataset.map(_transform)
    return transformed_dataset


# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("template_file_path", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--aug_note", action="store_true")
    parser.add_argument("--aug_type", action="store_true")
    parser.add_argument("--aug_tag", action="store_true")
    parser.add_argument("--replace_basket_tags", action="store_true")
    parser.add_argument("--neg_copy", action="store_true")
    parser.add_argument("--dev_split_mode", type=str, default="none", choices=["none", "step", "random"])
    parser.add_argument("--dev_split_ratio", type=float)
    parser.add_argument("--dev_split_step", type=int)

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.dataset)
    assert isinstance(
        dataset, datasets.DatasetDict
    ), "dataset should be of type `datasets.Dataset` or `datasets.DatasetDict`"
    lbl = dataset["train"].features["relation"]

    def _update_relation(x):
        x["label"] = lbl.int2str(x["relation"])
        return x

    dataset = dataset.map(_update_relation)
    dataset = dataset.remove_columns("relation")
    dataset = dataset.rename_column("label", "relation")

    # split to dev
    if args.dev_split_mode == "none":
        pass
    elif args.dev_split_mode == "step":
        dev_index = list(range(0, len(dataset["train"]), args.dev_split_step))
        dataset["dev"] = dataset["train"].select(dev_index)
        dataset["train"] = dataset["train"].select(list(set(range(len(dataset["train"]))) - set(dev_index)))
    elif args.dev_split_mode == "random":
        # split with train_test_split
        d = dataset["train"].train_test_split(test_size=args.dev_split_ratio)
        dataset["train"] = d["train"]
        dataset["dev"] = d["test"]

    transformed_dataset = transform_dataset(
        dataset,
        args.template_file_path,
        aug_note=args.aug_note,
        aug_type=args.aug_type,
        aug_tag=args.aug_tag,
        replace_basket_tags=args.replace_basket_tags,
        neg_copy=args.neg_copy,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # save as json
        if isinstance(transformed_dataset, datasets.DatasetDict):
            for split, data in transformed_dataset.items():
                output_dir = os.path.join(args.output_dir, f"{split}.json")
                data.to_json(output_dir)
                # d = data.to_list()
                # with open(output_dir, "w") as f:
                #     json.dump(d, f)
    else:
        print(transformed_dataset)
