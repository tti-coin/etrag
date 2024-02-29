from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input", type=str)
    args = parser.parse_args()
    with open(args.input) as f:
        for i, l in enumerate(f):
            if i % 4 == 0:
                idx = l.split()[0]
            elif i % 4 == 1:
                print(idx, l.strip(), sep="\t")
