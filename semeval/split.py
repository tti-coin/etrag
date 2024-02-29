from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("file", type=str)
parser.add_argument("step", type=int)
parser.add_argument("out1", type=str)
parser.add_argument("out2", type=str)
parser.add_argument("--n_line", default=4, type=int)
args = parser.parse_args()

in_f = open(args.file)
out1_f = open(args.out1, "w")
out2_f = open(args.out2, "w")

for l, line in enumerate(in_f):
    if (l // args.n_line + 1) % args.step:
        out1_f.write(line)
    else:
        out2_f.write(line)

in_f.close()
out1_f.close()
out2_f.close()
