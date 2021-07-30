from argparse import ArgumentParser
from train import run_net


def main(name, mode):
    sub = run_net(name, mode)
    sub.to_csv('submission.csv', index=False)

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='tab')
    parser.add_argument('--mode', type=str, default='cv')
    args = parser.parse_args()

    main(args.name, args.mode)