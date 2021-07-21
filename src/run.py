from argparse import ArgumentParser
from train import run_simple_net


def main(name):
    if args.name == 'simple':
        sub = run_simple_net(args.mode)
        sub.to_csv('submission.csv', index=False)

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='simple')
    parser.add_argument('--mode', type=str, default='cv')
    args = parser.parse_args()

    main(args)