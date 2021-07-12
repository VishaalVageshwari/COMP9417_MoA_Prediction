from argparse import ArgumentParser
from train import train_simple_net


def main(name):
    if name == 'simple':
        train_simple_net()
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='simple')
    args = parser.parse_args()

    main(args.name)