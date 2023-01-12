import argparse



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=None, help='')
    args = parser.parse_args()
    return args