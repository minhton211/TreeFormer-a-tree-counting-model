import os
from helper import Yosemite
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def argument_parser():
    parser = ArgumentParser(description="Density map generation",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", help="name of the dataset", type=str)

    parser.add_argument("train_paths", help="folder path of training data", type=str)
    parser.add_argument("test_paths", help="folder path of testing data", type=str)
    parser.add_argument("--dataset_path", help="folder path for this run h5 files", default=os.path.join(os.getcwd(), f"density_map"), type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    assert args.dataset in ['yosemite', 'london'], "Invalid dataset name, must be either 'yosemite' or 'london'"
    assert os.path.exists(args.train_paths), "Train path does not exist"
    assert os.path.exists(args.test_paths), "Test path does not exist"

    if args.dataset == 'yosemite':
        dataset = Yosemite(train_path=args.train_paths, test_path=args.test_paths, dataset_path=args.dataset_path)
        dataset.gen()