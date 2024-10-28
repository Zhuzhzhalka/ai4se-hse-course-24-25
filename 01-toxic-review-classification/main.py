import argparse
from pathlib import Path

from toxic_clf.data import load_dataset, prepare, save_dataset
from toxic_clf.models import classifier


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    default_data_path = Path('./prepared-dataset')
    prepare_data_parser = subparsers.add_parser('prepare-data')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        'input',
        help='Path to load raw dataset',
        type=Path,
    )
    prepare_data_parser.add_argument(
        '-o',
        '--output',
        help='Path to save prepared dataset to',
        type=Path,
        default=default_data_path,
    )

    predict_parser = subparsers.add_parser('classify')
    predict_parser.set_defaults(func=classify)
    predict_parser.add_argument(
        '-d',
        '--dataset',
        help='Path to prepared dataset',
        type=Path,
        default=default_data_path,
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        choices=['classic_ml', 'roberta-base'],
        default='classic_ml',
    )

    return parser.parse_args()


def prepare_data(args):
    dataset = prepare(args.input)
    save_dataset(dataset, args.output)


def classify(args):
    dataset = load_dataset(args.dataset)
    classifier(dataset, args.model)


if __name__ == '__main__':
    main()
