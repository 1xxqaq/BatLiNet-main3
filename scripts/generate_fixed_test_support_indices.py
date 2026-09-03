import argparse
from pathlib import Path

import torch

from scripts.pipeline import build_dataset, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate a fixed test support protocol for fair BatLiNet '
            'evaluation.'))
    parser.add_argument('config_path')
    parser.add_argument('--workspace')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--test-support-size', type=int, default=32)
    parser.add_argument(
        '--output-path',
        required=True,
        help='Path to save the generated protocol (.pt).')
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_config(args.config_path, args.workspace)
    dataset = build_dataset(configs, args.device)

    num_train = len(dataset.train_data)
    num_test = len(dataset.test_data)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(args.seed)
    indices = torch.randint(
        low=0,
        high=num_train,
        size=(num_test, args.test_support_size),
        generator=generator,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'seed': args.seed,
            'num_train_samples': num_train,
            'num_test_samples': num_test,
            'test_support_size': args.test_support_size,
            'indices': indices,
        },
        output_path,
    )
    print(f'Saved fixed test support protocol to {output_path}')
    print(
        f'num_test_samples={num_test}, '
        f'test_support_size={args.test_support_size}')


if __name__ == '__main__':
    main()
