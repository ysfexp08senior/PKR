import argparse
import pandas as pd
from .pkr import PKR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pure Kernel Regression")
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--test", required=True, help="Path to test CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output", required=True, help="Where to save predictions CSV")
    parser.add_argument("--kernels-out", default="kernels.txt", help="Path to save kernels")
    parser.add_argument("--max-dim", type=int, default=3, help="Maximum kernel dimension")
    parser.add_argument("--window", type=float, default=0.5, help="Numeric window width")
    parser.add_argument("--step", type=float, default=0.25, help="Step for numeric windows")
    parser.add_argument("--min-purity", type=float, default=0.9, help="Minimum kernel purity")
    parser.add_argument("--min-count", type=int, default=10, help="Minimum rows per kernel")
    parser.add_argument("--top-k", type=int, default=120, help="Number of kernels to keep")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (-1 all cores)")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    model = PKR(
        max_dim=args.max_dim,
        window=args.window,
        step=args.step,
        min_purity=args.min_purity,
        min_count=args.min_count,
        top_k=args.top_k,
        n_jobs=args.n_jobs,
    ).fit(train_df, target=args.target)

    preds = model.predict(test_df)
    pd.DataFrame({args.target: preds}).to_csv(args.output, index=False)

    if args.kernels_out:
        model.get_kernels().to_csv(args.kernels_out, index=False, sep="\t")


if __name__ == "__main__":
    main()
