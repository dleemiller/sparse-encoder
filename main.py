import argparse
from pathlib import Path
from sparse_encoder.config import load_config
from sparse_encoder.train import train

def main():
    parser = argparse.ArgumentParser(prog="sparse-encoder-train")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train(cfg)

if __name__ == "__main__":
    main()

