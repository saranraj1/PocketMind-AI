import argparse
from pocketmind.config import PocketMindConfig

def main():
    parser = argparse.ArgumentParser(description="Print resolved PocketMind configuration")
    parser.add_argument("--config", type=str, default="configs/debug.yaml", help="Path to config yaml file")
    args = parser.parse_args()

    try:
        config = PocketMindConfig.from_yaml(args.config)
        print("Resolved Configuration:")
        for k, v in config.model_dump().items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

if __name__ == "__main__":
    main()
