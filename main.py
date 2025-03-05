import argparse
import yaml
from loguru import logger
from src.data import process_data
from src.model import load_model
from src.model_utils import load_tokenizer
from src.train import train_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/domain_router_config.yaml",
        help="Path to the config file (YAML)",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    logger.info(f"Config loaded: {config['name']}")

    # Process data & load tokenizer
    logger.info("Processing data...")
    dataset, category_map = process_data(config["data_path"], config["name"])
    logger.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config["model_name"])

    # Load model
    logger.info("Loading model...")
    peft_model = load_model(config["model_name"], config["model"], len(category_map))

    # Train model
    logger.info("Training model...")
    train_model(peft_model, dataset, category_map, tokenizer, config["train"])
