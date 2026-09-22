"""Train the production startup funding model bundles."""

from __future__ import annotations

import logging

from startup_funding.config import BASE_DIR, DATA_DIR, MODELS_DIR
from startup_funding.data import load_and_clean_data, write_data_artifacts
from startup_funding.training import train_all, write_training_summary


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "model_training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading and validating startup funding data")
    data = load_and_clean_data(DATA_DIR)
    write_data_artifacts(data, DATA_DIR)
    logger.info("Prepared %s clean startup rows", len(data))

    logger.info("Training production model bundles")
    summary = train_all(data, MODELS_DIR)
    write_training_summary(summary, BASE_DIR / "training_results.json")
    logger.info("Training summary written to training_results.json")


if __name__ == "__main__":
    main()
