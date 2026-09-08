"""
Text-to-SQL dataset preparation pipeline.

Downloads the b-mc2/sql-create-context dataset,
cleans the examples, creates train/validation/test splits,
and saves them to disk.

This script is intentionally independent of model training.
"""

from pathlib import Path

from datasets import load_dataset


DATASET_ID = "b-mc2/sql-create-context"
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def load_raw_dataset():
    """Download the original dataset from Hugging Face."""
    print(f"Loading dataset: {DATASET_ID}")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
    )

    print(f"Loaded {len(dataset):,} examples")

    return dataset


def clean_dataset(dataset):
    """
    Remove invalid examples and normalize fields.

    Expected fields:
        question
        context
        answer
    """

    required_columns = {"question", "context", "answer"}

    missing_columns = required_columns - set(dataset.column_names)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    def is_valid(example):
        question = example["question"]
        context = example["context"]
        answer = example["answer"]

        return all(
            isinstance(value, str) and value.strip()
            for value in [question, context, answer]
        )

    before = len(dataset)

    dataset = dataset.filter(is_valid)

    after = len(dataset)

    print(f"Removed {before - after:,} invalid examples")
    print(f"Remaining examples: {after:,}")

    return dataset


def create_splits(dataset):
    """
    Create reproducible train/validation/test splits.

    Split:
        90% train
         5% validation
         5% test
    """

    split = dataset.train_test_split(
        test_size=0.10,
        seed=SEED,
    )

    train_dataset = split["train"]

    temp_dataset = split["test"].train_test_split(
        test_size=0.50,
        seed=SEED,
    )

    validation_dataset = temp_dataset["train"]
    test_dataset = temp_dataset["test"]

    print("\nDataset splits:")
    print(f"Train      : {len(train_dataset):,}")
    print(f"Validation : {len(validation_dataset):,}")
    print(f"Test       : {len(test_dataset):,}")

    return train_dataset, validation_dataset, test_dataset


def save_splits(train_dataset, validation_dataset, test_dataset):
    """Save processed datasets to disk."""

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLITS_DIR / "train"
    validation_path = SPLITS_DIR / "validation"
    test_path = SPLITS_DIR / "test"

    train_dataset.save_to_disk(str(train_path))
    validation_dataset.save_to_disk(str(validation_path))
    test_dataset.save_to_disk(str(test_path))

    print("\nSaved datasets:")
    print(f"Train      : {train_path}")
    print(f"Validation : {validation_path}")
    print(f"Test       : {test_path}")


def main():
    """Run the complete dataset preparation pipeline."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_raw_dataset()

    dataset = clean_dataset(dataset)

    train_dataset, validation_dataset, test_dataset = create_splits(
        dataset
    )

    save_splits(
        train_dataset,
        validation_dataset,
        test_dataset,
    )

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()