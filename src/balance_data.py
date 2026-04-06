"""
Balance train_data.csv so each class has ~52K rows (matching the smallest class).

Class 0 (label 1): COUNT         – keep all 52,515
Class 1 (label 2): BOOLEAN       – downsample to 52,515
Class 2 (label 3): Details       – downsample to 52,515
Class 3 (label 4): Fine-details  – downsample to 52,515

Produces: train_data_balanced.csv (original file is NOT modified)
"""

import csv
import random

INPUT_FILE = "train_data.csv"
OUTPUT_FILE = "train_data_balanced.csv"
RANDOM_SEED = 42
TARGET_PER_CLASS = 52000


def main():
    random.seed(RANDOM_SEED)

    # Group rows by label
    by_label = {}
    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            label = row[-1].strip()
            by_label.setdefault(label, []).append(row)

    print("Original distribution:")
    for label in sorted(by_label):
        print(f"  Label {label}: {len(by_label[label]):>10,}")

    # Balance: downsample larger classes, keep smaller classes as-is
    balanced = []
    for label in sorted(by_label):
        rows = by_label[label]
        if len(rows) > TARGET_PER_CLASS:
            rows = random.sample(rows, TARGET_PER_CLASS)
        balanced.extend(rows)

    # Shuffle
    random.shuffle(balanced)

    # Write
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(balanced)

    # Summary
    out_labels = {}
    for row in balanced:
        l = row[-1].strip()
        out_labels[l] = out_labels.get(l, 0) + 1

    print(f"\nBalanced distribution ({OUTPUT_FILE}):")
    for label in sorted(out_labels):
        print(f"  Label {label}: {out_labels[label]:>10,}")
    print(f"  Total:   {len(balanced):>10,}")


if __name__ == "__main__":
    main()
