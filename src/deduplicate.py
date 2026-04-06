"""
Deduplicate train_data.csv by removing rows that give the same meaning.

"Same meaning" = rows that share the same question template and label,
differing only in entity values (device names, optics IDs, port numbers, IP addresses).

Produces: train_data_deduped.csv (original file is NOT modified)
"""

import csv
import re

INPUT_FILE = "train_data.csv"
OUTPUT_FILE = "train_data_deduped.csv"


def normalize(question: str) -> str:
    """Replace entity values with placeholders to extract the question template."""
    q = question.strip().lower()

    # Replace optics identifiers like optics0/4/0/3
    q = re.sub(r'optics\d+/\d+/\d+/\d+', '{OPTICS}', q)

    # Replace IP addresses (must come before generic number replacement)
    q = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?', '{IP}', q)

    # Replace hostnames with hyphens/dots (e.g. c8k-01.pth.lab, mx10k-02.cw.net)
    q = re.sub(r'[a-z0-9]+(?:-[a-z0-9]+)+(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)*', '{DEVICE}', q, flags=re.IGNORECASE)

    # Replace device-like names with dots (e.g. devlf1clr01.vftest.de)
    q = re.sub(r'[a-z]{2,}\d+[a-z0-9]*(?:\.[a-z0-9]+)+', '{DEVICE}', q, flags=re.IGNORECASE)

    # Replace device-like names without dots (e.g. ffmdls1, dectevagg6, dectevpreagg16)
    q = re.sub(r'\b[a-z]{2,}\d+[a-z0-9]*\b', '{DEVICE}', q, flags=re.IGNORECASE)

    # Replace port-style identifiers like 0/4/0/3
    q = re.sub(r'\d+/\d+(?:/\d+)*', '{PORT}', q)

    # Replace standalone numbers (remaining)
    q = re.sub(r'\b\d+\b', '{NUM}', q)

    return q


def main():
    seen_templates = set()
    kept_rows = []
    total = 0
    exact_dupes = 0
    template_dupes = 0

    seen_exact = set()

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            total += 1
            question = row[0].strip()
            label = row[-1].strip()

            # Skip exact duplicates
            exact_key = (question, label)
            if exact_key in seen_exact:
                exact_dupes += 1
                continue
            seen_exact.add(exact_key)

            # Skip template duplicates (same meaning, different entities)
            template = normalize(question)
            template_key = (template, label)
            if template_key in seen_templates:
                template_dupes += 1
                continue
            seen_templates.add(template_key)

            kept_rows.append(row)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(kept_rows)

    print(f"{'Total rows:':<30} {total:>10,}")
    print(f"{'Exact duplicates removed:':<30} {exact_dupes:>10,}")
    print(f"{'Same-meaning rows removed:':<30} {template_dupes:>10,}")
    print(f"{'Rows kept:':<30} {len(kept_rows):>10,}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
