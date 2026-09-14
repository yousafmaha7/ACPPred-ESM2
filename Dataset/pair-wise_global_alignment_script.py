import pandas as pd
from pathlib import Path
from Bio import pairwise2

# ============================================================
# 1. Input files
# ============================================================

train_path = Path("acp_train_bert_features.csv")
test_path = Path("acp_test_bert_features.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# ============================================================
# 2. Check required columns
# ============================================================

required_columns = ["ID", "Sequence", "Label"]

for col in required_columns:
    if col not in train.columns:
        raise ValueError(f"Column '{col}' not found in training dataset.")

    if col not in test.columns:
        raise ValueError(f"Column '{col}' not found in testing dataset.")


# ============================================================
# 3. Clean sequences
# ============================================================

def clean_sequence(sequence):
    return str(sequence).strip().upper()


train_sequences = [
    (
        str(row["ID"]),
        clean_sequence(row["Sequence"]),
        row["Label"]
    )
    for _, row in train.iterrows()
]

test_sequences = [
    (
        str(row["ID"]),
        clean_sequence(row["Sequence"]),
        row["Label"]
    )
    for _, row in test.iterrows()
]


# ============================================================
# 4. Pairwise global sequence similarity analysis
# ============================================================

results = []

for test_id, test_seq, test_label in test_sequences:

    best_match = None

    for train_id, train_seq, train_label in train_sequences:

        # Global sequence alignment
        alignment = pairwise2.align.globalxx(
            test_seq,
            train_seq,
            one_alignment_only=True
        )[0]

        aligned_test = alignment.seqA
        aligned_train = alignment.seqB

        # Alignment length
        alignment_length = len(aligned_test)

        # Number of identical residues
        matches = sum(
            a == b
            for a, b in zip(aligned_test, aligned_train)
        )

        # Percentage sequence identity
        identity = (
            100 * matches / alignment_length
            if alignment_length > 0
            else 0
        )

        current_match = {
            "Test_ID": test_id,
            "Test_Label": test_label,
            "Test_Sequence": test_seq,

            "Best_Training_ID": train_id,
            "Training_Label": train_label,
            "Training_Sequence": train_seq,

            "Sequence_Identity_percent": identity,
            "Alignment_Length": alignment_length,

            "Test_Length": len(test_seq),
            "Training_Length": len(train_seq)
        }

        # Keep the highest-identity training sequence
        if (
            best_match is None
            or identity >
            best_match["Sequence_Identity_percent"]
        ):
            best_match = current_match

    results.append(best_match)


# ============================================================
# 5. Create results dataframe
# ============================================================

similarity_results = pd.DataFrame(results)

# Sort from highest to lowest sequence identity
similarity_results = similarity_results.sort_values(
    "Sequence_Identity_percent",
    ascending=False
)


# ============================================================
# 6. Save complete results
# ============================================================

output_file = "train_test_sequence_similarity.csv"

similarity_results.to_csv(
    output_file,
    index=False
)


# ============================================================
# 7. Print summary
# ============================================================

print("\n==============================================")
print("TRAIN–TEST SEQUENCE SIMILARITY ANALYSIS")
print("==============================================")

print(f"Training sequences: {len(train_sequences)}")
print(f"Testing sequences:  {len(test_sequences)}")

print(
    f"\nTest sequences with >=90% identity: "
    f"{(similarity_results['Sequence_Identity_percent'] >= 90).sum()}"
)

print(
    f"Test sequences with >=80% identity: "
    f"{(similarity_results['Sequence_Identity_percent'] >= 80).sum()}"
)

print(
    f"Test sequences with >=70% identity: "
    f"{(similarity_results['Sequence_Identity_percent'] >= 70).sum()}"
)

print(
    f"\nMaximum sequence identity: "
    f"{similarity_results['Sequence_Identity_percent'].max():.2f}%"
)

print(
    f"Mean best-match identity: "
    f"{similarity_results['Sequence_Identity_percent'].mean():.2f}%"
)

print(
    f"Median best-match identity: "
    f"{similarity_results['Sequence_Identity_percent'].median():.2f}%"
)

print("\nTop 20 train–test matches:")
print(
    similarity_results[
        [
            "Test_ID",
            "Best_Training_ID",
            "Sequence_Identity_percent",
            "Alignment_Length",
            "Test_Length",
            "Training_Length"
        ]
    ].head(20).to_string(index=False)
)

print(f"\nResults saved to: {output_file}")
