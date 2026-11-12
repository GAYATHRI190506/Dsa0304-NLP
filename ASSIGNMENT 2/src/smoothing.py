import re
import math
from collections import Counter


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = "data/train.txt"
DEV_FILE = "data/dev.txt"


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    text = text.lower()

    return re.findall(
        r"[a-z]+(?:'[a-z]+)?|\d+(?::\d+)?|[^\w\s]",
        text
    )


# ============================================================
# READ DATA
# ============================================================

def read_sentences(filename):

    with open(filename, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


train_sentences = read_sentences(TRAIN_FILE)
dev_sentences = read_sentences(DEV_FILE)


# ============================================================
# TOKENIZE
# ============================================================

train_data = []

dev_data = []

for sentence in train_sentences:
    train_data.append(
        ["<s>"] + tokenize(sentence) + ["</s>"]
    )

for sentence in dev_sentences:
    dev_data.append(
        ["<s>"] + tokenize(sentence) + ["</s>"]
    )


# ============================================================
# N-GRAM COUNTS
# ============================================================

unigram_counts = Counter()
bigram_counts = Counter()

for sentence in train_data:

    for word in sentence:
        unigram_counts[word] += 1

    for i in range(len(sentence) - 1):

        bigram_counts[
            (sentence[i], sentence[i + 1])
        ] += 1


vocabulary = set(unigram_counts)

V = len(vocabulary)

total_tokens = sum(unigram_counts.values())


# ============================================================
# ADD-K SMOOTHING
# ============================================================

def add_k_probability(w1, w2, k):

    numerator = (
        bigram_counts[(w1, w2)] + k
    )

    denominator = (
        unigram_counts[w1] + k * V
    )

    return numerator / denominator


# ============================================================
# BIGRAM MLE
# ============================================================

def mle_bigram_probability(w1, w2):

    denominator = unigram_counts[w1]

    if denominator == 0:
        return 0.0

    return (
        bigram_counts[(w1, w2)]
        / denominator
    )


# ============================================================
# GOOD-TURING COUNTS
# ============================================================

bigram_frequency_of_frequency = Counter(
    bigram_counts.values()
)


def good_turing_adjusted_count(count):

    """
    Good-Turing adjusted count:

    c* = (c + 1) * N(c+1) / N(c)

    where N(c) is the number of n-grams
    occurring c times.
    """

    Nc = bigram_frequency_of_frequency.get(
        count,
        0
    )

    Nc_plus_1 = bigram_frequency_of_frequency.get(
        count + 1,
        0
    )

    if Nc == 0 or Nc_plus_1 == 0:
        return float(count)

    return (
        (count + 1)
        * Nc_plus_1
        / Nc
    )


# ============================================================
# GOOD-TURING BIGRAM PROBABILITY
# ============================================================

def good_turing_probability(w1, w2):

    count = bigram_counts[
        (w1, w2)
    ]

    denominator = unigram_counts[w1]

    if denominator == 0:
        return 0.0

    if count == 0:

        # Estimate unseen probability mass
        N1 = bigram_frequency_of_frequency.get(
            1,
            0
        )

        total_bigrams = sum(
            bigram_counts.values()
        )

        if total_bigrams == 0:
            return 0.0

        return (
            N1 /
            (total_bigrams * V)
        )

    adjusted_count = (
        good_turing_adjusted_count(count)
    )

    return adjusted_count / denominator


# ============================================================
# SENTENCE LOG PROBABILITY
# ============================================================

def log_probability(
    sentence,
    probability_function
):

    total_log_probability = 0.0

    for i in range(1, len(sentence)):

        probability = probability_function(
            sentence[i - 1],
            sentence[i]
        )

        if probability <= 0:
            return float("-inf")

        total_log_probability += math.log(
            probability
        )

    return total_log_probability


# ============================================================
# PERPLEXITY
# ============================================================

def calculate_perplexity(
    dataset,
    probability_function
):

    total_log_probability = 0.0
    total_predictions = 0

    zero_probability_sentences = 0

    for sentence in dataset:

        sentence_log_probability = (
            log_probability(
                sentence,
                probability_function
            )
        )

        if sentence_log_probability == float("-inf"):

            zero_probability_sentences += 1

            # Give a very small probability
            # so perplexity can still be measured.
            sentence_log_probability = (
                math.log(1e-12)
                * (len(sentence) - 1)
            )

        total_log_probability += (
            sentence_log_probability
        )

        total_predictions += (
            len(sentence) - 1
        )

    if total_predictions == 0:
        return float("inf")

    perplexity = math.exp(
        -total_log_probability
        / total_predictions
    )

    return perplexity, zero_probability_sentences


# ============================================================
# DISPLAY HEADER
# ============================================================

print("=" * 70)
print("TASK 3 - SMOOTHING LANGUAGE MODELS")
print("=" * 70)


# ============================================================
# CORPUS INFORMATION
# ============================================================

print("\nCORPUS INFORMATION")
print("-" * 70)

print(
    f"Training sentences : {len(train_data)}"
)

print(
    f"Development sentences : {len(dev_data)}"
)

print(
    f"Vocabulary size : {V}"
)

print(
    f"Unique bigrams : {len(bigram_counts)}"
)


# ============================================================
# ADD-K COMPARISON
# ============================================================

print("\nADD-K SMOOTHING")
print("-" * 70)

k_values = [
    0.01,
    0.1,
    0.5,
    1.0
]

add_k_results = []

for k in k_values:

    perplexity, zero_sentences = (
        calculate_perplexity(
            dev_data,
            lambda w1, w2:
                add_k_probability(w1, w2, k)
        )
    )

    add_k_results.append(
        (k, perplexity, zero_sentences)
    )

    print(
        f"k = {k:<5} "
        f"Perplexity = {perplexity:.4f} "
        f"Zero-probability sentences = "
        f"{zero_sentences}"
    )


# ============================================================
# UNSMOOTHED MLE
# ============================================================

print("\nUNSMOOTHED MLE")
print("-" * 70)

mle_perplexity, mle_zero = (
    calculate_perplexity(
        dev_data,
        mle_bigram_probability
    )
)

print(
    f"MLE Perplexity = "
    f"{mle_perplexity:.4f}"
)

print(
    f"Zero-probability sentences = "
    f"{mle_zero}"
)


# ============================================================
# GOOD-TURING
# ============================================================

print("\nGOOD-TURING SMOOTHING")
print("-" * 70)

gt_perplexity, gt_zero = (
    calculate_perplexity(
        dev_data,
        good_turing_probability
    )
)

print(
    f"Good-Turing Perplexity = "
    f"{gt_perplexity:.4f}"
)

print(
    f"Zero-probability sentences = "
    f"{gt_zero}"
)


# ============================================================
# HAND-COMPUTED EXAMPLE
# ============================================================

print("\nHAND-COMPUTED ADD-K EXAMPLE")
print("-" * 70)

w1 = "the"
w2 = "train"
k = 1.0

count_bigram = bigram_counts[
    (w1, w2)
]

count_context = unigram_counts[
    w1
]

laplace_probability = (
    count_bigram + k
) / (
    count_context + k * V
)

print(
    f"Bigram: ({w1}, {w2})"
)

print(
    f"Count({w1},{w2}) = "
    f"{count_bigram}"
)

print(
    f"Count({w1}) = "
    f"{count_context}"
)

print(
    f"Vocabulary size = {V}"
)

print(
    f"k = {k}"
)

print(
    f"P({w2}|{w1}) = "
    f"({count_bigram} + {k}) / "
    f"({count_context} + {k} × {V})"
)

print(
    f"Probability = "
    f"{laplace_probability:.10f}"
)


# ============================================================
# GOOD-TURING EXAMPLES
# ============================================================

print("\nGOOD-TURING COUNT EXAMPLES")
print("-" * 70)

for count in [1, 2, 3]:

    adjusted = (
        good_turing_adjusted_count(count)
    )

    Nc = bigram_frequency_of_frequency.get(
        count,
        0
    )

    Nc1 = bigram_frequency_of_frequency.get(
        count + 1,
        0
    )

    print(
        f"c={count}: "
        f"N(c)={Nc}, "
        f"N(c+1)={Nc1}, "
        f"c*={adjusted:.6f}"
    )


# ============================================================
# BEST ADD-K
# ============================================================

best_add_k = min(
    add_k_results,
    key=lambda x: x[1]
)

print("\nBEST ADD-K MODEL")
print("-" * 70)

print(
    f"Best k = {best_add_k[0]}"
)

print(
    f"Perplexity = "
    f"{best_add_k[1]:.4f}"
)


# ============================================================
# COMPARISON
# ============================================================

print("\nMODEL COMPARISON")
print("-" * 70)

print(
    f"{'Model':<25}"
    f"{'Perplexity':>15}"
)

print("-" * 40)

print(
    f"{'MLE':<25}"
    f"{mle_perplexity:>15.4f}"
)

for k, perplexity, _ in add_k_results:

    print(
        f"{'Add-k (k=' + str(k) + ')':<25}"
        f"{perplexity:>15.4f}"
    )

print(
    f"{'Good-Turing':<25}"
    f"{gt_perplexity:>15.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "results/task3_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 3 - SMOOTHING LANGUAGE MODELS\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Training sentences: {len(train_data)}\n"
    )

    file.write(
        f"Development sentences: {len(dev_data)}\n"
    )

    file.write(
        f"Vocabulary size: {V}\n"
    )

    file.write(
        f"Unique bigrams: {len(bigram_counts)}\n\n"
    )

    file.write(
        "MODEL COMPARISON\n"
    )

    file.write(
        f"MLE: {mle_perplexity:.4f}\n"
    )

    for k, perplexity, _ in add_k_results:

        file.write(
            f"Add-k (k={k}): "
            f"{perplexity:.4f}\n"
        )

    file.write(
        f"Good-Turing: "
        f"{gt_perplexity:.4f}\n"
    )

    file.write(
        f"\nBest Add-k: k={best_add_k[0]}\n"
    )

    file.write(
        f"Best Add-k perplexity: "
        f"{best_add_k[1]:.4f}\n"
    )


print("\n" + "=" * 70)
print("TASK 3 IMPLEMENTATION COMPLETED")
print("=" * 70)

print(
    "\nResults saved to:"
    "\nresults\\task3_results.txt"
)