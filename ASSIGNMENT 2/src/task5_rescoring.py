import re
import math
import random
from collections import Counter


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = "data/train.txt"
DEV_FILE = "data/dev.txt"

RESULT_FILE = "results/task5_results.txt"
HYPOTHESIS_FILE = "results/recognizer_hypotheses.txt"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

NUM_HYPOTHESIS_SETS = 200
NUM_HYPOTHESES = 5


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
# READ FILE
# ============================================================

def read_sentences(filename):

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        return [
            line.strip()
            for line in file
            if line.strip()
        ]


train_sentences = read_sentences(TRAIN_FILE)
dev_sentences = read_sentences(DEV_FILE)


# ============================================================
# TOKENIZED DATA
# ============================================================

train_data = [
    ["<s>"] + tokenize(sentence) + ["</s>"]
    for sentence in train_sentences
]

dev_data = [
    ["<s>"] + tokenize(sentence) + ["</s>"]
    for sentence in dev_sentences
]


# ============================================================
# N-GRAM COUNTS
# ============================================================

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for sentence in train_data:

    for word in sentence:
        unigram_counts[word] += 1

    for i in range(len(sentence) - 1):

        bigram_counts[
            (sentence[i], sentence[i + 1])
        ] += 1

    for i in range(len(sentence) - 2):

        trigram_counts[
            (
                sentence[i],
                sentence[i + 1],
                sentence[i + 2]
            )
        ] += 1


VOCABULARY = set(unigram_counts)

VOCAB_SIZE = len(VOCABULARY)

TOTAL_TOKENS = sum(
    unigram_counts.values()
)


# ============================================================
# ADD-K TRIGRAM LANGUAGE MODEL
# ============================================================

K = 0.1


def trigram_probability(
    w1,
    w2,
    w3
):

    numerator = (
        trigram_counts[
            (w1, w2, w3)
        ]
        + K
    )

    denominator = (
        bigram_counts[
            (w1, w2)
        ]
        + K * VOCAB_SIZE
    )

    return numerator / denominator


# ============================================================
# SENTENCE LOG PROBABILITY
# ============================================================

def sentence_log_probability(words):

    tokens = (
        ["<s>", "<s>"]
        + words
        + ["</s>"]
    )

    total_log_probability = 0.0

    for i in range(2, len(tokens)):

        probability = trigram_probability(
            tokens[i - 2],
            tokens[i - 1],
            tokens[i]
        )

        total_log_probability += math.log(
            probability
        )

    return total_log_probability


# ============================================================
# PERPLEXITY
# ============================================================

def calculate_perplexity(dataset):

    total_log_probability = 0.0
    total_predictions = 0

    for sentence in dataset:

        tokens = sentence

        for i in range(2, len(tokens)):

            probability = trigram_probability(
                tokens[i - 2],
                tokens[i - 1],
                tokens[i]
            )

            total_log_probability += math.log(
                probability
            )

            total_predictions += 1

    return math.exp(
        -total_log_probability
        / total_predictions
    )


# ============================================================
# ENTROPY / CROSS-ENTROPY
# ============================================================

def calculate_cross_entropy(
    dataset
):

    total_log_probability = 0.0
    total_predictions = 0

    for sentence in dataset:

        for i in range(2, len(sentence)):

            probability = trigram_probability(
                sentence[i - 2],
                sentence[i - 1],
                sentence[i]
            )

            total_log_probability += math.log2(
                probability
            )

            total_predictions += 1

    return (
        -total_log_probability
        / total_predictions
    )


def calculate_entropy(
    dataset
):

    """
    Empirical unigram entropy:

    H(X) = -sum p(x) log2 p(x)
    """

    counts = Counter()

    total = 0

    for sentence in dataset:

        for word in sentence:

            counts[word] += 1
            total += 1

    entropy = 0.0

    for count in counts.values():

        probability = count / total

        entropy -= (
            probability
            * math.log2(probability)
        )

    return entropy


# ============================================================
# WORD ERROR RATE
# ============================================================

def edit_distance(
    reference,
    hypothesis
):

    n = len(reference)
    m = len(hypothesis)

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if reference[i - 1] == hypothesis[j - 1]:

                dp[i][j] = dp[i - 1][j - 1]

            else:

                substitution = (
                    dp[i - 1][j - 1] + 1
                )

                deletion = (
                    dp[i - 1][j] + 1
                )

                insertion = (
                    dp[i][j - 1] + 1
                )

                dp[i][j] = min(
                    substitution,
                    deletion,
                    insertion
                )

    return dp[n][m]


def word_error_rate(
    reference,
    hypothesis
):

    reference_words = tokenize(reference)
    hypothesis_words = tokenize(hypothesis)

    if len(reference_words) == 0:
        return 0.0

    distance = edit_distance(
        reference_words,
        hypothesis_words
    )

    return distance / len(reference_words)


# ============================================================
# CREATE HYPOTHESIS VARIANTS
# ============================================================

def substitute_word(words):

    result = words.copy()

    if len(result) == 0:
        return result

    index = random.randrange(
        len(result)
    )

    replacement_words = [
        "train",
        "bus",
        "ticket",
        "route",
        "fare",
        "station",
        "today",
        "tomorrow",
        "travel",
        "platform"
    ]

    result[index] = random.choice(
        replacement_words
    )

    return result


def delete_word(words):

    result = words.copy()

    if len(result) <= 2:
        return result

    index = random.randrange(
        len(result)
    )

    del result[index]

    return result


def insert_word(words):

    result = words.copy()

    insertion_words = [
        "train",
        "bus",
        "today",
        "please",
        "available",
        "ticket",
        "route"
    ]

    index = random.randrange(
        len(result) + 1
    )

    result.insert(
        index,
        random.choice(insertion_words)
    )

    return result


def swap_words(words):

    result = words.copy()

    if len(result) < 2:
        return result

    index = random.randrange(
        len(result) - 1
    )

    result[index], result[index + 1] = (
        result[index + 1],
        result[index]
    )

    return result


# ============================================================
# CREATE 200 RECOGNIZER HYPOTHESIS SETS
# ============================================================

hypothesis_sets = []

selected_references = dev_sentences[
    :NUM_HYPOTHESIS_SETS
]

for set_id, reference in enumerate(
    selected_references,
    start=1
):

    reference_words = tokenize(
        reference
    )

    hypotheses = []

    # --------------------------------------------------------
    # Hypothesis 1
    # Intentionally noisy recognizer output
    # --------------------------------------------------------

    h1 = substitute_word(
        reference_words
    )

    # --------------------------------------------------------
    # Hypothesis 2
    # Deletion error
    # --------------------------------------------------------

    h2 = delete_word(
        reference_words
    )

    # --------------------------------------------------------
    # Hypothesis 3
    # Insertion error
    # --------------------------------------------------------

    h3 = insert_word(
        reference_words
    )

    # --------------------------------------------------------
    # Hypothesis 4
    # Word order error
    # --------------------------------------------------------

    h4 = swap_words(
        reference_words
    )

    # --------------------------------------------------------
    # Hypothesis 5
    # Another substitution
    # --------------------------------------------------------

    h5 = substitute_word(
        reference_words
    )

    hypotheses = [
        h1,
        h2,
        h3,
        h4,
        h5
    ]

    hypothesis_sets.append(
        {
            "id": set_id,
            "reference": reference,
            "hypotheses": [
                " ".join(h)
                for h in hypotheses
            ]
        }
    )


# ============================================================
# RESCORE HYPOTHESES
# ============================================================

before_wer_values = []
after_wer_values = []

correct_before = 0
correct_after = 0


for item in hypothesis_sets:

    reference = item["reference"]

    hypotheses = item["hypotheses"]

    # --------------------------------------------------------
    # BEFORE LM RESCORING
    # Baseline = recognizer's first hypothesis
    # --------------------------------------------------------

    baseline = hypotheses[0]

    baseline_wer = word_error_rate(
        reference,
        baseline
    )

    before_wer_values.append(
        baseline_wer
    )

    # --------------------------------------------------------
    # LM RESCORING
    # --------------------------------------------------------

    scores = []

    for hypothesis in hypotheses:

        words = tokenize(
            hypothesis
        )

        score = sentence_log_probability(
            words
        )

        scores.append(score)

    best_index = max(
        range(len(scores)),
        key=lambda i: scores[i]
    )

    rescored_hypothesis = (
        hypotheses[best_index]
    )

    rescored_wer = word_error_rate(
        reference,
        rescored_hypothesis
    )

    after_wer_values.append(
        rescored_wer
    )

    if baseline_wer == 0:
        correct_before += 1

    if rescored_wer == 0:
        correct_after += 1


# ============================================================
# AVERAGE WER
# ============================================================

average_before = (
    sum(before_wer_values)
    / len(before_wer_values)
)

average_after = (
    sum(after_wer_values)
    / len(after_wer_values)
)


relative_improvement = 0.0

if average_before > 0:

    relative_improvement = (
        (average_before - average_after)
        / average_before
    ) * 100


# ============================================================
# CALCULATE ENTROPY / CROSS-ENTROPY / PERPLEXITY
# ============================================================

entropy = calculate_entropy(
    dev_data
)

cross_entropy = calculate_cross_entropy(
    dev_data
)

perplexity_value = calculate_perplexity(
    dev_data
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 75)
print("TASK 5 - ENTROPY, PERPLEXITY AND RECOGNIZER RESCORING")
print("=" * 75)


print("\nLANGUAGE MODEL INFORMATION")
print("-" * 75)

print(
    f"Training sentences : "
    f"{len(train_data)}"
)

print(
    f"Development sentences : "
    f"{len(dev_data)}"
)

print(
    f"Vocabulary size : "
    f"{VOCAB_SIZE}"
)

print(
    f"Trigram vocabulary : "
    f"{len(trigram_counts)}"
)

print(
    f"Add-k value : "
    f"{K}"
)


# ============================================================
# ENTROPY RESULTS
# ============================================================

print("\nENTROPY MEASURES")
print("-" * 75)

print(
    f"Unigram entropy : "
    f"{entropy:.6f} bits/token"
)

print(
    f"Cross-entropy : "
    f"{cross_entropy:.6f} bits/token"
)

print(
    f"Perplexity : "
    f"{perplexity_value:.6f}"
)


# ============================================================
# RECOGNIZER RESULTS
# ============================================================

print("\nRECOGNIZER HYPOTHESIS RESCORING")
print("-" * 75)

print(
    f"Number of hypothesis sets : "
    f"{len(hypothesis_sets)}"
)

print(
    f"Hypotheses per set : "
    f"{NUM_HYPOTHESES}"
)

print(
    f"Total hypotheses evaluated : "
    f"{len(hypothesis_sets) * NUM_HYPOTHESES}"
)

print()

print(
    f"Average WER before rescoring : "
    f"{average_before:.4f} "
    f"({average_before * 100:.2f}%)"
)

print(
    f"Average WER after rescoring  : "
    f"{average_after:.4f} "
    f"({average_after * 100:.2f}%)"
)

print(
    f"Relative WER improvement : "
    f"{relative_improvement:.2f}%"
)

print()

print(
    f"Perfect hypotheses before : "
    f"{correct_before}/{NUM_HYPOTHESIS_SETS}"
)

print(
    f"Perfect hypotheses after  : "
    f"{correct_after}/{NUM_HYPOTHESIS_SETS}"
)


# ============================================================
# SAVE HYPOTHESIS DATA
# ============================================================

with open(
    HYPOTHESIS_FILE,
    "w",
    encoding="utf-8"
) as file:

    for item in hypothesis_sets:

        file.write(
            f"SET {item['id']}\n"
        )

        file.write(
            f"REFERENCE: "
            f"{item['reference']}\n"
        )

        for index, hypothesis in enumerate(
            item["hypotheses"],
            start=1
        ):

            file.write(
                f"H{index}: "
                f"{hypothesis}\n"
            )

        file.write("\n")


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 5 - ENTROPY, PERPLEXITY AND "
        "RECOGNIZER RESCORING\n"
    )

    file.write("=" * 75 + "\n\n")

    file.write(
        f"Random seed: {RANDOM_SEED}\n"
    )

    file.write(
        f"Number of hypothesis sets: "
        f"{len(hypothesis_sets)}\n"
    )

    file.write(
        f"Hypotheses per set: "
        f"{NUM_HYPOTHESES}\n\n"
    )

    file.write(
        "LANGUAGE MODEL\n"
    )

    file.write(
        f"Add-k: {K}\n"
    )

    file.write(
        f"Vocabulary: {VOCAB_SIZE}\n"
    )

    file.write(
        f"Entropy: {entropy:.6f}\n"
    )

    file.write(
        f"Cross-entropy: "
        f"{cross_entropy:.6f}\n"
    )

    file.write(
        f"Perplexity: "
        f"{perplexity_value:.6f}\n\n"
    )

    file.write(
        "WER COMPARISON\n"
    )

    file.write(
        f"Before rescoring: "
        f"{average_before:.6f}\n"
    )

    file.write(
        f"After rescoring: "
        f"{average_after:.6f}\n"
    )

    file.write(
        f"Relative improvement: "
        f"{relative_improvement:.2f}%\n"
    )

    file.write(
        f"Perfect before: "
        f"{correct_before}/{NUM_HYPOTHESIS_SETS}\n"
    )

    file.write(
        f"Perfect after: "
        f"{correct_after}/{NUM_HYPOTHESIS_SETS}\n"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\nFILES CREATED")
print("-" * 75)

print(
    f"1. {RESULT_FILE}"
)

print(
    f"2. {HYPOTHESIS_FILE}"
)

print("\n" + "=" * 75)
print("TASK 5 IMPLEMENTATION COMPLETED")
print("=" * 75)