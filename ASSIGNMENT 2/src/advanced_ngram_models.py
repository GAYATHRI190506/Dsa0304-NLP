import re
import math
from collections import Counter, defaultdict


# ============================================================
# FILES
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
# PREPARE SENTENCES
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

unigram = Counter()
bigram = Counter()
trigram = Counter()

for sentence in train_data:

    for word in sentence:
        unigram[word] += 1

    for i in range(len(sentence) - 1):
        bigram[
            (sentence[i], sentence[i + 1])
        ] += 1

    for i in range(len(sentence) - 2):
        trigram[
            (
                sentence[i],
                sentence[i + 1],
                sentence[i + 2]
            )
        ] += 1


vocabulary = set(unigram)
V = len(vocabulary)

total_tokens = sum(unigram.values())


# ============================================================
# BASIC MLE PROBABILITIES
# ============================================================

def p_unigram(word):

    return unigram[word] / total_tokens


def p_bigram(w1, w2):

    denominator = unigram[w1]

    if denominator == 0:
        return 0.0

    return bigram[(w1, w2)] / denominator


def p_trigram(w1, w2, w3):

    denominator = bigram[(w1, w2)]

    if denominator == 0:
        return 0.0

    return trigram[
        (w1, w2, w3)
    ] / denominator


# ============================================================
# GOOD-TURING FREQUENCY TABLES
# ============================================================

bigram_Nc = Counter(
    bigram.values()
)

trigram_Nc = Counter(
    trigram.values()
)


def good_turing_count(
    count,
    frequency_table
):

    Nc = frequency_table.get(
        count,
        0
    )

    Nc1 = frequency_table.get(
        count + 1,
        0
    )

    if Nc == 0 or Nc1 == 0:
        return float(count)

    return (
        (count + 1)
        * Nc1
        / Nc
    )


# ============================================================
# KATZ BACKOFF
# ============================================================

def katz_bigram(w1, w2):

    count = bigram[
        (w1, w2)
    ]

    # Seen bigram
    if count > 0:

        adjusted = good_turing_count(
            count,
            bigram_Nc
        )

        denominator = unigram[w1]

        if denominator == 0:
            return 0.0

        return adjusted / denominator

    # Unseen bigram -> unigram backoff
    return 0.4 * p_unigram(w2)


def katz_trigram(w1, w2, w3):

    count = trigram[
        (w1, w2, w3)
    ]

    # Seen trigram
    if count > 0:

        adjusted = good_turing_count(
            count,
            trigram_Nc
        )

        denominator = bigram[
            (w1, w2)
        ]

        if denominator == 0:
            return katz_bigram(
                w2,
                w3
            )

        return adjusted / denominator

    # Trigram unseen -> bigram backoff
    return 0.4 * katz_bigram(
        w2,
        w3
    )


# ============================================================
# STUPID BACKOFF
# ============================================================

STUPID_ALPHA = 0.4


def stupid_backoff(w1, w2, w3):

    # Try trigram
    if trigram[
        (w1, w2, w3)
    ] > 0:

        return p_trigram(
            w1,
            w2,
            w3
        )

    # Back off to bigram
    if bigram[
        (w2, w3)
    ] > 0:

        return (
            STUPID_ALPHA
            * p_bigram(w2, w3)
        )

    # Back off to unigram
    return (
        STUPID_ALPHA
        * STUPID_ALPHA
        * p_unigram(w3)
    )


# ============================================================
# LINEAR INTERPOLATION
# ============================================================

def interpolated_probability(
    w1,
    w2,
    w3,
    lambdas
):

    lambda1, lambda2, lambda3 = (
        lambdas
    )

    return (
        lambda1 * p_unigram(w3)
        +
        lambda2 * p_bigram(w2, w3)
        +
        lambda3 * p_trigram(
            w1,
            w2,
            w3
        )
    )


# ============================================================
# DELETED INTERPOLATION
# ============================================================

def deleted_interpolation_training():

    counts = {
        "unigram": 0.0,
        "bigram": 0.0,
        "trigram": 0.0
    }

    total = 0.0

    for sentence in train_data:

        for i in range(2, len(sentence)):

            w1 = sentence[i - 2]
            w2 = sentence[i - 1]
            w3 = sentence[i]

            tri_count = trigram[
                (w1, w2, w3)
            ]

            bi_count = bigram[
                (w2, w3)
            ]

            uni_count = unigram[w3]

            tri_denominator = bigram[
                (w1, w2)
            ]

            bi_denominator = unigram[w2]

            # Leave-one-out estimates
            if tri_denominator > 1:

                p3 = (
                    (tri_count - 1)
                    /
                    (tri_denominator - 1)
                )

            else:
                p3 = 0.0

            if bi_denominator > 1:

                p2 = (
                    (bi_count - 1)
                    /
                    (bi_denominator - 1)
                )

            else:
                p2 = 0.0

            total_minus_one = (
                total_tokens - 1
            )

            if total_minus_one > 0:

                p1 = (
                    (uni_count - 1)
                    /
                    total_minus_one
                )

            else:
                p1 = 0.0

            values = [
                ("unigram", p1),
                ("bigram", p2),
                ("trigram", p3)
            ]

            best_model = max(
                values,
                key=lambda x: x[1]
            )[0]

            counts[best_model] += 1
            total += 1

    if total == 0:
        return (
            0.33,
            0.33,
            0.34
        )

    return (
        counts["unigram"] / total,
        counts["bigram"] / total,
        counts["trigram"] / total
    )


# ============================================================
# PERPLEXITY
# ============================================================

def perplexity(
    dataset,
    probability_function
):

    log_probability_sum = 0.0
    predictions = 0
    zero_count = 0

    for sentence in dataset:

        for i in range(2, len(sentence)):

            w1 = sentence[i - 2]
            w2 = sentence[i - 1]
            w3 = sentence[i]

            probability = probability_function(
                w1,
                w2,
                w3
            )

            if probability <= 0:

                zero_count += 1
                probability = 1e-12

            log_probability_sum += math.log(
                probability
            )

            predictions += 1

    if predictions == 0:
        return float("inf"), zero_count

    pp = math.exp(
        -log_probability_sum / predictions
    )

    return pp, zero_count


# ============================================================
# DELETED INTERPOLATION WEIGHTS
# ============================================================

deleted_lambdas = (
    deleted_interpolation_training()
)


# ============================================================
# SEARCH FOR BEST DEV LAMBDAS
# ============================================================

def find_best_lambdas():

    best = None

    for l1_int in range(0, 11):

        for l2_int in range(0, 11):

            l3_int = 10 - l1_int - l2_int

            if l3_int < 0:
                continue

            lambdas = (
                l1_int / 10,
                l2_int / 10,
                l3_int / 10
            )

            pp, zeros = perplexity(
                dev_data,
                lambda w1, w2, w3:
                    interpolated_probability(
                        w1,
                        w2,
                        w3,
                        lambdas
                    )
            )

            if best is None or pp < best[0]:

                best = (
                    pp,
                    lambdas,
                    zeros
                )

    return best


best_lambda_result = (
    find_best_lambdas()
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 75)
print("TASK 4 - ADVANCED N-GRAM LANGUAGE MODELS")
print("=" * 75)


print("\nCORPUS INFORMATION")
print("-" * 75)

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
    f"Unique bigrams : {len(bigram)}"
)

print(
    f"Unique trigrams : {len(trigram)}"
)


# ============================================================
# DELETED INTERPOLATION
# ============================================================

print("\nDELETED INTERPOLATION")
print("-" * 75)

print(
    f"λ1 (unigram) = "
    f"{deleted_lambdas[0]:.4f}"
)

print(
    f"λ2 (bigram)  = "
    f"{deleted_lambdas[1]:.4f}"
)

print(
    f"λ3 (trigram) = "
    f"{deleted_lambdas[2]:.4f}"
)

print(
    f"Sum of λ values = "
    f"{sum(deleted_lambdas):.4f}"
)


# ============================================================
# DEV-OPTIMIZED LINEAR INTERPOLATION
# ============================================================

print("\nLINEAR INTERPOLATION")
print("-" * 75)

best_pp = best_lambda_result[0]
best_lambdas = best_lambda_result[1]
best_zero = best_lambda_result[2]

print(
    f"Best λ1 = {best_lambdas[0]:.1f}"
)

print(
    f"Best λ2 = {best_lambdas[1]:.1f}"
)

print(
    f"Best λ3 = {best_lambdas[2]:.1f}"
)

print(
    f"Dev perplexity = {best_pp:.4f}"
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\nMODEL COMPARISON")
print("-" * 75)

models = []


katz_pp, katz_zero = perplexity(
    dev_data,
    katz_trigram
)

stupid_pp, stupid_zero = perplexity(
    dev_data,
    stupid_backoff
)

linear_pp, linear_zero = perplexity(
    dev_data,
    lambda w1, w2, w3:
        interpolated_probability(
            w1,
            w2,
            w3,
            best_lambdas
        )
)

deleted_pp, deleted_zero = perplexity(
    dev_data,
    lambda w1, w2, w3:
        interpolated_probability(
            w1,
            w2,
            w3,
            deleted_lambdas
        )
)


models.append(
    (
        "Katz Backoff",
        katz_pp,
        katz_zero
    )
)

models.append(
    (
        "Stupid Backoff",
        stupid_pp,
        stupid_zero
    )
)

models.append(
    (
        "Linear Interpolation",
        linear_pp,
        linear_zero
    )
)

models.append(
    (
        "Deleted Interpolation",
        deleted_pp,
        deleted_zero
    )
)


print(
    f"{'Model':<25}"
    f"{'Perplexity':>15}"
    f"{'Zero events':>15}"
)

print("-" * 55)

for name, pp, zeros in models:

    print(
        f"{name:<25}"
        f"{pp:>15.4f}"
        f"{zeros:>15}"
    )


# ============================================================
# BEST MODEL
# ============================================================

best_model = min(
    models,
    key=lambda x: x[1]
)

print("\nBEST MODEL")
print("-" * 75)

print(
    f"Model      : {best_model[0]}"
)

print(
    f"Perplexity : {best_model[1]:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "results/task4_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 4 - ADVANCED N-GRAM LANGUAGE MODELS\n"
    )

    file.write("=" * 75 + "\n\n")

    file.write(
        "DELETED INTERPOLATION WEIGHTS\n"
    )

    file.write(
        f"Lambda 1: {deleted_lambdas[0]:.6f}\n"
    )

    file.write(
        f"Lambda 2: {deleted_lambdas[1]:.6f}\n"
    )

    file.write(
        f"Lambda 3: {deleted_lambdas[2]:.6f}\n\n"
    )

    file.write(
        "DEV-OPTIMIZED LINEAR INTERPOLATION\n"
    )

    file.write(
        f"Lambda 1: {best_lambdas[0]:.2f}\n"
    )

    file.write(
        f"Lambda 2: {best_lambdas[1]:.2f}\n"
    )

    file.write(
        f"Lambda 3: {best_lambdas[2]:.2f}\n\n"
    )

    file.write(
        "MODEL COMPARISON\n"
    )

    for name, pp, zeros in models:

        file.write(
            f"{name}: "
            f"Perplexity={pp:.6f}, "
            f"Zero events={zeros}\n"
        )

    file.write(
        f"\nBest model: {best_model[0]}\n"
    )

    file.write(
        f"Best perplexity: "
        f"{best_model[1]:.6f}\n"
    )


print("\n" + "=" * 75)
print("TASK 4 IMPLEMENTATION COMPLETED")
print("=" * 75)

print(
    "\nResults saved to:"
    "\nresults\\task4_results.txt"
)