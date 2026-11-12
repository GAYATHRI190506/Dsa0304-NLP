import re
import random
import math
from collections import Counter


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = "data/train.txt"

# Fixed random seed for reproducibility
random.seed(42)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert a sentence into lowercase word tokens.
    Sentence boundary markers are added separately.
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-z]+(?:'[a-z]+)?|\d+(?::\d+)?|[^\w\s]",
        text
    )

    return tokens


# ============================================================
# READ TRAINING CORPUS
# ============================================================

def read_sentences(filename):

    with open(filename, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


sentences = read_sentences(TRAIN_FILE)


# ============================================================
# ADD SENTENCE BOUNDARIES
# ============================================================

tokenized_sentences = []

for sentence in sentences:

    tokens = tokenize(sentence)

    # Add beginning and ending markers
    tokens = ["<s>"] + tokens + ["</s>"]

    tokenized_sentences.append(tokens)


# ============================================================
# CREATE N-GRAM COUNTS
# ============================================================

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for tokens in tokenized_sentences:

    # Unigrams
    for token in tokens:
        unigram_counts[token] += 1

    # Bigrams
    for i in range(len(tokens) - 1):
        bigram = (
            tokens[i],
            tokens[i + 1]
        )
        bigram_counts[bigram] += 1

    # Trigrams
    for i in range(len(tokens) - 2):
        trigram = (
            tokens[i],
            tokens[i + 1],
            tokens[i + 2]
        )
        trigram_counts[trigram] += 1


# ============================================================
# TOTAL COUNTS
# ============================================================

total_unigrams = sum(unigram_counts.values())

vocabulary = set(unigram_counts.keys())

vocab_size = len(vocabulary)


# ============================================================
# MLE PROBABILITIES
# ============================================================

def unigram_probability(word):
    """
    MLE unigram probability:

    P(w) = Count(w) / Total number of tokens
    """

    return unigram_counts[word] / total_unigrams


def bigram_probability(previous_word, word):
    """
    MLE bigram probability:

    P(w_i | w_(i-1))
    =
    Count(w_(i-1), w_i)
    /
    Count(w_(i-1))
    """

    denominator = unigram_counts[previous_word]

    if denominator == 0:
        return 0.0

    numerator = bigram_counts[
        (previous_word, word)
    ]

    return numerator / denominator


def trigram_probability(
    word1,
    word2,
    word3
):
    """
    MLE trigram probability:

    P(w_i | w_(i-2), w_(i-1))
    =
    Count(w_(i-2), w_(i-1), w_i)
    /
    Count(w_(i-2), w_(i-1))
    """

    denominator = bigram_counts[
        (word1, word2)
    ]

    if denominator == 0:
        return 0.0

    numerator = trigram_counts[
        (word1, word2, word3)
    ]

    return numerator / denominator


# ============================================================
# SENTENCE PROBABILITY
# ============================================================

def sentence_probability_unigram(tokens):

    probability = 1.0

    for word in tokens:

        p = unigram_probability(word)

        if p == 0:
            return 0.0

        probability *= p

    return probability


def sentence_probability_bigram(tokens):

    probability = 1.0

    for i in range(1, len(tokens)):

        p = bigram_probability(
            tokens[i - 1],
            tokens[i]
        )

        if p == 0:
            return 0.0

        probability *= p

    return probability


def sentence_probability_trigram(tokens):

    probability = 1.0

    for i in range(2, len(tokens)):

        p = trigram_probability(
            tokens[i - 2],
            tokens[i - 1],
            tokens[i]
        )

        if p == 0:
            return 0.0

        probability *= p

    return probability


# ============================================================
# CHAIN RULE EXAMPLE
# ============================================================

print("=" * 70)
print("TASK 2 - N-GRAM LANGUAGE MODELS")
print("=" * 70)

print("\nCHAIN RULE")
print("-" * 70)

print(
    "P(w1,w2,...,wn) = "
    "P(w1) P(w2|w1) P(w3|w1,w2) ..."
)

print(
    "\nThe chain rule expresses the probability of a complete "
    "sentence as a product of conditional probabilities."
)


# ============================================================
# MARKOV ASSUMPTIONS
# ============================================================

print("\nMARKOV ASSUMPTIONS")
print("-" * 70)

print(
    "Bigram model : "
    "P(w_i | w_1,...,w_(i-1)) ≈ P(w_i | w_(i-1))"
)

print(
    "Trigram model: "
    "P(w_i | w_1,...,w_(i-1)) ≈ "
    "P(w_i | w_(i-2), w_(i-1))"
)


# ============================================================
# CORPUS STATISTICS
# ============================================================

print("\nCORPUS STATISTICS")
print("-" * 70)

print(f"Training sentences       : {len(sentences)}")
print(f"Vocabulary size          : {vocab_size}")
print(f"Unigram observations     : {total_unigrams}")
print(f"Unique bigrams           : {len(bigram_counts)}")
print(f"Unique trigrams          : {len(trigram_counts)}")


# ============================================================
# HAND-COMPUTED PROBABILITY EXAMPLES
# ============================================================

print("\nHAND-COMPUTED MLE EXAMPLES")
print("-" * 70)


def show_unigram(word):

    count = unigram_counts[word]

    probability = unigram_probability(word)

    print(
        f"P({word}) = "
        f"Count({word}) / Total = "
        f"{count} / {total_unigrams} = "
        f"{probability:.8f}"
    )


def show_bigram(word1, word2):

    count = bigram_counts[(word1, word2)]

    denominator = unigram_counts[word1]

    probability = bigram_probability(
        word1,
        word2
    )

    print(
        f"P({word2}|{word1}) = "
        f"Count({word1},{word2}) / Count({word1}) = "
        f"{count} / {denominator} = "
        f"{probability:.8f}"
    )


def show_trigram(word1, word2, word3):

    count = trigram_counts[
        (word1, word2, word3)
    ]

    denominator = bigram_counts[
        (word1, word2)
    ]

    probability = trigram_probability(
        word1,
        word2,
        word3
    )

    print(
        f"P({word3}|{word1},{word2}) = "
        f"Count({word1},{word2},{word3}) / "
        f"Count({word1},{word2}) = "
        f"{count} / {denominator} = "
        f"{probability:.8f}"
    )


show_unigram("train")

show_bigram(
    "the",
    "train"
)

show_trigram(
    "the",
    "train",
    "from"
)


# ============================================================
# ZERO PROBABILITY EXAMPLE
# ============================================================

print("\nZERO-PROBABILITY EXAMPLE")
print("-" * 70)

zero_bigram = ("train", "banana")

zero_probability = bigram_probability(
    zero_bigram[0],
    zero_bigram[1]
)

print(
    f"P({zero_bigram[1]}|{zero_bigram[0]}) "
    f"= {zero_probability}"
)

print(
    "This demonstrates the zero-frequency problem "
    "of unsmoothed MLE."
)


# ============================================================
# SPARSITY ANALYSIS
# ============================================================

print("\nSPARSITY ANALYSIS")
print("-" * 70)

possible_bigrams = vocab_size ** 2
possible_trigrams = vocab_size ** 3

observed_bigrams = len(bigram_counts)
observed_trigrams = len(trigram_counts)

bigram_sparsity = (
    1 -
    observed_bigrams / possible_bigrams
) * 100

trigram_sparsity = (
    1 -
    observed_trigrams / possible_trigrams
) * 100

print(
    f"Possible bigrams : {possible_bigrams:,}"
)

print(
    f"Observed bigrams : {observed_bigrams:,}"
)

print(
    f"Bigram sparsity  : {bigram_sparsity:.4f}%"
)

print()

print(
    f"Possible trigrams: {possible_trigrams:,}"
)

print(
    f"Observed trigrams: {observed_trigrams:,}"
)

print(
    f"Trigram sparsity : {trigram_sparsity:.4f}%"
)


# ============================================================
# MOST FREQUENT N-GRAMS
# ============================================================

print("\nTOP 15 BIGRAMS")
print("-" * 70)

for rank, (ngram, count) in enumerate(
    bigram_counts.most_common(15),
    start=1
):

    print(
        f"{rank:2d}. "
        f"{ngram} -> {count}"
    )


print("\nTOP 15 TRIGRAMS")
print("-" * 70)

for rank, (ngram, count) in enumerate(
    trigram_counts.most_common(15),
    start=1
):

    print(
        f"{rank:2d}. "
        f"{ngram} -> {count}"
    )


# ============================================================
# SENTENCE PROBABILITY EXAMPLE
# ============================================================

example_sentence = [
    "<s>",
    "what",
    "is",
    "the",
    "fare",
    "</s>"
]

print("\nSENTENCE PROBABILITY EXAMPLE")
print("-" * 70)

print(
    "Sentence:",
    " ".join(example_sentence)
)

uni_prob = sentence_probability_unigram(
    example_sentence
)

bi_prob = sentence_probability_bigram(
    example_sentence
)

tri_prob = sentence_probability_trigram(
    example_sentence
)

print(
    f"Unigram probability : {uni_prob:.12e}"
)

print(
    f"Bigram probability  : {bi_prob:.12e}"
)

print(
    f"Trigram probability : {tri_prob:.12e}"
)


# ============================================================
# SENTENCE GENERATION
# ============================================================

def generate_sentence(
    model="bigram",
    max_length=15
):

    generated = ["<s>"]

    for _ in range(max_length):

        previous = generated[-1]

        candidates = []

        if model == "bigram":

            for (w1, w2), count in bigram_counts.items():

                if w1 == previous:
                    candidates.append(
                        (w2, count)
                    )

        elif model == "trigram":

            if len(generated) >= 2:

                previous2 = generated[-2]

                for (w1, w2, w3), count in trigram_counts.items():

                    if (
                        w1 == previous2
                        and w2 == previous
                    ):
                        candidates.append(
                            (w3, count)
                        )

        if not candidates:
            break

        words = [
            item[0]
            for item in candidates
        ]

        frequencies = [
            item[1]
            for item in candidates
        ]

        next_word = random.choices(
            words,
            weights=frequencies,
            k=1
        )[0]

        generated.append(next_word)

        if next_word == "</s>":
            break

    return generated


print("\nGENERATED SENTENCES")
print("-" * 70)

for i in range(5):

    sentence = generate_sentence(
        model="bigram"
    )

    print(
        f"Bigram {i + 1}: "
        f"{' '.join(sentence)}"
    )

print()

for i in range(5):

    sentence = generate_sentence(
        model="trigram"
    )

    print(
        f"Trigram {i + 1}: "
        f"{' '.join(sentence)}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "results/task2_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TASK 2 - N-GRAM LANGUAGE MODELS\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(
        f"Training sentences: {len(sentences)}\n"
    )

    file.write(
        f"Vocabulary size: {vocab_size}\n"
    )

    file.write(
        f"Unigram observations: {total_unigrams}\n"
    )

    file.write(
        f"Unique bigrams: {observed_bigrams}\n"
    )

    file.write(
        f"Unique trigrams: {observed_trigrams}\n"
    )

    file.write(
        f"Bigram sparsity: {bigram_sparsity:.4f}%\n"
    )

    file.write(
        f"Trigram sparsity: {trigram_sparsity:.4f}%\n"
    )

    file.write("\n")

    file.write(
        "Unigram example:\n"
    )

    file.write(
        f"P(train) = "
        f"{unigram_probability('train'):.8f}\n"
    )

    file.write(
        "\nBigram example:\n"
    )

    file.write(
        f"P(train|the) = "
        f"{bigram_probability('the', 'train'):.8f}\n"
    )

    file.write(
        "\nTrigram example:\n"
    )

    file.write(
        f"P(from|the,train) = "
        f"{trigram_probability('the', 'train', 'from'):.8f}\n"
    )

print("\n" + "=" * 70)
print("TASK 2 IMPLEMENTATION COMPLETED")
print("=" * 70)

print(
    "\nResults saved to:"
    "\nresults\\task2_results.txt"
)