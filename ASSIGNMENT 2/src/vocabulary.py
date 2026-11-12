import re
import os
from collections import Counter
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

TRAIN_FILE = "data/train.txt"
DEV_FILE = "data/dev.txt"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# LEMMATIZER
# ---------------------------------------------------------

lemmatizer = WordNetLemmatizer()


# ---------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------

def tokenize(text):
    """
    Tokenization decisions:
    1. Convert all text to lowercase.
    2. Keep words, numbers and apostrophes as tokens.
    3. Punctuation marks are treated as separate tokens.
    """

    text = text.lower()

    # Words, numbers, times and punctuation
    tokens = re.findall(
        r"[a-z]+(?:'[a-z]+)?|\d+(?::\d+)?|[^\w\s]",
        text
    )

    return tokens


# ---------------------------------------------------------
# LEMMATIZATION
# ---------------------------------------------------------

def lemmatize_tokens(tokens):
    """
    Apply WordNet lemmatization.
    The default noun POS is used because the corpus
    does not contain manually supplied POS information.
    """

    return [lemmatizer.lemmatize(token) for token in tokens]


# ---------------------------------------------------------
# READ CORPUS
# ---------------------------------------------------------

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


train_sentences = read_file(TRAIN_FILE)
dev_sentences = read_file(DEV_FILE)


# ---------------------------------------------------------
# TOKENIZE TRAINING DATA
# ---------------------------------------------------------

train_tokens = []

for sentence in train_sentences:
    train_tokens.extend(tokenize(sentence))


# ---------------------------------------------------------
# TOKEN COUNTS
# ---------------------------------------------------------

token_count = len(train_tokens)

word_form_counter = Counter(train_tokens)
word_form_types = len(word_form_counter)

lemma_tokens = lemmatize_tokens(train_tokens)
lemma_counter = Counter(lemma_tokens)
lemma_types = len(lemma_counter)


# ---------------------------------------------------------
# PRINT BASIC COUNTS
# ---------------------------------------------------------

print("=" * 60)
print("TASK 1 - COUNTING WORDS AND BUILDING THE VOCABULARY")
print("=" * 60)

print("\nTRAINING CORPUS")
print("-" * 60)

print(f"Number of sentences       : {len(train_sentences)}")
print(f"Number of word tokens     : {token_count}")
print(f"Number of word-form types : {word_form_types}")
print(f"Number of lemma types     : {lemma_types}")


# ---------------------------------------------------------
# TOKENIZATION DECISIONS
# ---------------------------------------------------------

print("\nTOKENIZATION DECISIONS")
print("-" * 60)

print("1. All text is converted to lowercase.")
print("2. Words are separated into individual tokens.")
print("3. Numeric expressions such as 7:30 are kept as tokens.")
print("4. Punctuation is treated as separate tokens.")
print("5. Apostrophes inside words are retained.")


# ---------------------------------------------------------
# EFFECT OF TOKENIZATION DECISIONS
# ---------------------------------------------------------

def simple_whitespace_tokenize(text):
    return text.lower().split()


def punctuation_removed_tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?::\d+)?", text)


whitespace_tokens = []

punctuation_removed_tokens = []

for sentence in train_sentences:
    whitespace_tokens.extend(simple_whitespace_tokenize(sentence))
    punctuation_removed_tokens.extend(
        punctuation_removed_tokenize(sentence)
    )


print("\nEFFECT OF TOKENIZATION DECISIONS")
print("-" * 60)

print(
    f"Whitespace vocabulary size       : "
    f"{len(set(whitespace_tokens))}"
)

print(
    f"Punctuation-aware vocabulary size: "
    f"{len(set(punctuation_removed_tokens))}"
)

print(
    f"Final tokenizer vocabulary size  : "
    f"{word_form_types}"
)


# ---------------------------------------------------------
# ZIPF'S LAW
# ---------------------------------------------------------

frequency_sorted = sorted(
    word_form_counter.values(),
    reverse=True
)

ranks = list(range(1, len(frequency_sorted) + 1))

plt.figure(figsize=(10, 6))

plt.loglog(
    ranks,
    frequency_sorted,
    marker=".",
    linestyle="none"
)

plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf's Law: Word Rank vs Frequency")
plt.grid(True)

zipf_path = os.path.join(
    RESULTS_DIR,
    "zipf_plot.png"
)

plt.savefig(zipf_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nZIPF PLOT")
print("-" * 60)
print(f"Saved to: {zipf_path}")


# ---------------------------------------------------------
# VOCABULARY THRESHOLDS
# ---------------------------------------------------------

print("\nVOCABULARY THRESHOLD ANALYSIS")
print("-" * 60)

thresholds = [1, 2, 3, 5]

for threshold in thresholds:

    vocabulary = {
        word
        for word, count in word_form_counter.items()
        if count >= threshold
    }

    rare_tokens = sum(
        count
        for word, count in word_form_counter.items()
        if count < threshold
    )

    unk_percentage = (
        rare_tokens / token_count
    ) * 100

    print(
        f"Threshold >= {threshold}: "
        f"Vocabulary = {len(vocabulary):5d}, "
        f"Tokens mapped to <UNK> = "
        f"{rare_tokens:6d} "
        f"({unk_percentage:.2f}%)"
    )


# ---------------------------------------------------------
# DEVELOPMENT SET OOV RATE
# ---------------------------------------------------------

dev_tokens = []

for sentence in dev_sentences:
    dev_tokens.extend(tokenize(sentence))


train_vocabulary = set(word_form_counter.keys())

oov_tokens = [
    token
    for token in dev_tokens
    if token not in train_vocabulary
]

oov_rate = (
    len(oov_tokens) / len(dev_tokens)
) * 100


print("\nDEVELOPMENT SET OOV ANALYSIS")
print("-" * 60)

print(f"Development tokens : {len(dev_tokens)}")
print(f"OOV tokens         : {len(oov_tokens)}")
print(f"OOV rate           : {oov_rate:.2f}%")


# ---------------------------------------------------------
# TOP 20 WORDS
# ---------------------------------------------------------

print("\nTOP 20 MOST FREQUENT WORD FORMS")
print("-" * 60)

for rank, (word, frequency) in enumerate(
    word_form_counter.most_common(20),
    start=1
):
    print(f"{rank:2d}. {word:<20} {frequency}")


# ---------------------------------------------------------
# SAVE RESULTS TO FILE
# ---------------------------------------------------------

results_file = os.path.join(
    RESULTS_DIR,
    "task1_results.txt"
)

with open(results_file, "w", encoding="utf-8") as file:

    file.write("TASK 1 - COUNTING WORDS AND BUILDING THE VOCABULARY\n")
    file.write("=" * 60 + "\n\n")

    file.write("TRAINING CORPUS\n")
    file.write(f"Sentences: {len(train_sentences)}\n")
    file.write(f"Word tokens: {token_count}\n")
    file.write(f"Word-form types: {word_form_types}\n")
    file.write(f"Lemma types: {lemma_types}\n\n")

    file.write("TOKENIZATION DECISIONS\n")
    file.write("1. Lowercase normalization\n")
    file.write("2. Words separated into tokens\n")
    file.write("3. Numbers and times retained\n")
    file.write("4. Punctuation handled separately\n")
    file.write("5. Apostrophes retained inside words\n\n")

    file.write("VOCABULARY THRESHOLDS\n")

    for threshold in thresholds:

        vocabulary = {
            word
            for word, count in word_form_counter.items()
            if count >= threshold
        }

        rare_tokens = sum(
            count
            for word, count in word_form_counter.items()
            if count < threshold
        )

        unk_percentage = (
            rare_tokens / token_count
        ) * 100

        file.write(
            f"Threshold {threshold}: "
            f"Vocabulary={len(vocabulary)}, "
            f"<UNK> tokens={rare_tokens}, "
            f"percentage={unk_percentage:.2f}%\n"
        )

    file.write("\nDEVELOPMENT OOV\n")
    file.write(f"Development tokens: {len(dev_tokens)}\n")
    file.write(f"OOV tokens: {len(oov_tokens)}\n")
    file.write(f"OOV rate: {oov_rate:.2f}%\n")


print("\nRESULT FILE")
print("-" * 60)
print(f"Saved to: {results_file}")

print("\nTASK 1 COMPLETED SUCCESSFULLY.")