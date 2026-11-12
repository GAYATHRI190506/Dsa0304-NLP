import re
import math
import time
from collections import Counter, defaultdict
from itertools import product

TRAIN_FILE = "data/train.txt"
TEST_FILE = "data/test.txt"

# ------------------------------------------------------------
# TAG SET
# ------------------------------------------------------------

TAGS = [
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV",
    "PRON", "DET", "PREP", "CONJ", "NUM", "WHPRON",
    "WHADV", "TO", "PART", "POSS", "PUNCT", "INTERJ",
    "EXIST", "SYM", "FOREIGN", "UNKNOWN"
]

# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------

def tokenize(text):
    return re.findall(
        r"\d{1,2}:\d{2}|\d+(?:\.\d+)?|[a-zA-Z]+(?:'[a-zA-Z]+)?|[^\w\s]",
        text.lower()
    )


# ------------------------------------------------------------
# LEXICAL INFORMATION
# ------------------------------------------------------------

CITIES = {
    "delhi", "mumbai", "chennai", "hyderabad", "bengaluru",
    "bangalore", "pune", "kolkata", "jaipur", "ahmedabad",
    "vijayawada", "guntur", "tirupati", "madurai", "salem",
    "trichy", "coimbatore", "mysuru", "warangal", "nellore",
    "vizag", "visakhapatnam", "nagpur", "indore", "bhopal",
    "patna", "lucknow", "kanpur", "agra", "surat", "vadodara",
    "rajkot", "goa", "kochi", "thiruvananthapuram"
}

PRONOUNS = {
    "i", "you", "he", "she", "we", "they", "it", "me", "us",
    "them", "my", "your", "our", "their"
}

DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those",
    "any", "some", "each", "every"
}

AUXILIARIES = {
    "is", "are", "am", "was", "were", "be", "been", "being",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "do", "does", "did", "have",
    "has", "had"
}

PREPOSITIONS = {
    "from", "to", "at", "in", "on", "for", "with", "by",
    "near", "between", "from", "via", "before", "after"
}

CONJUNCTIONS = {
    "and", "or", "but", "because", "if", "while", "than"
}

WH_WORDS = {
    "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how"
}

ADJECTIVES = {
    "best", "fast", "fastest", "slow", "slowest", "early",
    "late", "morning", "evening", "night", "available",
    "direct", "cheap", "cheapest", "expensive", "local",
    "intercity", "express", "special", "next", "first",
    "last", "shortest", "longest"
}

ADVERBS = {
    "today", "tomorrow", "yesterday", "early", "late",
    "quickly", "slowly", "there", "here"
}

VERBS = {
    "travel", "book", "reserve", "check", "find", "give",
    "show", "tell", "need", "want", "know", "leave",
    "arrive", "reach", "take", "go", "come", "see",
    "provide", "get", "cost", "depart", "run"
}

NOUNS = {
    "train", "bus", "ticket", "tickets", "fare", "price",
    "route", "seat", "seats", "schedule", "station",
    "time", "timing", "journey", "trip", "platform",
    "service", "services", "booking", "reservation",
    "distance", "duration", "availability", "number",
    "coach", "passenger", "metro", "intercity"
}


# ------------------------------------------------------------
# GOLD TAGGER
# Used to create training supervision from the synthetic corpus.
# ------------------------------------------------------------

def gold_tag(word):
    w = word.lower()

    if re.fullmatch(r"\d{1,2}:\d{2}", w):
        return "NUM"

    if re.fullmatch(r"\d+(?:\.\d+)?", w):
        return "NUM"

    if w in CITIES:
        return "PROPN"

    if w in PRONOUNS:
        return "PRON"

    if w in DETERMINERS:
        return "DET"

    if w in AUXILIARIES:
        return "AUX"

    if w in PREPOSITIONS:
        return "PREP"

    if w in CONJUNCTIONS:
        return "CONJ"

    if w in {"to"}:
        return "TO"

    if w in WH_WORDS:
        if w == "how":
            return "WHADV"
        return "WHPRON"

    if w in ADJECTIVES:
        return "ADJ"

    if w in ADVERBS:
        return "ADV"

    if w in VERBS:
        return "VERB"

    if w in NOUNS:
        return "NOUN"

    if w.endswith("ing"):
        return "VERB"

    if w.endswith("ed"):
        return "VERB"

    if w.endswith("ly"):
        return "ADV"

    if w.endswith(("ous", "ful", "able", "ible", "al", "ive")):
        return "ADJ"

    if w.endswith("s") and len(w) > 3:
        return "NOUN"

    if re.fullmatch(r"[^\w\s]", w):
        return "PUNCT"

    return "NOUN"


# ------------------------------------------------------------
# READ CORPUS
# ------------------------------------------------------------

def read_sentences(filename):
    sentences = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                words = tokenize(line)

                if words:
                    sentences.append(words)

    return sentences


# ------------------------------------------------------------
# CREATE TAGGED TRAINING DATA
# ------------------------------------------------------------

def create_tagged_data(sentences):
    tagged = []

    for sentence in sentences:
        tagged_sentence = []

        for word in sentence:
            tagged_sentence.append((word, gold_tag(word)))

        tagged.append(tagged_sentence)

    return tagged


# ------------------------------------------------------------
# HMM TRAINING
# ------------------------------------------------------------

def train_hmm(tagged_sentences):
    tag_counts = Counter()

    transition_counts = defaultdict(Counter)
    emission_counts = defaultdict(Counter)

    vocabulary = set()

    for sentence in tagged_sentences:

        previous_tag = "<START>"

        for word, tag in sentence:
            tag_counts[tag] += 1
            transition_counts[previous_tag][tag] += 1
            emission_counts[tag][word] += 1

            vocabulary.add(word)

            previous_tag = tag

        transition_counts[previous_tag]["<END>"] += 1

    return (
        tag_counts,
        transition_counts,
        emission_counts,
        vocabulary
    )


# ------------------------------------------------------------
# TRANSITION PROBABILITY
# Add-one smoothing
# ------------------------------------------------------------

def transition_probability(previous_tag, current_tag,
                           transition_counts, tag_counts):

    count = transition_counts[previous_tag][current_tag]

    total = sum(transition_counts[previous_tag].values())

    # Add-one smoothing
    denominator = total + len(TAGS) + 1

    return (count + 1) / denominator


# ------------------------------------------------------------
# EMISSION PROBABILITY
# Add-one smoothing
# ------------------------------------------------------------

def emission_probability(word, tag,
                         emission_counts,
                         tag_counts,
                         vocabulary_size):

    count = emission_counts[tag][word]

    denominator = tag_counts[tag] + vocabulary_size + 1

    return (count + 1) / denominator


# ------------------------------------------------------------
# UNKNOWN WORD TAGGING
# ------------------------------------------------------------

def unknown_word_tag(word):

    w = word.lower()

    if re.fullmatch(r"\d{1,2}:\d{2}", w):
        return "NUM"

    if re.fullmatch(r"\d+(?:\.\d+)?", w):
        return "NUM"

    if w.endswith("ing"):
        return "VERB"

    if w.endswith("ed"):
        return "VERB"

    if w.endswith("ly"):
        return "ADV"

    if w.endswith(("ous", "ful", "able", "ible", "ive", "al")):
        return "ADJ"

    return None


# ------------------------------------------------------------
# UNIGRAM BASELINE
# ------------------------------------------------------------

def unigram_baseline(sentence, tag_counts):

    most_common_tag = tag_counts.most_common(1)[0][0]

    result = []

    for word in sentence:

        tag = unknown_word_tag(word)

        if tag is None:
            tag = most_common_tag

        result.append(tag)

    return result


# ------------------------------------------------------------
# VITERBI ALGORITHM
# ------------------------------------------------------------

def viterbi(
    sentence,
    transition_counts,
    emission_counts,
    tag_counts,
    vocabulary
):

    n = len(sentence)

    if n == 0:
        return []

    # Only use the useful tags for this corpus.
    # This significantly speeds up Viterbi.
    active_tags = [
        tag for tag in TAGS
        if tag_counts[tag] > 0
    ]

    vocabulary_size = len(vocabulary)

    # --------------------------------------------------------
    # Dynamic programming tables
    # --------------------------------------------------------

    viterbi_table = []
    backpointer = []

    # --------------------------------------------------------
    # First word
    # --------------------------------------------------------

    first_word = sentence[0]

    first_scores = {}
    first_back = {}

    forced_tag = unknown_word_tag(first_word)

    for tag in active_tags:

        if forced_tag is not None and tag != forced_tag:
            continue

        transition_prob = transition_probability(
            "<START>",
            tag,
            transition_counts,
            tag_counts
        )

        emission_prob = emission_probability(
            first_word,
            tag,
            emission_counts,
            tag_counts,
            vocabulary_size
        )

        score = (
            math.log(transition_prob)
            + math.log(emission_prob)
        )

        first_scores[tag] = score
        first_back[tag] = None

    # Safety fallback
    if not first_scores:
        best_tag = tag_counts.most_common(1)[0][0]

        first_scores[best_tag] = 0.0
        first_back[best_tag] = None

    viterbi_table.append(first_scores)
    backpointer.append(first_back)

    # --------------------------------------------------------
    # Remaining words
    # --------------------------------------------------------

    for position in range(1, n):

        word = sentence[position]

        current_scores = {}
        current_back = {}

        forced_tag = unknown_word_tag(word)

        previous_tags = list(viterbi_table[position - 1].keys())

        current_tag_list = active_tags

        if forced_tag is not None:
            current_tag_list = [forced_tag]

        for current_tag in current_tag_list:

            best_score = float("-inf")
            best_previous = None

            emission_prob = emission_probability(
                word,
                current_tag,
                emission_counts,
                tag_counts,
                vocabulary_size
            )

            log_emission = math.log(emission_prob)

            for previous_tag in previous_tags:

                transition_prob = transition_probability(
                    previous_tag,
                    current_tag,
                    transition_counts,
                    tag_counts
                )

                score = (
                    viterbi_table[position - 1][previous_tag]
                    + math.log(transition_prob)
                    + log_emission
                )

                if score > best_score:
                    best_score = score
                    best_previous = previous_tag

            current_scores[current_tag] = best_score
            current_back[current_tag] = best_previous

        viterbi_table.append(current_scores)
        backpointer.append(current_back)

    # --------------------------------------------------------
    # Find best final tag
    # --------------------------------------------------------

    final_scores = {}

    for tag in viterbi_table[-1]:

        end_transition = transition_probability(
            tag,
            "<END>",
            transition_counts,
            tag_counts
        )

        final_scores[tag] = (
            viterbi_table[-1][tag]
            + math.log(end_transition)
        )

    best_final_tag = max(
        final_scores,
        key=final_scores.get
    )

    # --------------------------------------------------------
    # Backtracking
    # --------------------------------------------------------

    best_path = [best_final_tag]

    for position in range(n - 1, 0, -1):

        previous_tag = backpointer[position][best_path[-1]]

        best_path.append(previous_tag)

    best_path.reverse()

    return best_path


# ------------------------------------------------------------
# EXHAUSTIVE DECODING
# ONLY USED FOR A VERY SHORT SENTENCE
# ------------------------------------------------------------

def exhaustive_decode(
    sentence,
    transition_counts,
    emission_counts,
    tag_counts,
    vocabulary
):

    active_tags = [
        tag for tag in TAGS
        if tag_counts[tag] > 0
    ]

    # Keep exhaustive search safe.
    # For a sentence of length 3 and ~10 tags,
    # this is manageable.
    if len(sentence) > 3:
        sentence = sentence[:3]

    vocabulary_size = len(vocabulary)

    best_path = None
    best_score = float("-inf")

    # Limit candidate tags for exhaustive demonstration.
    candidate_tags = active_tags[:10]

    for path in product(candidate_tags, repeat=len(sentence)):

        score = 0.0

        previous_tag = "<START>"

        valid = True

        for word, current_tag in zip(sentence, path):

            forced_tag = unknown_word_tag(word)

            if forced_tag is not None and current_tag != forced_tag:
                valid = False
                break

            tp = transition_probability(
                previous_tag,
                current_tag,
                transition_counts,
                tag_counts
            )

            ep = emission_probability(
                word,
                current_tag,
                emission_counts,
                tag_counts,
                vocabulary_size
            )

            score += math.log(tp) + math.log(ep)

            previous_tag = current_tag

        if not valid:
            continue

        score += math.log(
            transition_probability(
                previous_tag,
                "<END>",
                transition_counts,
                tag_counts
            )
        )

        if score > best_score:
            best_score = score
            best_path = path

    return best_path, best_score


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

def calculate_accuracy(predictions, gold):

    correct = 0
    total = 0

    for pred_sentence, gold_sentence in zip(predictions, gold):

        for pred, actual in zip(pred_sentence, gold_sentence):

            total += 1

            if pred == actual:
                correct += 1

    if total == 0:
        return 0.0

    return (correct / total) * 100


# ------------------------------------------------------------
# SAVE FULL TRELLIS
# ------------------------------------------------------------

def save_trellis(
    sentence,
    transition_counts,
    emission_counts,
    tag_counts,
    vocabulary,
    filename
):

    n = len(sentence)

    if n == 0:
        return

    active_tags = [
        tag for tag in TAGS
        if tag_counts[tag] > 0
    ]

    vocabulary_size = len(vocabulary)

    table = []

    # First column
    first_scores = {}

    for tag in active_tags:

        tp = transition_probability(
            "<START>",
            tag,
            transition_counts,
            tag_counts
        )

        ep = emission_probability(
            sentence[0],
            tag,
            emission_counts,
            tag_counts,
            vocabulary_size
        )

        first_scores[tag] = (
            math.log(tp) + math.log(ep)
        )

    table.append(first_scores)

    # Remaining columns
    for position in range(1, n):

        word = sentence[position]

        current_scores = {}

        for current_tag in active_tags:

            ep = emission_probability(
                word,
                current_tag,
                emission_counts,
                tag_counts,
                vocabulary_size
            )

            best_score = float("-inf")

            for previous_tag in table[position - 1]:

                tp = transition_probability(
                    previous_tag,
                    current_tag,
                    transition_counts,
                    tag_counts
                )

                score = (
                    table[position - 1][previous_tag]
                    + math.log(tp)
                    + math.log(ep)
                )

                best_score = max(best_score, score)

            current_scores[current_tag] = best_score

        table.append(current_scores)

    with open(filename, "w", encoding="utf-8") as f:

        f.write("HMM VITERBI TRELLIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("Sentence:\n")
        f.write(" ".join(sentence) + "\n\n")

        f.write(
            "Rows = HMM tags, Columns = word positions.\n"
        )
        f.write(
            "Values are best log-probability scores.\n\n"
        )

        for position, scores in enumerate(table):

            f.write(
                f"WORD POSITION {position + 1}: "
                f"{sentence[position]}\n"
            )

            for tag, score in sorted(scores.items()):

                f.write(
                    f"{tag:10s} {score:12.4f}\n"
                )

            f.write("\n")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("HMM PART-OF-SPEECH TAGGER")
    print("=" * 70)

    print("\nLoading training corpus...")

    train_sentences = read_sentences(TRAIN_FILE)

    print(
        f"Training sentences : {len(train_sentences)}"
    )

    print("Creating tagged training data...")

    tagged_train = create_tagged_data(
        train_sentences
    )

    print("Training HMM...")

    training_start = time.perf_counter()

    (
        tag_counts,
        transition_counts,
        emission_counts,
        vocabulary
    ) = train_hmm(tagged_train)

    training_time = (
        time.perf_counter()
        - training_start
    )

    print(
        f"Training time       : {training_time:.4f} seconds"
    )

    print(
        f"Vocabulary size     : {len(vocabulary)}"
    )

    print(
        f"Number of tags      : "
        f"{sum(1 for t in TAGS if tag_counts[t] > 0)}"
    )

    # --------------------------------------------------------
    # TRANSITION EXAMPLES
    # --------------------------------------------------------

    print("\nTransition probability examples:")

    examples = [
        ("<START>", "WHPRON"),
        ("WHPRON", "AUX"),
        ("AUX", "VERB"),
        ("VERB", "PREP"),
        ("PREP", "PROPN")
    ]

    for previous, current in examples:

        probability = transition_probability(
            previous,
            current,
            transition_counts,
            tag_counts
        )

        print(
            f"P({current}|{previous}) = "
            f"{probability:.6f}"
        )

    # --------------------------------------------------------
    # EMISSION EXAMPLES
    # --------------------------------------------------------

    print("\nEmission probability examples:")

    emission_examples = [
        ("train", "NOUN"),
        ("travel", "VERB"),
        ("the", "DET"),
        ("hyderabad", "PROPN")
    ]

    for word, tag in emission_examples:

        probability = emission_probability(
            word,
            tag,
            emission_counts,
            tag_counts,
            len(vocabulary)
        )

        print(
            f"P({word}|{tag}) = "
            f"{probability:.6f}"
        )

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    print("\nLoading test corpus...")

    test_sentences = read_sentences(TEST_FILE)

    gold_sentences = [
        [gold_tag(word) for word in sentence]
        for sentence in test_sentences
    ]

    print(
        f"Test sentences     : {len(test_sentences)}"
    )

    # --------------------------------------------------------
    # UNIGRAM BASELINE
    # --------------------------------------------------------

    print("\nRunning unigram baseline...")

    baseline_start = time.perf_counter()

    baseline_predictions = []

    for sentence in test_sentences:

        baseline_predictions.append(
            unigram_baseline(
                sentence,
                tag_counts
            )
        )

    baseline_time = (
        time.perf_counter()
        - baseline_start
    )

    baseline_accuracy = calculate_accuracy(
        baseline_predictions,
        gold_sentences
    )

    print(
        f"Baseline accuracy   : "
        f"{baseline_accuracy:.2f}%"
    )

    print(
        f"Baseline time       : "
        f"{baseline_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # EXHAUSTIVE DECODING
    # --------------------------------------------------------

    print("\nRunning small exhaustive-decoding demonstration...")

    short_sentence = test_sentences[0][:3]

    print(
        "Sentence used: "
        + " ".join(short_sentence)
    )

    exhaustive_start = time.perf_counter()

    exhaustive_path, exhaustive_score = exhaustive_decode(
        short_sentence,
        transition_counts,
        emission_counts,
        tag_counts,
        vocabulary
    )

    exhaustive_time = (
        time.perf_counter()
        - exhaustive_start
    )

    print(
        "Best exhaustive tags: "
        + str(exhaustive_path)
    )

    print(
        f"Exhaustive score    : "
        f"{exhaustive_score:.4f}"
    )

    print(
        f"Exhaustive time     : "
        f"{exhaustive_time:.6f} seconds"
    )

    # --------------------------------------------------------
    # VITERBI
    # --------------------------------------------------------

    print("\nRunning Viterbi on complete test set...")

    viterbi_start = time.perf_counter()

    viterbi_predictions = []

    for index, sentence in enumerate(test_sentences):

        prediction = viterbi(
            sentence,
            transition_counts,
            emission_counts,
            tag_counts,
            vocabulary
        )

        viterbi_predictions.append(prediction)

        if (index + 1) % 500 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(test_sentences)} test sentences..."
            )

    viterbi_time = (
        time.perf_counter()
        - viterbi_start
    )

    viterbi_accuracy = calculate_accuracy(
        viterbi_predictions,
        gold_sentences
    )

    print(
        f"\nViterbi accuracy    : "
        f"{viterbi_accuracy:.2f}%"
    )

    print(
        f"Viterbi time       : "
        f"{viterbi_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # UNKNOWN WORD ANALYSIS
    # --------------------------------------------------------

    unknown_words = []

    for sentence in test_sentences:

        for word in sentence:

            if word not in vocabulary:
                unknown_words.append(word)

    print("\nUNKNOWN WORD ANALYSIS")

    print(
        f"Unknown tokens     : "
        f"{len(unknown_words)}"
    )

    if unknown_words:

        print(
            "Unknown-word method: "
            "numeric/time/suffix-based heuristic + "
            "add-one emission smoothing"
        )

    else:

        print(
            "Unknown-word method: "
            "add-one emission smoothing"
        )

    # --------------------------------------------------------
    # SAVE TRELLIS
    # --------------------------------------------------------

    trellis_file = "results/task8_trellis.txt"

    save_trellis(
        test_sentences[0],
        transition_counts,
        emission_counts,
        tag_counts,
        vocabulary,
        trellis_file
    )

    print(
        f"\nFull trellis saved to: "
        f"{trellis_file}"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_file = "results/task8_results.txt"

    with open(results_file, "w", encoding="utf-8") as f:

        f.write("TASK 8 - HMM PART-OF-SPEECH TAGGER\n")
        f.write("=" * 70 + "\n\n")

        f.write("DATASET\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Training sentences : {len(train_sentences)}\n"
        )
        f.write(
            f"Test sentences     : {len(test_sentences)}\n"
        )
        f.write(
            f"Vocabulary size    : {len(vocabulary)}\n"
        )
        f.write(
            f"Active tags        : "
            f"{sum(1 for t in TAGS if tag_counts[t] > 0)}\n"
        )

        f.write("\nTRAINING\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Training time      : "
            f"{training_time:.4f} seconds\n"
        )

        f.write("\nTRANSITION PROBABILITY EXAMPLES\n")
        f.write("-" * 70 + "\n")

        for previous, current in examples:

            probability = transition_probability(
                previous,
                current,
                transition_counts,
                tag_counts
            )

            f.write(
                f"P({current}|{previous}) = "
                f"{probability:.6f}\n"
            )

        f.write("\nEMISSION PROBABILITY EXAMPLES\n")
        f.write("-" * 70 + "\n")

        for word, tag in emission_examples:

            probability = emission_probability(
                word,
                tag,
                emission_counts,
                tag_counts,
                len(vocabulary)
            )

            f.write(
                f"P({word}|{tag}) = "
                f"{probability:.6f}\n"
            )

        f.write("\nDECODING COMPARISON\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Unigram baseline accuracy : "
            f"{baseline_accuracy:.2f}%\n"
        )

        f.write(
            f"Unigram baseline time     : "
            f"{baseline_time:.4f} seconds\n"
        )

        f.write(
            f"Exhaustive demo sentence  : "
            f"{' '.join(short_sentence)}\n"
        )

        f.write(
            f"Exhaustive best tags      : "
            f"{exhaustive_path}\n"
        )

        f.write(
            f"Exhaustive time           : "
            f"{exhaustive_time:.6f} seconds\n"
        )

        f.write(
            f"Viterbi accuracy          : "
            f"{viterbi_accuracy:.2f}%\n"
        )

        f.write(
            f"Viterbi time              : "
            f"{viterbi_time:.4f} seconds\n"
        )

        f.write("\nSMOOTHING\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Add-one smoothing was used for transition and "
            "emission probabilities.\n"
        )

        f.write("\nUNKNOWN WORD STRATEGY\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Numeric/time expressions -> NUM\n"
        )
        f.write(
            "-ing / -ed words -> VERB\n"
        )
        f.write(
            "-ly words -> ADV\n"
        )
        f.write(
            "Common adjective suffixes -> ADJ\n"
        )
        f.write(
            "Otherwise add-one emission smoothing is used.\n"
        )

        f.write("\nTRELLIS\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Complete Viterbi trellis for the first test "
            "sentence is stored separately in "
            "results/task8_trellis.txt.\n"
        )

    print(
        f"\nResults saved to: "
        f"{results_file}"
    )

    print("\n" + "=" * 70)
    print("TASK 8 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()