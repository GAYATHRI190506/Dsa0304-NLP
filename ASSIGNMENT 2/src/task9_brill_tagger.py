import re
import time
import csv
from collections import Counter

TRAIN_FILE = "data/train.txt"
TEST_FILE = "data/test.txt"

# ============================================================
# TAG SET
# ============================================================

TAGS = [
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV",
    "PRON", "DET", "PREP", "CONJ", "NUM", "WHPRON",
    "WHADV", "TO", "PART", "POSS", "PUNCT", "INTERJ",
    "EXIST", "SYM", "FOREIGN", "UNKNOWN"
]

# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    return re.findall(
        r"\d{1,2}:\d{2}|\d+(?:\.\d+)?|[a-zA-Z]+(?:'[a-zA-Z]+)?|[^\w\s]",
        text.lower()
    )


# ============================================================
# LEXICAL INFORMATION
# ============================================================

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
    "i", "you", "he", "she", "we", "they", "it",
    "me", "us", "them", "my", "your", "our", "their"
}

DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those",
    "any", "some", "each", "every"
}

AUXILIARIES = {
    "is", "are", "am", "was", "were", "be", "been",
    "being", "can", "could", "will", "would", "shall",
    "should", "may", "might", "must", "do", "does",
    "did", "have", "has", "had"
}

PREPOSITIONS = {
    "from", "to", "at", "in", "on", "for", "with", "by",
    "near", "between", "via", "before", "after"
}

CONJUNCTIONS = {
    "and", "or", "but", "because", "if", "while", "than"
}

WH_WORDS = {
    "what", "which", "who", "whom", "whose",
    "where", "when", "why", "how"
}

ADJECTIVES = {
    "best", "fast", "fastest", "slow", "slowest",
    "early", "late", "morning", "evening", "night",
    "available", "direct", "cheap", "cheapest",
    "expensive", "local", "intercity", "express",
    "special", "next", "first", "last",
    "shortest", "longest"
}

ADVERBS = {
    "today", "tomorrow", "yesterday",
    "early", "late", "quickly", "slowly",
    "there", "here"
}

VERBS = {
    "travel", "book", "reserve", "check", "find",
    "give", "show", "tell", "need", "want", "know",
    "leave", "arrive", "reach", "take", "go", "come",
    "see", "provide", "get", "cost", "depart", "run"
}

NOUNS = {
    "train", "bus", "ticket", "tickets", "fare",
    "price", "route", "seat", "seats", "schedule",
    "station", "time", "timing", "journey", "trip",
    "platform", "service", "services", "booking",
    "reservation", "distance", "duration",
    "availability", "number", "coach", "passenger",
    "metro", "intercity"
}


# ============================================================
# GOLD TAGGER
# ============================================================

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

    if w == "to":
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


# ============================================================
# READ DATA
# ============================================================

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


# ============================================================
# INITIAL TAGGER
# ============================================================

def initial_tag(word):

    w = word.lower()

    if re.fullmatch(r"[^\w\s]", w):
        return "PUNCT"

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

    if w == "to":
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

    # Intentionally classify many open-class words
    # as NOUN so that transformation rules have
    # useful errors to correct.
    return "NOUN"


# ============================================================
# CREATE SENTENCE RECORDS
# ============================================================

def create_records(sentences):

    records = []

    for sentence in sentences:

        words = list(sentence)

        gold = [gold_tag(w) for w in words]

        initial = [initial_tag(w) for w in words]

        records.append({
            "words": words,
            "gold": gold,
            "tags": initial
        })

    return records


# ============================================================
# ACCURACY
# ============================================================

def accuracy(records):

    correct = 0
    total = 0

    for record in records:

        for predicted, actual in zip(
            record["tags"],
            record["gold"]
        ):

            total += 1

            if predicted == actual:
                correct += 1

    if total == 0:
        return 0.0

    return correct / total * 100


# ============================================================
# TRANSFORMATION CLASS
# ============================================================

class Transformation:

    def __init__(
        self,
        old_tag,
        new_tag,
        template,
        condition
    ):

        self.old_tag = old_tag
        self.new_tag = new_tag
        self.template = template
        self.condition = condition

    def applies(self, record, index):

        tags = record["tags"]
        words = record["words"]

        old_tag = self.old_tag

        if tags[index] != old_tag:
            return False

        # ----------------------------------------------------
        # Template 1:
        # Previous tag
        # ----------------------------------------------------

        if self.template == 1:

            if index == 0:
                return False

            return tags[index - 1] == self.condition

        # ----------------------------------------------------
        # Template 2:
        # Next tag
        # ----------------------------------------------------

        if self.template == 2:

            if index >= len(tags) - 1:
                return False

            return tags[index + 1] == self.condition

        # ----------------------------------------------------
        # Template 3:
        # Previous two tags
        # ----------------------------------------------------

        if self.template == 3:

            if index < 2:
                return False

            a, b = self.condition

            return (
                tags[index - 2] == a
                and tags[index - 1] == b
            )

        # ----------------------------------------------------
        # Template 4:
        # Next two tags
        # ----------------------------------------------------

        if self.template == 4:

            if index >= len(tags) - 2:
                return False

            a, b = self.condition

            return (
                tags[index + 1] == a
                and tags[index + 2] == b
            )

        # ----------------------------------------------------
        # Template 5:
        # Previous word
        # ----------------------------------------------------

        if self.template == 5:

            if index == 0:
                return False

            return words[index - 1] == self.condition

        # ----------------------------------------------------
        # Template 6:
        # Next word
        # ----------------------------------------------------

        if self.template == 6:

            if index >= len(words) - 1:
                return False

            return words[index + 1] == self.condition

        # ----------------------------------------------------
        # Template 7:
        # Current word
        # ----------------------------------------------------

        if self.template == 7:

            return words[index] == self.condition

        # ----------------------------------------------------
        # Template 8:
        # Previous word + previous tag
        # ----------------------------------------------------

        if self.template == 8:

            if index == 0:
                return False

            previous_word, previous_tag = self.condition

            return (
                words[index - 1] == previous_word
                and tags[index - 1] == previous_tag
            )

        # ----------------------------------------------------
        # Template 9:
        # Next word + next tag
        # ----------------------------------------------------

        if self.template == 9:

            if index >= len(words) - 1:
                return False

            next_word, next_tag = self.condition

            return (
                words[index + 1] == next_word
                and tags[index + 1] == next_tag
            )

        # ----------------------------------------------------
        # Template 10:
        # Word suffix
        # ----------------------------------------------------

        if self.template == 10:

            return words[index].endswith(self.condition)

        return False

    def description(self):

        names = {
            1: "previous tag",
            2: "next tag",
            3: "previous two tags",
            4: "next two tags",
            5: "previous word",
            6: "next word",
            7: "current word",
            8: "previous word + previous tag",
            9: "next word + next tag",
            10: "word suffix"
        }

        return (
            f"Change {self.old_tag} -> {self.new_tag} "
            f"when {names[self.template]} = {self.condition}"
        )


# ============================================================
# CANDIDATE GENERATION
#
# Optimized:
# Instead of generating every possible transformation,
# candidates are generated only from actual errors.
# ============================================================

def generate_candidates(records):

    candidates = set()

    # Only common tags are needed for this synthetic corpus.
    useful_tags = [
        "NOUN", "PROPN", "VERB", "AUX",
        "ADJ", "ADV", "PRON", "DET",
        "PREP", "CONJ", "NUM", "WHPRON",
        "WHADV", "TO"
    ]

    for record in records:

        words = record["words"]
        tags = record["tags"]
        gold = record["gold"]

        for i in range(len(words)):

            if tags[i] == gold[i]:
                continue

            old_tag = tags[i]
            new_tag = gold[i]

            if new_tag not in useful_tags:
                continue

            # Template 1
            if i > 0:
                candidates.add(
                    (
                        old_tag,
                        new_tag,
                        1,
                        tags[i - 1]
                    )
                )

            # Template 2
            if i < len(tags) - 1:
                candidates.add(
                    (
                        old_tag,
                        new_tag,
                        2,
                        tags[i + 1]
                    )
                )

            # Template 3
            if i >= 2:
                candidates.add(
                    (
                        old_tag,
                        new_tag,
                        3,
                        (tags[i - 2], tags[i - 1])
                    )
                )

            # Template 4
            if i < len(tags) - 2:
                candidates.add(
                    (
                        old_tag,
                        new_tag,
                        4,
                        (tags[i + 1], tags[i + 2])
                    )
                )

            # Template 5
            if i > 0:
                candidates.add(
                    (
                        old_tag,
                        new_tag,
                        5,
                        words[i - 1]
                    )
                )

            # Template 6
            if i < len(words) - 1:
                candidates.add(
                    (
                        old_tag,
                        new_tag,
                        6,
                        words[i + 1]
                    )
                )

            # Template 7
            candidates.add(
                (
                    old_tag,
                    new_tag,
                    7,
                    words[i]
                )
            )

            # Template 8
            if i > 0:
                candidates.add(
                    (
                        old_tag,
                        new_tag,
                        8,
                        (words[i - 1], tags[i - 1])
                    )
                )

            # Template 9
            if i < len(words) - 1:
                candidates.add(
                    (
                        old_tag,
                        new_tag,
                        9,
                        (words[i + 1], tags[i + 1])
                    )
                )

            # Template 10
            word = words[i]

            for suffix in [
                "ing", "ed", "ly", "ous",
                "ful", "able", "ible", "ive", "al", "s"
            ]:

                if word.endswith(suffix):

                    candidates.add(
                        (
                            old_tag,
                            new_tag,
                            10,
                            suffix
                        )
                    )

    return candidates


# ============================================================
# SCORE TRANSFORMATION
# ============================================================

def score_transformation(
    transformation,
    records
):

    good = 0
    bad = 0

    for record in records:

        words = record["words"]
        tags = record["tags"]
        gold = record["gold"]

        for i in range(len(words)):

            if not transformation.applies(
                record,
                i
            ):
                continue

            current = tags[i]
            actual = gold[i]

            if current == actual:
                # Already correct.
                bad += 1

            elif transformation.new_tag == actual:
                # Corrects an error.
                good += 1

            else:
                # Makes a wrong tag.
                bad += 1

    return good - bad, good, bad


# ============================================================
# APPLY TRANSFORMATION
# ============================================================

def apply_transformation(
    transformation,
    records
):

    changes = 0

    for record in records:

        for i in range(len(record["words"])):

            if transformation.applies(
                record,
                i
            ):

                if (
                    record["tags"][i]
                    != transformation.new_tag
                ):

                    record["tags"][i] = (
                        transformation.new_tag
                    )

                    changes += 1

    return changes


# ============================================================
# TRAIN BRILL-STYLE TAGGER
# ============================================================

def train_brill(
    records,
    max_rules=10,
    min_score=2
):

    learned_rules = []

    accuracy_curve = []

    print("\nGenerating transformation candidates...")

    start = time.perf_counter()

    for iteration in range(max_rules):

        candidates = generate_candidates(records)

        if not candidates:
            print("No more useful candidates.")
            break

        best_transformation = None
        best_score = float("-inf")
        best_good = 0
        best_bad = 0

        # Evaluate candidates.
        for old_tag, new_tag, template, condition in candidates:

            transformation = Transformation(
                old_tag,
                new_tag,
                template,
                condition
            )

            score, good, bad = score_transformation(
                transformation,
                records
            )

            if score > best_score:

                best_score = score
                best_transformation = transformation
                best_good = good
                best_bad = bad

        if (
            best_transformation is None
            or best_score < min_score
        ):

            print(
                "No transformation reached "
                f"minimum score {min_score}."
            )

            break

        changes = apply_transformation(
            best_transformation,
            records
        )

        learned_rules.append(
            best_transformation
        )

        current_accuracy = accuracy(records)

        accuracy_curve.append(
            (
                iteration + 1,
                current_accuracy
            )
        )

        print(
            f"Rule {iteration + 1:02d}: "
            f"score={best_score}, "
            f"good={best_good}, "
            f"bad={best_bad}, "
            f"changes={changes}, "
            f"accuracy={current_accuracy:.2f}%"
        )

        if changes == 0:
            break

    elapsed = time.perf_counter() - start

    print(
        f"\nTransformation training time: "
        f"{elapsed:.2f} seconds"
    )

    return learned_rules, accuracy_curve, elapsed


# ============================================================
# EVALUATE TEST SET
# ============================================================

def evaluate_test(
    test_records,
    learned_rules
):

    for transformation in learned_rules:

        apply_transformation(
            transformation,
            test_records
        )

    return accuracy(test_records)


# ============================================================
# ERROR ANALYSIS
# ============================================================

def error_analysis(records, limit=15):

    errors = Counter()

    examples = []

    for record in records:

        words = record["words"]
        predicted = record["tags"]
        gold = record["gold"]

        for word, p, g in zip(
            words,
            predicted,
            gold
        ):

            if p != g:

                errors[(p, g)] += 1

                if len(examples) < limit:

                    examples.append(
                        (word, p, g)
                    )

    return errors, examples


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TASK 9 - BRILL TRANSFORMATION-BASED TAGGER")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading training data...")

    train_sentences = read_sentences(
        TRAIN_FILE
    )

    print(
        f"Training sentences : "
        f"{len(train_sentences)}"
    )

    print("\nLoading test data...")

    test_sentences = read_sentences(
        TEST_FILE
    )

    print(
        f"Test sentences     : "
        f"{len(test_sentences)}"
    )

    # --------------------------------------------------------
    # CREATE RECORDS
    # --------------------------------------------------------

    train_records = create_records(
        train_sentences
    )

    test_records = create_records(
        test_sentences
    )

    # --------------------------------------------------------
    # INITIAL ACCURACY
    # --------------------------------------------------------

    initial_train_accuracy = accuracy(
        train_records
    )

    initial_test_accuracy = accuracy(
        test_records
    )

    print("\nINITIAL TAGGER")

    print(
        f"Training accuracy  : "
        f"{initial_train_accuracy:.2f}%"
    )

    print(
        f"Test accuracy      : "
        f"{initial_test_accuracy:.2f}%"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING TRANSFORMATION-BASED TAGGER")
    print("=" * 70)

    learned_rules, accuracy_curve, training_time = (
        train_brill(
            train_records,
            max_rules=10,
            min_score=2
        )
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    final_test_accuracy = evaluate_test(
        test_records,
        learned_rules
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Initial test accuracy : "
        f"{initial_test_accuracy:.2f}%"
    )

    print(
        f"Final test accuracy   : "
        f"{final_test_accuracy:.2f}%"
    )

    print(
        f"Improvement           : "
        f"{final_test_accuracy - initial_test_accuracy:.2f} percentage points"
    )

    print(
        f"Rules learned         : "
        f"{len(learned_rules)}"
    )

    print(
        f"Training time         : "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # FIRST 10 TRANSFORMATIONS
    # --------------------------------------------------------

    print("\nFIRST 10 TRANSFORMATIONS")

    for index, transformation in enumerate(
        learned_rules[:10],
        start=1
    ):

        print(
            f"{index}. "
            f"{transformation.description()}"
        )

    # --------------------------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------------------------

    errors, examples = error_analysis(
        test_records
    )

    print("\nTOP ERROR TYPES")

    for (predicted, gold), count in errors.most_common(10):

        print(
            f"{predicted} -> {gold}: "
            f"{count}"
        )

    print("\nERROR EXAMPLES")

    for word, predicted, gold in examples:

        print(
            f"{word}: "
            f"{predicted} -> {gold}"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_file = "results/task9_results.txt"

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TASK 9 - BRILL TRANSFORMATION-BASED TAGGER\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write("DATASET\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Training sentences : "
            f"{len(train_sentences)}\n"
        )

        f.write(
            f"Test sentences     : "
            f"{len(test_sentences)}\n"
        )

        f.write("\nINITIAL ACCURACY\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Training accuracy : "
            f"{initial_train_accuracy:.2f}%\n"
        )

        f.write(
            f"Test accuracy     : "
            f"{initial_test_accuracy:.2f}%\n"
        )

        f.write("\nTRANSFORMATION TEMPLATES\n")
        f.write("-" * 70 + "\n")

        templates = [
            "1. Change tag based on previous tag",
            "2. Change tag based on next tag",
            "3. Change tag based on previous two tags",
            "4. Change tag based on next two tags",
            "5. Change tag based on previous word",
            "6. Change tag based on next word",
            "7. Change tag based on current word",
            "8. Change tag based on previous word + tag",
            "9. Change tag based on next word + tag",
            "10. Change tag based on word suffix"
        ]

        for template in templates:

            f.write(
                template + "\n"
            )

        f.write("\nLEARNED TRANSFORMATIONS\n")
        f.write("-" * 70 + "\n")

        for index, transformation in enumerate(
            learned_rules[:10],
            start=1
        ):

            f.write(
                f"{index}. "
                f"{transformation.description()}\n"
            )

        f.write("\nACCURACY CURVE\n")
        f.write("-" * 70 + "\n")

        f.write(
            "Initial training accuracy: "
            f"{initial_train_accuracy:.2f}%\n"
        )

        for iteration, acc in accuracy_curve:

            f.write(
                f"After rule {iteration}: "
                f"{acc:.2f}%\n"
            )

        f.write("\nFINAL PERFORMANCE\n")
        f.write("-" * 70 + "\n")

        f.write(
            f"Final test accuracy : "
            f"{final_test_accuracy:.2f}%\n"
        )

        f.write(
            f"Accuracy improvement: "
            f"{final_test_accuracy - initial_test_accuracy:.2f} "
            f"percentage points\n"
        )

        f.write(
            f"Rules learned      : "
            f"{len(learned_rules)}\n"
        )

        f.write(
            f"Training time      : "
            f"{training_time:.2f} seconds\n"
        )

        f.write("\nERROR ANALYSIS\n")
        f.write("-" * 70 + "\n")

        for (predicted, gold), count in errors.most_common(15):

            f.write(
                f"{predicted} -> {gold}: "
                f"{count}\n"
            )

        f.write("\nERROR EXAMPLES\n")
        f.write("-" * 70 + "\n")

        for word, predicted, gold in examples:

            f.write(
                f"{word}: "
                f"{predicted} -> {gold}\n"
            )

    # --------------------------------------------------------
    # SAVE TRANSFORMATIONS
    # --------------------------------------------------------

    transformation_file = (
        "results/task9_transformations.txt"
    )

    with open(
        transformation_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FIRST 10 BRILL TRANSFORMATIONS\n"
        )

        f.write("=" * 70 + "\n\n")

        for index, transformation in enumerate(
            learned_rules[:10],
            start=1
        ):

            f.write(
                f"{index}. "
                f"{transformation.description()}\n"
            )

    # --------------------------------------------------------
    # SAVE ACCURACY CURVE
    # --------------------------------------------------------

    curve_file = (
        "results/task9_accuracy_curve.csv"
    )

    with open(
        curve_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Rule_Number",
            "Training_Accuracy"
        ])

        writer.writerow([
            0,
            round(initial_train_accuracy, 4)
        ])

        for iteration, acc in accuracy_curve:

            writer.writerow([
                iteration,
                round(acc, 4)
            ])

    print("\nFiles saved:")
    print(
        f"1. {results_file}"
    )
    print(
        f"2. {transformation_file}"
    )
    print(
        f"3. {curve_file}"
    )

    print("\n" + "=" * 70)
    print("TASK 9 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()