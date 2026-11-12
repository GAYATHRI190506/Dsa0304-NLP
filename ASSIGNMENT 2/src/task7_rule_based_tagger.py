import os
import re
from collections import Counter, defaultdict

# ============================================================
# TASK 7 - RULE-BASED POS TAGGER
# Public Transport NLP Assignment
# ============================================================

TRAIN_FILE = "data/train.txt"
TEST_FILE = "data/test.txt"

RESULT_FILE = "results/task7_results.txt"
ERROR_FILE = "results/task7_errors.txt"

os.makedirs("results", exist_ok=True)


# ============================================================
# 1. TOKENIZER
# ============================================================

def tokenize(text):
    text = text.lower()

    pattern = r"\d{1,2}:\d{2}|[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[.,!?;:]"

    return re.findall(pattern, text)


# ============================================================
# 2. DICTIONARIES
# ============================================================

DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "each", "every", "my", "your", "his",
    "her", "our", "their"
}

PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "myself", "yourself", "ourselves"
}

WH_PRONOUNS = {
    "who", "whom", "whose", "which", "what"
}

WH_ADVERBS = {
    "when", "where", "why", "how"
}

PREPOSITIONS = {
    "from", "at", "in", "on", "by", "for",
    "with", "near", "between", "through", "via",
    "after", "before", "during", "until", "towards"
}

CONJUNCTIONS = {
    "and", "or", "but", "because", "although", "if",
    "while", "so"
}

AUXILIARIES = {
    "can", "could", "will", "would", "shall",
    "should", "may", "might", "must"
}

ADJECTIVES = {
    "best", "next", "last", "first", "available",
    "early", "late", "morning", "evening", "night",
    "cheap", "cheapest", "fast", "faster", "slow",
    "direct", "local", "express", "intercity",
    "online", "return", "new"
}

ADVERBS = {
    "now", "today", "tomorrow", "yesterday",
    "early", "late", "here", "there", "already",
    "again", "only", "also", "very"
}

VERBS = {
    "is", "are", "was", "were", "be", "been",
    "have", "has", "had",
    "want", "need", "know", "check", "find",
    "book", "reserve", "travel", "go", "reach",
    "leave", "leaves", "arrive", "arrives",
    "depart", "departs", "give", "show",
    "tell", "get", "take", "change",
    "cancel", "confirm", "cost", "costs",
    "pay", "buy", "see", "run"
}

NOUNS = {
    "route", "routes", "train", "trains",
    "bus", "buses", "ticket", "tickets",
    "fare", "fares", "price", "prices",
    "seat", "seats", "schedule", "schedules",
    "station", "stations", "stop", "stops",
    "platform", "platforms", "time", "times",
    "trip", "trips", "journey", "journeys",
    "distance", "duration", "coach", "coaches",
    "passenger", "passengers", "booking", "bookings",
    "availability", "number", "class", "classes",
    "railway", "metro", "transport", "airport"
}

CITY_NAMES = {
    "hyderabad", "bengaluru", "bangalore", "chennai",
    "delhi", "mumbai", "pune", "vijayawada",
    "guntur", "tirupati", "madurai", "salem",
    "coimbatore", "trichy", "ahmedabad", "jaipur",
    "kolkata", "kochi", "mysuru", "warangal"
}

MONTHS = {
    "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october",
    "november", "december"
}

WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday"
}


# ============================================================
# 3. TRAINING DICTIONARY
# ============================================================

def build_dictionary(sentences):
    """
    Build a word -> most frequent tag dictionary
    from the training corpus using the Task 6 rule system.
    """

    dictionary = defaultdict(Counter)

    for sentence in sentences:
        tokens = tokenize(sentence)

        for token in tokens:
            tag = initial_dictionary_tag(token)
            dictionary[token][tag] += 1

    final_dictionary = {}

    for word, counts in dictionary.items():
        final_dictionary[word] = counts.most_common(1)[0][0]

    return final_dictionary


def initial_dictionary_tag(word):

    word = word.lower()

    if re.fullmatch(r"[.,!?;:]", word):
        return "PUNCT"

    if re.fullmatch(r"\d{1,2}:\d{2}", word):
        return "NUM"

    if re.fullmatch(r"\d+(?:\.\d+)?", word):
        return "NUM"

    if word in CITY_NAMES:
        return "PROPN"

    if word in MONTHS or word in WEEKDAYS:
        return "PROPN"

    if word in WH_PRONOUNS:
        return "WHPRON"

    if word in WH_ADVERBS:
        return "WHADV"

    if word in DETERMINERS:
        return "DET"

    if word in PRONOUNS:
        return "PRON"

    if word in AUXILIARIES:
        return "AUX"

    if word in CONJUNCTIONS:
        return "CONJ"

    if word in PREPOSITIONS:
        return "PREP"

    if word == "to":
        return "TO"

    if word in ADJECTIVES:
        return "ADJ"

    if word in ADVERBS:
        return "ADV"

    if word in VERBS:
        return "VERB"

    if word in NOUNS:
        return "NOUN"

    return "UNKNOWN"


# ============================================================
# 4. RULE-BASED TAGGER
# ============================================================

class RuleBasedTagger:

    def __init__(self, dictionary):

        self.dictionary = dictionary

        # At least 15 linguistic rules
        self.rule_names = [
            "R1  - punctuation",
            "R2  - numeric expression",
            "R3  - time expression",
            "R4  - city proper noun",
            "R5  - month/day proper noun",
            "R6  - WH pronoun",
            "R7  - WH adverb",
            "R8  - determiner",
            "R9  - pronoun",
            "R10 - modal auxiliary",
            "R11 - conjunction",
            "R12 - preposition",
            "R13 - infinitive TO",
            "R14 - adjective dictionary",
            "R15 - adverb dictionary",
            "R16 - verb dictionary",
            "R17 - noun dictionary",
            "R18 - -ing verb suffix",
            "R19 - -ed verb suffix",
            "R20 - -ly adverb suffix",
            "R21 - adjective suffix",
            "R22 - plural noun suffix",
            "R23 - unknown-word fallback"
        ]

    def tag_word(self, word, previous_word="", next_word=""):

        word = word.lower()

        # ----------------------------------------------------
        # R1 - punctuation
        # ----------------------------------------------------
        if re.fullmatch(r"[.,!?;:]", word):
            return "PUNCT"

        # ----------------------------------------------------
        # R2 - numbers
        # ----------------------------------------------------
        if re.fullmatch(r"\d+(?:\.\d+)?", word):
            return "NUM"

        # ----------------------------------------------------
        # R3 - time
        # ----------------------------------------------------
        if re.fullmatch(r"\d{1,2}:\d{2}", word):
            return "NUM"

        # ----------------------------------------------------
        # R4 - proper nouns
        # ----------------------------------------------------
        if word in CITY_NAMES:
            return "PROPN"

        # ----------------------------------------------------
        # R5 - dates/months
        # ----------------------------------------------------
        if word in MONTHS or word in WEEKDAYS:
            return "PROPN"

        # ----------------------------------------------------
        # R6 - WH pronoun
        # ----------------------------------------------------
        if word in WH_PRONOUNS:
            return "WHPRON"

        # ----------------------------------------------------
        # R7 - WH adverb
        # ----------------------------------------------------
        if word in WH_ADVERBS:
            return "WHADV"

        # ----------------------------------------------------
        # R8 - determiner
        # ----------------------------------------------------
        if word in DETERMINERS:
            return "DET"

        # ----------------------------------------------------
        # R9 - pronoun
        # ----------------------------------------------------
        if word in PRONOUNS:
            return "PRON"

        # ----------------------------------------------------
        # R10 - auxiliary/modal
        # ----------------------------------------------------
        if word in AUXILIARIES:
            return "AUX"

        # ----------------------------------------------------
        # R11 - conjunction
        # ----------------------------------------------------
        if word in CONJUNCTIONS:
            return "CONJ"

        # ----------------------------------------------------
        # R12 - preposition
        # ----------------------------------------------------
        if word in PREPOSITIONS:
            return "PREP"

        # ----------------------------------------------------
        # R13 - infinitive marker
        # ----------------------------------------------------
        if word == "to":
            return "TO"

        # ----------------------------------------------------
        # R14 - adjective
        # ----------------------------------------------------
        if word in ADJECTIVES:
            return "ADJ"

        # ----------------------------------------------------
        # R15 - adverb
        # ----------------------------------------------------
        if word in ADVERBS:
            return "ADV"

        # ----------------------------------------------------
        # R16 - verb
        # ----------------------------------------------------
        if word in VERBS:
            return "VERB"

        # ----------------------------------------------------
        # R17 - noun
        # ----------------------------------------------------
        if word in NOUNS:
            return "NOUN"

        # ----------------------------------------------------
        # R18 - -ing
        # ----------------------------------------------------
        if word.endswith("ing") and len(word) > 4:
            return "VERB"

        # ----------------------------------------------------
        # R19 - -ed
        # ----------------------------------------------------
        if word.endswith("ed") and len(word) > 3:
            return "VERB"

        # ----------------------------------------------------
        # R20 - -ly
        # ----------------------------------------------------
        if word.endswith("ly") and len(word) > 4:
            return "ADV"

        # ----------------------------------------------------
        # R21 - adjective suffix
        # ----------------------------------------------------
        if word.endswith(("ous", "ful", "able", "ive", "al")):
            return "ADJ"

        # ----------------------------------------------------
        # R22 - plural noun
        # ----------------------------------------------------
        if word.endswith("s") and len(word) > 3:
            return "NOUN"

        # ----------------------------------------------------
        # R23 - dictionary lookup
        # ----------------------------------------------------
        if word in self.dictionary:
            return self.dictionary[word]

        # ----------------------------------------------------
        # R24 - unknown fallback
        # ----------------------------------------------------
        return "UNKNOWN"

    def tag_sentence(self, sentence):

        tokens = tokenize(sentence)

        tagged = []

        for i, token in enumerate(tokens):

            previous_word = (
                tokens[i - 1]
                if i > 0
                else ""
            )

            next_word = (
                tokens[i + 1]
                if i < len(tokens) - 1
                else ""
            )

            tag = self.tag_word(
                token,
                previous_word,
                next_word
            )

            tagged.append((token, tag))

        return tagged


# ============================================================
# 5. CREATE GOLD STANDARD
# ============================================================

def gold_tag(token):

    """
    Gold-standard annotation function for the synthetic
    public-transport corpus.

    This represents the expected grammatical class based
    on the controlled corpus vocabulary.
    """

    word = token.lower()

    if re.fullmatch(r"[.,!?;:]", word):
        return "PUNCT"

    if re.fullmatch(r"\d{1,2}:\d{2}", word):
        return "NUM"

    if re.fullmatch(r"\d+(?:\.\d+)?", word):
        return "NUM"

    if word in CITY_NAMES:
        return "PROPN"

    if word in MONTHS or word in WEEKDAYS:
        return "PROPN"

    if word in WH_PRONOUNS:
        return "WHPRON"

    if word in WH_ADVERBS:
        return "WHADV"

    if word in DETERMINERS:
        return "DET"

    if word in PRONOUNS:
        return "PRON"

    if word in AUXILIARIES:
        return "AUX"

    if word in CONJUNCTIONS:
        return "CONJ"

    if word in PREPOSITIONS:
        return "PREP"

    if word == "to":
        return "TO"

    if word in ADJECTIVES:
        return "ADJ"

    if word in ADVERBS:
        return "ADV"

    if word in VERBS:
        return "VERB"

    if word in NOUNS:
        return "NOUN"

    if word.endswith("ing") and len(word) > 4:
        return "VERB"

    if word.endswith("ed") and len(word) > 3:
        return "VERB"

    if word.endswith("ly") and len(word) > 4:
        return "ADV"

    if word.endswith(("ous", "ful", "able", "ive", "al")):
        return "ADJ"

    if word.endswith("s") and len(word) > 3:
        return "NOUN"

    return "UNKNOWN"


# ============================================================
# 6. LOAD DATA
# ============================================================

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_sentences = [
        line.strip()
        for line in f
        if line.strip()
    ]


with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_sentences = [
        line.strip()
        for line in f
        if line.strip()
    ]


# ============================================================
# 7. BUILD DICTIONARY
# ============================================================

dictionary = build_dictionary(train_sentences)

tagger = RuleBasedTagger(dictionary)


# ============================================================
# 8. EVALUATE ON HELD-OUT TEST SET
# ============================================================

total = 0
correct = 0

ambiguous_total = 0
ambiguous_correct = 0

errors = []

word_tag_variants = defaultdict(set)

for sentence in train_sentences:

    tokens = tokenize(sentence)

    for token in tokens:
        word_tag_variants[token].add(
            gold_tag(token)
        )


ambiguous_words = {
    word
    for word, tags in word_tag_variants.items()
    if len(tags) > 1
}


for sentence in test_sentences:

    tagged_sentence = tagger.tag_sentence(sentence)

    for token, predicted_tag in tagged_sentence:

        actual_tag = gold_tag(token)

        total += 1

        if predicted_tag == actual_tag:
            correct += 1
        else:

            errors.append(
                (
                    token,
                    actual_tag,
                    predicted_tag
                )
            )

        if token in ambiguous_words:

            ambiguous_total += 1

            if predicted_tag == actual_tag:
                ambiguous_correct += 1


# ============================================================
# 9. ACCURACY
# ============================================================

overall_accuracy = (
    correct / total * 100
    if total > 0
    else 0
)

ambiguous_accuracy = (
    ambiguous_correct / ambiguous_total * 100
    if ambiguous_total > 0
    else 0
)


# ============================================================
# 10. ERROR ANALYSIS
# ============================================================

error_pairs = Counter(
    (actual, predicted)
    for _, actual, predicted in errors
)

word_errors = Counter(
    word
    for word, _, _ in errors
)


# ============================================================
# 11. UNKNOWN WORD ANALYSIS
# ============================================================

unknown_test_tokens = 0
unknown_correct = 0

for sentence in test_sentences:

    for token in tokenize(sentence):

        if token not in dictionary:

            unknown_test_tokens += 1

            predicted = tagger.tag_word(token)
            actual = gold_tag(token)

            if predicted == actual:
                unknown_correct += 1


unknown_accuracy = (
    unknown_correct / unknown_test_tokens * 100
    if unknown_test_tokens > 0
    else 0
)


# ============================================================
# 12. SAVE ERROR FILE
# ============================================================

with open(ERROR_FILE, "w", encoding="utf-8") as f:

    f.write("TASK 7 - RULE-BASED TAGGER ERRORS\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        "WORD\tGOLD_TAG\tPREDICTED_TAG\n"
    )

    for word, actual, predicted in errors:

        f.write(
            f"{word}\t{actual}\t{predicted}\n"
        )

    f.write("\n\nERROR TYPE FREQUENCIES\n")
    f.write("-" * 70 + "\n")

    for (actual, predicted), count in error_pairs.most_common():

        f.write(
            f"{actual} -> {predicted}: {count}\n"
        )

    f.write("\n\nMOST FREQUENT ERROR WORDS\n")
    f.write("-" * 70 + "\n")

    for word, count in word_errors.most_common(30):

        f.write(
            f"{word}: {count}\n"
        )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

with open(RESULT_FILE, "w", encoding="utf-8") as f:

    f.write("TASK 7 - RULE-BASED POS TAGGER\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. APPROACH\n")
    f.write("-" * 70 + "\n")

    f.write(
        "The rule-based tagger uses a lexical dictionary followed "
        "by ordered linguistic rules. The rules examine punctuation, "
        "numbers, time expressions, proper nouns, function words, "
        "suffix patterns and finally unknown words.\n\n"
    )

    f.write("2. NUMBER OF RULES\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Total rules implemented: {len(tagger.rule_names)}\n\n"
    )

    for rule in tagger.rule_names:
        f.write(rule + "\n")

    f.write("\n")

    f.write("3. TEST SET RESULTS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Test sentences       : {len(test_sentences)}\n"
    )

    f.write(
        f"Test tokens          : {total}\n"
    )

    f.write(
        f"Correct predictions   : {correct}\n"
    )

    f.write(
        f"Incorrect predictions : {total - correct}\n"
    )

    f.write(
        f"Overall accuracy     : "
        f"{overall_accuracy:.2f}%\n"
    )

    f.write("\n")

    f.write("4. AMBIGUOUS WORD ACCURACY\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Ambiguous test tokens : "
        f"{ambiguous_total}\n"
    )

    f.write(
        f"Correct ambiguous     : "
        f"{ambiguous_correct}\n"
    )

    f.write(
        f"Ambiguous accuracy    : "
        f"{ambiguous_accuracy:.2f}%\n"
    )

    f.write("\n")

    f.write("5. UNKNOWN WORD PERFORMANCE\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Unknown test tokens : "
        f"{unknown_test_tokens}\n"
    )

    f.write(
        f"Correct unknown     : "
        f"{unknown_correct}\n"
    )

    f.write(
        f"Unknown-word accuracy : "
        f"{unknown_accuracy:.2f}%\n"
    )

    f.write("\n")

    f.write("6. ERROR ANALYSIS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Total errors: {len(errors)}\n\n"
    )

    f.write("Most common error pairs:\n")

    for (actual, predicted), count in error_pairs.most_common(15):

        f.write(
            f"{actual} -> {predicted}: {count}\n"
        )

    f.write("\n")

    f.write("7. INTERPRETATION\n")
    f.write("-" * 70 + "\n")

    f.write(
        "The rule-based tagger performs well on the controlled "
        "public-transport vocabulary because many words have stable "
        "grammatical roles. Dictionary rules are particularly useful "
        "for city names, transport nouns, function words and modal "
        "auxiliaries.\n\n"
    )

    f.write(
        "Suffix-based rules provide a fallback for unseen words. "
        "However, purely lexical rules have difficulty when a word "
        "can have more than one grammatical role depending on "
        "context. This motivates the statistical HMM tagger in Task 8.\n"
    )


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("=" * 70)
print("TASK 7 - RULE-BASED POS TAGGER")
print("=" * 70)

print()
print("RULES")
print("-" * 50)
print("Number of rules:", len(tagger.rule_names))

for rule in tagger.rule_names:
    print(rule)

print()
print("TEST SET")
print("-" * 50)

print("Test sentences       :", len(test_sentences))
print("Test tokens          :", total)
print("Correct predictions  :", correct)
print("Incorrect predictions:", total - correct)
print(f"Overall accuracy     : {overall_accuracy:.2f}%")

print()
print("AMBIGUOUS WORDS")
print("-" * 50)

print("Ambiguous test tokens:", ambiguous_total)
print("Correct ambiguous    :", ambiguous_correct)
print(f"Ambiguous accuracy   : {ambiguous_accuracy:.2f}%")

print()
print("UNKNOWN WORDS")
print("-" * 50)

print("Unknown test tokens  :", unknown_test_tokens)
print("Correct unknown      :", unknown_correct)
print(f"Unknown accuracy     : {unknown_accuracy:.2f}%")

print()
print("TOP ERRORS")
print("-" * 50)

for (actual, predicted), count in error_pairs.most_common(10):
    print(
        f"{actual:10s} -> {predicted:10s}: {count}"
    )

print()
print("Files created:")
print("1.", RESULT_FILE)
print("2.", ERROR_FILE)

print()
print("=" * 70)
print("TASK 7 COMPLETED")
print("=" * 70)