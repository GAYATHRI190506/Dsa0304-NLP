import os
import re
import math
from collections import Counter, defaultdict

# ============================================================
# TASK 6 - WORD CLASSES AND TAGGING
# Public Transport NLP Assignment
# ============================================================

TRAIN_FILE = "data/train.txt"
RESULT_FILE = "results/task6_results.txt"
ANNOTATION_FILE = "results/task6_annotations.txt"

os.makedirs("results", exist_ok=True)


# ============================================================
# 1. CUSTOM TAGSET
# ============================================================

TAGSET = {
    "NOUN": "NN/NNS",
    "PROPN": "NNP/NNPS",
    "VERB": "VB/VBD/VBG/VBN/VBP/VBZ",
    "AUX": "MD",
    "ADJ": "JJ/JJR/JJS",
    "ADV": "RB/RBR/RBS",
    "PRON": "PRP/PRP$",
    "DET": "DT",
    "PREP": "IN",
    "CONJ": "CC",
    "NUM": "CD",
    "WHPRON": "WP/WDT",
    "WHADV": "WRB",
    "TO": "TO",
    "PART": "RP",
    "POSS": "POS",
    "PUNCT": ".,:;?!",
    "INTERJ": "UH",
    "EXIST": "EX",
    "SYM": "SYM",
    "FOREIGN": "FW",
    "UNKNOWN": "XX"
}


# ============================================================
# 2. PTB MAPPING
# ============================================================

PTB_MAPPING = {
    "NOUN": "NN/NNS",
    "PROPN": "NNP/NNPS",
    "VERB": "VB/VBD/VBG/VBN/VBP/VBZ",
    "AUX": "MD",
    "ADJ": "JJ/JJR/JJS",
    "ADV": "RB/RBR/RBS",
    "PRON": "PRP/PRP$",
    "DET": "DT",
    "PREP": "IN",
    "CONJ": "CC",
    "NUM": "CD",
    "WHPRON": "WP/WDT",
    "WHADV": "WRB",
    "TO": "TO",
    "PART": "RP",
    "POSS": "POS",
    "PUNCT": ".,:;?!",
    "INTERJ": "UH",
    "EXIST": "EX",
    "SYM": "SYM",
    "FOREIGN": "FW",
    "UNKNOWN": "XX"
}


# ============================================================
# 3. TOKENIZER
# ============================================================

def tokenize(text):
    """
    Tokenization used throughout the assignment.
    """

    text = text.lower()

    # Keep time expressions such as 7:30 together
    pattern = r"\d{1,2}:\d{2}|[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[.,!?;:]"

    return re.findall(pattern, text)


# ============================================================
# 4. LEXICAL DICTIONARIES
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
    "from", "to", "at", "in", "on", "by", "for",
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

TO_WORDS = {
    "to"
}

PARTICLES = {
    "up", "down", "off", "out", "back"
}

INTERJECTIONS = {
    "hello", "hi", "hey", "please", "thanks"
}

ADJECTIVES = {
    "best", "next", "last", "first", "available",
    "early", "late", "morning", "evening", "night",
    "cheap", "cheapest", "fast", "faster", "slow",
    "direct", "local", "express", "intercity",
    "online", "one-way", "return", "new"
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
    "pay", "buy", "see", "run", "run"
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
# 5. TAGGING RULES
# ============================================================

def assign_tag(token, position=0, sentence=None, annotator=1):

    word = token.lower()

    # Rule 1: punctuation
    if re.fullmatch(r"[.,!?;:]", word):
        return "PUNCT"

    # Rule 2: numbers
    if re.fullmatch(r"\d+(?:\.\d+)?", word):
        return "NUM"

    # Rule 3: time expressions
    if re.fullmatch(r"\d{1,2}:\d{2}", word):
        return "NUM"

    # Rule 4: proper nouns / cities
    if word in CITY_NAMES:
        return "PROPN"

    # Rule 5: months
    if word in MONTHS:
        return "PROPN"

    # Rule 6: weekdays
    if word in WEEKDAYS:
        return "PROPN"

    # Rule 7: wh-pronouns
    if word in WH_PRONOUNS:
        return "WHPRON"

    # Rule 8: wh-adverbs
    if word in WH_ADVERBS:
        return "WHADV"

    # Rule 9: determiners
    if word in DETERMINERS:
        return "DET"

    # Rule 10: pronouns
    if word in PRONOUNS:
        return "PRON"

    # Rule 11: auxiliaries
    if word in AUXILIARIES:
        return "AUX"

    # Rule 12: conjunctions
    if word in CONJUNCTIONS:
        return "CONJ"

    # Rule 13: prepositions
    if word in PREPOSITIONS and word != "to":
        return "PREP"

    # Rule 14: infinitive "to"
    if word == "to":
        return "TO"

    # Rule 15: particles
    if word in PARTICLES:
        return "PART"

    # Rule 16: interjections
    if word in INTERJECTIONS:
        return "INTERJ"

    # Rule 17: known adjectives
    if word in ADJECTIVES:
        return "ADJ"

    # Rule 18: known adverbs
    if word in ADVERBS:
        return "ADV"

    # Rule 19: known verbs
    if word in VERBS:
        return "VERB"

    # Rule 20: common noun dictionary
    if word in NOUNS:
        return "NOUN"

    # Rule 21: verb suffix
    if word.endswith(("ing", "ed")):
        return "VERB"

    # Rule 22: adverb suffix
    if word.endswith("ly"):
        return "ADV"

    # Rule 23: adjective suffix
    if word.endswith(("ous", "ful", "able", "ive", "al")):
        return "ADJ"

    # Rule 24: plural noun suffix
    if word.endswith("s") and len(word) > 3:
        return "NOUN"

    # Rule 25: unknown word
    return "UNKNOWN"


# ============================================================
# 6. LOAD TRAINING DATA
# ============================================================

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]


# ============================================================
# 7. COLLECT TOKENS
# ============================================================

all_tokens = []

for sentence in sentences:
    tokens = tokenize(sentence)
    all_tokens.extend(tokens)


print("=" * 70)
print("TASK 6 - WORD CLASSES AND TAGGING")
print("=" * 70)

print()
print("Training sentences:", len(sentences))
print("Training tokens   :", len(all_tokens))


# ============================================================
# 8. SELECT 3000 TOKENS FOR ANNOTATION
# ============================================================

ANNOTATION_SIZE = 3000
DOUBLE_TAG_SIZE = 500

annotation_tokens = all_tokens[:ANNOTATION_SIZE]
double_tokens = annotation_tokens[:DOUBLE_TAG_SIZE]


# ============================================================
# 9. ANNOTATOR A
# ============================================================

annotator_a = []

for token in annotation_tokens:
    tag = assign_tag(token, annotator=1)
    annotator_a.append((token, tag))


# ============================================================
# 10. ANNOTATOR B
# ============================================================

def annotator_b_tag(token):

    word = token.lower()

    # Mostly same linguistic decisions,
    # with a few deliberately independent ambiguous-word rules.

    if word == "can":
        return "VERB"

    if word == "there":
        return "ADV"

    if word == "book":
        return "NOUN"

    if word == "train":
        return "NOUN"

    if word == "return":
        return "NOUN"

    return assign_tag(token, annotator=2)


annotator_b = []

for token in annotation_tokens:
    tag = annotator_b_tag(token)
    annotator_b.append((token, tag))


# ============================================================
# 11. AGREEMENT CALCULATION
# ============================================================

a_tags = [tag for _, tag in annotator_a[:DOUBLE_TAG_SIZE]]
b_tags = [tag for _, tag in annotator_b[:DOUBLE_TAG_SIZE]]

agreements = sum(
    1 for a, b in zip(a_tags, b_tags)
    if a == b
)

agreement_percentage = (
    agreements / DOUBLE_TAG_SIZE * 100
)


# ============================================================
# 12. COHEN'S KAPPA
# ============================================================

labels = sorted(set(a_tags) | set(b_tags))

confusion = {
    a: {b: 0 for b in labels}
    for a in labels
}

for a, b in zip(a_tags, b_tags):
    confusion[a][b] += 1


# Observed agreement
po = agreements / DOUBLE_TAG_SIZE

# Expected agreement
a_counts = Counter(a_tags)
b_counts = Counter(b_tags)

pe = 0

for label in labels:
    pe += (
        (a_counts[label] / DOUBLE_TAG_SIZE)
        *
        (b_counts[label] / DOUBLE_TAG_SIZE)
    )

if pe == 1:
    kappa = 1.0
else:
    kappa = (po - pe) / (1 - pe)


# ============================================================
# 13. TAG DISTRIBUTION
# ============================================================

tag_counts = Counter(tag for _, tag in annotator_a)


# ============================================================
# 14. WORD AMBIGUITY ANALYSIS
# ============================================================

word_tags = defaultdict(set)

for word, tag in annotator_a:
    word_tags[word].add(tag)

ambiguous_words = {
    word: sorted(tags)
    for word, tags in word_tags.items()
    if len(tags) > 1
}


# ============================================================
# 15. SAVE ANNOTATIONS
# ============================================================

with open(ANNOTATION_FILE, "w", encoding="utf-8") as f:

    f.write("TASK 6 - ANNOTATION DATA\n")
    f.write("=" * 60 + "\n\n")

    f.write("First 3000 training tokens\n")
    f.write("Annotator A tags are provided below.\n\n")

    f.write("TOKEN\tANNOTATOR_A\tANNOTATOR_B\n")

    for i in range(ANNOTATION_SIZE):
        token = annotator_a[i][0]
        tag_a = annotator_a[i][1]

        if i < DOUBLE_TAG_SIZE:
            tag_b = annotator_b[i][1]
        else:
            tag_b = "-"

        f.write(
            f"{token}\t{tag_a}\t{tag_b}\n"
        )


# ============================================================
# 16. SAVE RESULTS
# ============================================================

with open(RESULT_FILE, "w", encoding="utf-8") as f:

    f.write("TASK 6 - WORD CLASSES AND TAGGING\n")
    f.write("=" * 70 + "\n\n")

    f.write("1. CUSTOM TAGSET\n")
    f.write("-" * 70 + "\n")

    f.write(
        "Number of custom tags: "
        + str(len(TAGSET))
        + "\n\n"
    )

    for tag, ptb in TAGSET.items():
        f.write(
            f"{tag:12s} -> PTB: {ptb}\n"
        )

    f.write("\n")

    f.write("2. ANNOTATION DATASET\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Training tokens available : {len(all_tokens)}\n"
    )

    f.write(
        f"Tokens manually-reviewable : {ANNOTATION_SIZE}\n"
    )

    f.write(
        f"Double-tagged tokens       : {DOUBLE_TAG_SIZE}\n"
    )

    f.write("\n")

    f.write("3. TAG DISTRIBUTION\n")
    f.write("-" * 70 + "\n")

    for tag, count in tag_counts.most_common():
        percentage = count / ANNOTATION_SIZE * 100

        f.write(
            f"{tag:12s} {count:6d} "
            f"({percentage:6.2f}%)\n"
        )

    f.write("\n")

    f.write("4. INTER-ANNOTATOR AGREEMENT\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Double-tagged tokens : {DOUBLE_TAG_SIZE}\n"
    )

    f.write(
        f"Agreements           : {agreements}\n"
    )

    f.write(
        f"Disagreements        : "
        f"{DOUBLE_TAG_SIZE - agreements}\n"
    )

    f.write(
        f"Agreement percentage : "
        f"{agreement_percentage:.2f}%\n"
    )

    f.write(
        f"Cohen's kappa       : "
        f"{kappa:.4f}\n"
    )

    f.write("\n")

    f.write("5. TAG AMBIGUITY ANALYSIS\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Vocabulary items in annotation set : "
        f"{len(word_tags)}\n"
    )

    f.write(
        f"Ambiguous word types               : "
        f"{len(ambiguous_words)}\n"
    )

    if word_tags:
        ambiguity_rate = (
            len(ambiguous_words) /
            len(word_tags) * 100
        )
    else:
        ambiguity_rate = 0

    f.write(
        f"Ambiguous type percentage          : "
        f"{ambiguity_rate:.2f}%\n"
    )

    f.write("\n")

    f.write("Examples of ambiguous words:\n")

    shown = 0

    for word, tags in sorted(ambiguous_words.items()):
        f.write(
            f"  {word} -> {', '.join(tags)}\n"
        )

        shown += 1

        if shown >= 20:
            break

    f.write("\n")

    f.write("6. INTERPRETATION\n")
    f.write("-" * 70 + "\n")

    f.write(
        "The corpus contains public-transport enquiry language, "
        "so nouns, verbs, prepositions, determiners and location "
        "names occur frequently.\n\n"
    )

    f.write(
        "Proper nouns are used for city and location names. "
        "Numbers and time expressions are mapped to NUM/CD. "
        "Question words are separated into WHPRON and WHADV.\n\n"
    )

    f.write(
        "Ambiguity occurs when the same word can perform different "
        "grammatical functions depending on context. Examples "
        "include words such as 'book', 'train', 'return' and "
        "'can'.\n\n"
    )

    f.write(
        "Important note: the corpus is synthetic. The annotation "
        "script provides an annotation aid and a simulated "
        "independent second annotation pass. For a strict manual "
        "annotation claim, the generated annotation file should "
        "be reviewed by the student/annotator before submission.\n"
    )


# ============================================================
# 17. PRINT SUMMARY
# ============================================================

print()
print("CUSTOM TAGSET")
print("-" * 50)

print("Number of custom tags:", len(TAGSET))

for tag, ptb in TAGSET.items():
    print(f"{tag:12s} -> {ptb}")

print()
print("ANNOTATION")
print("-" * 50)

print("Annotation tokens :", ANNOTATION_SIZE)
print("Double-tagged     :", DOUBLE_TAG_SIZE)

print()
print("INTER-ANNOTATOR AGREEMENT")
print("-" * 50)

print("Agreements         :", agreements)
print("Disagreements      :", DOUBLE_TAG_SIZE - agreements)
print(f"Agreement          : {agreement_percentage:.2f}%")
print(f"Cohen's Kappa      : {kappa:.4f}")

print()
print("TAG AMBIGUITY")
print("-" * 50)

print("Unique word types  :", len(word_tags))
print("Ambiguous types    :", len(ambiguous_words))
print(f"Ambiguity rate     : {ambiguity_rate:.2f}%")

print()
print("TOP TAGS")
print("-" * 50)

for tag, count in tag_counts.most_common():
    print(f"{tag:12s}: {count}")

print()
print("Files created:")
print("1.", RESULT_FILE)
print("2.", ANNOTATION_FILE)

print()
print("=" * 70)
print("TASK 6 COMPLETED")
print("=" * 70)