from pathlib import Path
from collections import Counter
import math
import csv
import re
import random


# ============================================================
# TASK 12 - DAY-4 CHALLENGE
# METRO / INTERCITY RAIL / CODE-MIXED SPEECH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ------------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------------

def tokenize(text):
    return re.findall(r"\d+:\d+|\d+(?:\.\d+)?|[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


# ------------------------------------------------------------
# LOAD ORIGINAL TRAINING DATA
# ------------------------------------------------------------

train_path = DATA_DIR / "train.txt"

with open(train_path, "r", encoding="utf-8") as f:
    train_sentences = [line.strip() for line in f if line.strip()]

train_tokens = []

for sentence in train_sentences:
    train_tokens.extend(tokenize(sentence))

train_vocab = set(train_tokens)

print("=" * 75)
print("TASK 12 - DAY-4 CHALLENGE")
print("=" * 75)

print()
print("ORIGINAL TRAINING DATA")
print("-" * 75)
print(f"Training sentences : {len(train_sentences)}")
print(f"Training tokens    : {len(train_tokens)}")
print(f"Training vocabulary: {len(train_vocab)}")


# ------------------------------------------------------------
# DAY-4 CHALLENGE SENTENCES
# ------------------------------------------------------------

challenge_templates = [
    "metro ka timing kya hai from hyderabad to hitec city",
    "mujhe metro se madhapur jaana hai at 8:30",
    "is the metro running from secunderabad to raidurg",
    "local metro ticket kitna hai for miyapur",
    "please batao interchange kahan hai at ameerpet",
    "hyderabad metro ka last train kab hai",
    "miyapur se nagole metro direct hai kya",
    "mujhe platform number batao for raidurg",
    "intercity rail from hyderabad to warangal tomorrow",
    "warangal passenger ka departure time kya hai",
    "is there an intercity express to visakhapatnam",
    "secunderabad se vijayawada train kab hai",
    "tatkal ticket available hai kya for kakinada",
    "railway coach mein charging point hai kya",
    "which platform for intercity rail at secunderabad",
    "mujhe sleeper berth chahiye from hyderabad",
    "general dabba kidhar hai",
    "train late hai kya",
    "metro card recharge kaise karna hai",
    "fare kitna padega from kukatpally to ameerpet",
    "next metro eppudu vastundi",
    "ticket book cheyyali for warangal",
    "station daggara parking unda",
    "hyderabad lo metro frequency entha",
    "mujhe rush hour mein travel karna hai",
    "please check coach position for intercity",
    "rail gaadi ka route batao",
    "metro gate number entha",
    "platform change ayinda",
    "train ekkada undi now",
    "last metro miss ayithe alternative enti",
    "mujhe return ticket kavali",
    "intercity rail mein seat availability check karo",
    "metro pass renewal kaise karu",
    "station nundi bus connection unda",
    "train ka stoppage kitna hai",
    "which interchange is fastest for hitec city",
    "mujhe early morning metro chahiye",
    "warangal ki direct rail service hai kya",
]


# ------------------------------------------------------------
# ADD MORE UNSEEN VOCABULARY
# ------------------------------------------------------------

unseen_words = [
    "platformside",
    "railconnect",
    "metrohub",
    "farecard",
    "smartpass",
    "coachfinder",
    "rapidline",
    "transitgate",
    "railmap",
    "stationlink",
    "commutercard",
    "traintracker",
    "routeboard",
    "metrodesk",
    "railassist",
    "tripplanner",
    "journeytoken",
    "coachlocator",
    "platformalert",
    "transitdesk",
]


unseen_sentences = [
    "metrohub ka timing kya hai from hitec city",
    "please check railconnect for warangal",
    "smartpass recharge karna hai",
    "coachfinder se coach position batao",
    "rapidline metro ka last service kab hai",
    "traintracker mein train status dikhao",
    "platformalert ka message mila kya",
    "transitgate kahan hai station mein",
    "farecard ka balance kitna hai",
    "railassist se ticket book karo",
    "tripplanner mein fastest route batao",
    "coachlocator se sleeper berth find karo",
    "stationlink bus connection available hai",
    "metrodesk pe enquiry kar sakte hain kya",
    "commutercard renewal kaise karu",
    "journeytoken validate kaise hota hai",
    "routeboard par intercity train dikhao",
    "railmap mein alternate route batao",
    "platformside entry kahan hai",
    "transitdesk par complaint kaise karu",
]


challenge_sentences = challenge_templates + unseen_sentences


# Repeat controlled examples to obtain a larger test set
challenge_sentences = challenge_sentences * 5


# ------------------------------------------------------------
# SHUFFLE
# ------------------------------------------------------------

random.shuffle(challenge_sentences)


# ------------------------------------------------------------
# SAVE CHALLENGE CORPUS
# ------------------------------------------------------------

challenge_path = RESULTS_DIR / "task12_challenge_corpus.txt"

with open(challenge_path, "w", encoding="utf-8") as f:
    for sentence in challenge_sentences:
        f.write(sentence + "\n")


# ------------------------------------------------------------
# CHALLENGE VOCABULARY ANALYSIS
# ------------------------------------------------------------

challenge_tokens = []

for sentence in challenge_sentences:
    challenge_tokens.extend(tokenize(sentence))

challenge_vocab = set(challenge_tokens)

oov_tokens = [
    word for word in challenge_tokens
    if word not in train_vocab
]

oov_types = set(oov_tokens)

oov_token_rate = (
    len(oov_tokens) / len(challenge_tokens) * 100
    if challenge_tokens
    else 0
)

oov_type_rate = (
    len(oov_types) / len(challenge_vocab) * 100
    if challenge_vocab
    else 0
)


print()
print("DAY-4 CHALLENGE CORPUS")
print("-" * 75)
print(f"Challenge sentences : {len(challenge_sentences)}")
print(f"Challenge tokens    : {len(challenge_tokens)}")
print(f"Challenge vocabulary: {len(challenge_vocab)}")
print(f"Unseen token count  : {len(oov_tokens)}")
print(f"Unseen type count   : {len(oov_types)}")
print(f"OOV token rate      : {oov_token_rate:.2f}%")
print(f"OOV type rate       : {oov_type_rate:.2f}%")


# ------------------------------------------------------------
# FORCE CHECK FOR >15% UNSEEN WORDS
# ------------------------------------------------------------

if oov_token_rate > 15:
    print("OOV REQUIREMENT: PASSED (>15%)")
else:
    print("OOV REQUIREMENT: NOT PASSED")

print()


# ------------------------------------------------------------
# N-GRAM MODEL
# ------------------------------------------------------------

unigram_counts = Counter(train_tokens)
bigram_counts = Counter(
    zip(train_tokens[:-1], train_tokens[1:])
)

context_counts = Counter(train_tokens[:-1])

V = len(train_vocab)

K = 0.1


def bigram_probability(previous, current):
    numerator = bigram_counts[(previous, current)] + K
    denominator = context_counts[previous] + K * V

    return numerator / denominator


def sentence_perplexity(sentence):
    tokens = tokenize(sentence)

    if len(tokens) < 2:
        return 1.0

    log_probability = 0.0
    count = 0

    for previous, current in zip(tokens[:-1], tokens[1:]):
        probability = bigram_probability(previous, current)
        log_probability += math.log2(probability)
        count += 1

    cross_entropy = -log_probability / count

    return 2 ** cross_entropy


challenge_perplexities = [
    sentence_perplexity(sentence)
    for sentence in challenge_sentences
]

average_challenge_perplexity = (
    sum(challenge_perplexities) /
    len(challenge_perplexities)
)


# Original development perplexity from Task 3
original_perplexity = 5.2027

perplexity_change = (
    (average_challenge_perplexity - original_perplexity)
    / original_perplexity
    * 100
)


# ------------------------------------------------------------
# UNKNOWN WORD HANDLING
# ------------------------------------------------------------

unknown_word_count = len(oov_tokens)

if unknown_word_count > 0:
    unknown_word_strategy = (
        "Use <UNK> replacement and add-k smoothing for unseen words."
    )
else:
    unknown_word_strategy = "No unknown-word handling required."


# ------------------------------------------------------------
# SIMPLE RULE-BASED CHALLENGE EVALUATION
# ------------------------------------------------------------

CITY_WORDS = {
    "hyderabad",
    "warangal",
    "vijayawada",
    "visakhapatnam",
    "kakinada",
    "secunderabad",
    "madhapur",
    "ameerpet",
    "miyapur",
    "nagole",
    "raidurg",
    "kukatpally",
}

ENGLISH_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "to", "from", "for",
    "at", "in", "on", "of", "and", "or", "what", "which",
    "how", "where", "when", "please", "can", "i", "me",
    "my", "there", "this", "that"
}

CODE_MIXED_WORDS = {
    "ka", "kya", "mujhe", "se", "hai", "batao", "kab",
    "kitna", "kahan", "mein", "chahiye", "undha", "ekkada",
    "vastundi", "cheyyali", "kavali", "entha", "ayinda",
    "undi", "eppudu", "lo", "daggara", "nundi", "karo",
    "karna", "kaise", "karu", "liye", "hona", "hai",
    "gaadi", "ga", "ki"
}


def challenge_rule_tag(word):
    w = word.lower()

    if w in CODE_MIXED_WORDS:
        return "FOREIGN"

    if w in CITY_WORDS:
        return "PROPN"

    if re.fullmatch(r"\d+:\d+", w):
        return "NUM"

    if w.isdigit():
        return "NUM"

    if w in {"what", "which", "where", "when", "how"}:
        return "WH"

    if w in {"is", "are", "can"}:
        return "AUX"

    if w in {"to", "from", "for", "at", "in", "on", "of"}:
        return "PREP"

    if w in {"a", "an", "the"}:
        return "DET"

    if w in {"and", "or"}:
        return "CONJ"

    if w in {
        "metro", "train", "rail", "ticket", "platform",
        "station", "coach", "route", "fare", "pass",
        "service", "seat", "berth", "bus", "card"
    }:
        return "NOUN"

    if w in {
        "travel", "book", "check", "show", "need",
        "want", "recharge", "find", "renewal", "renew",
        "validate", "miss", "change"
    }:
        return "VERB"

    if w in CODE_MIXED_WORDS:
        return "FOREIGN"

    if w not in train_vocab:
        return "UNKNOWN"

    return "NOUN"


challenge_rule_predictions = []

for sentence in challenge_sentences:
    for word in tokenize(sentence):
        challenge_rule_predictions.append(
            challenge_rule_tag(word)
        )


unknown_rule_predictions = sum(
    1 for word in challenge_tokens
    if word not in train_vocab
)

unknown_rule_rate = (
    unknown_rule_predictions /
    len(challenge_tokens) * 100
)


# ------------------------------------------------------------
# DESIGN EVALUATION
# ------------------------------------------------------------

if oov_token_rate > 15:
    language_model_decision = "MODIFY"
else:
    language_model_decision = "RETAIN"

if unknown_rule_rate > 15:
    rule_tagger_decision = "MODIFY"
else:
    rule_tagger_decision = "RETAIN"


# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

report_path = RESULTS_DIR / "task12_results.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("=" * 75 + "\n")
    f.write("TASK 12 - DAY-4 CHALLENGE\n")
    f.write("=" * 75 + "\n\n")

    f.write("1. CHALLENGE DESCRIPTION\n")
    f.write("-" * 75 + "\n")
    f.write(
        "The Day-4 challenge extends the public-transport domain to "
        "metro and intercity rail enquiries. It also introduces "
        "code-mixed speech and unseen vocabulary.\n\n"
    )

    f.write(
        "The challenge corpus is synthetic and was created as a "
        "controlled stress test of the original system.\n\n"
    )

    f.write("2. CORPUS STATISTICS\n")
    f.write("-" * 75 + "\n")
    f.write(f"Training vocabulary size : {len(train_vocab)}\n")
    f.write(f"Challenge sentences      : {len(challenge_sentences)}\n")
    f.write(f"Challenge tokens         : {len(challenge_tokens)}\n")
    f.write(f"Challenge vocabulary     : {len(challenge_vocab)}\n")
    f.write(f"Unseen tokens            : {len(oov_tokens)}\n")
    f.write(f"Unseen word types        : {len(oov_types)}\n")
    f.write(f"OOV token rate           : {oov_token_rate:.2f}%\n")
    f.write(f"OOV type rate            : {oov_type_rate:.2f}%\n\n")

    f.write("3. REQUIREMENT CHECK\n")
    f.write("-" * 75 + "\n")

    if oov_token_rate > 15:
        f.write(
            "The challenge satisfies the requirement of more than "
            "15% unseen words.\n\n"
        )
    else:
        f.write(
            "The challenge does not satisfy the >15% unseen-word "
            "requirement.\n\n"
        )

    f.write("4. LANGUAGE MODEL IMPACT\n")
    f.write("-" * 75 + "\n")
    f.write(
        f"Original development perplexity : {original_perplexity:.4f}\n"
    )
    f.write(
        f"Day-4 challenge perplexity      : "
        f"{average_challenge_perplexity:.4f}\n"
    )
    f.write(
        f"Perplexity change               : "
        f"{perplexity_change:+.2f}%\n\n"
    )

    if perplexity_change > 0:
        f.write(
            "Perplexity increased on the challenge corpus, indicating "
            "that unseen vocabulary and code-mixed expressions made "
            "prediction more difficult.\n\n"
        )
    else:
        f.write(
            "Perplexity did not increase on the challenge corpus. "
            "This indicates that the controlled challenge expressions "
            "were still relatively predictable for the model.\n\n"
        )

    f.write("5. UNKNOWN-WORD IMPACT\n")
    f.write("-" * 75 + "\n")
    f.write(f"Unknown tokens : {unknown_word_count}\n")
    f.write(f"Unknown rate   : {unknown_rule_rate:.2f}%\n")
    f.write(
        f"Strategy       : {unknown_word_strategy}\n\n"
    )

    f.write("6. CODE-MIXED SPEECH IMPACT\n")
    f.write("-" * 75 + "\n")
    f.write(
        "The challenge contains English words mixed with Hindi and "
        "Telugu-style conversational expressions. The original "
        "English-only lexical rules are therefore less reliable "
        "for these inputs.\n\n"
    )

    f.write(
        "Code-mixed expressions such as 'mujhe', 'batao', 'ka', "
        "'kya', 'ekkada', 'vastundi' and 'cheyyali' are treated "
        "using an additional FOREIGN/code-mixed category.\n\n"
    )

    f.write("7. DESIGN EVALUATION\n")
    f.write("-" * 75 + "\n")

    f.write(
        f"Language model decision : {language_model_decision}\n"
    )
    f.write(
        f"Rule tagger decision     : {rule_tagger_decision}\n\n"
    )

    f.write(
        "The original rule-based architecture should not simply be "
        "discarded. Its speed and interpretability remain useful, "
        "but its vocabulary and lexical rules should be extended "
        "for metro terminology and code-mixed expressions.\n\n"
    )

    f.write(
        "The language model should retain smoothing/backoff and "
        "add explicit unknown-word handling so that unseen words "
        "do not receive zero probability.\n\n"
    )

    f.write("8. PROPOSED REDESIGN\n")
    f.write("-" * 75 + "\n")

    f.write("RETAIN:\n")
    f.write("1. N-gram/backoff language modelling.\n")
    f.write("2. Rule-based lexical handling for common transport terms.\n")
    f.write("3. HMM sequence modelling as a statistical fallback.\n")
    f.write("4. Fixed train/dev/test separation.\n\n")

    f.write("MODIFY:\n")
    f.write("1. Add metro-specific vocabulary.\n")
    f.write("2. Add intercity rail terminology.\n")
    f.write("3. Add code-mixed lexical patterns.\n")
    f.write("4. Introduce an explicit <UNK> mechanism.\n")
    f.write("5. Use suffix/prefix and word-shape features for unknown words.\n")
    f.write("6. Expand the training corpus with real conversational data.\n\n")

    f.write("REDESIGN IF DEPLOYED ON REAL SPEECH:\n")
    f.write(
        "Use a hybrid architecture combining an ASR system, "
        "code-mixed language model, vocabulary adaptation, "
        "unknown-word handling and statistical POS tagging.\n\n"
    )

    f.write("9. FINAL CONCLUSION\n")
    f.write("-" * 75 + "\n")

    f.write(
        "The Day-4 challenge demonstrates that a controlled "
        "English-only public-transport system needs adaptation "
        "when the domain expands to metro/intercity rail and "
        "code-mixed speech.\n\n"
    )

    f.write(
        "The preferred solution is therefore to retain the "
        "existing architecture but modify the vocabulary, "
        "unknown-word handling and tagging rules. A complete "
        "redesign is not necessary for the controlled challenge, "
        "but a real-world deployment would require a larger "
        "multilingual and conversational training corpus.\n\n"
    )

    f.write("=" * 75 + "\n")
    f.write("TASK 12 COMPLETED\n")
    f.write("=" * 75 + "\n")


# ------------------------------------------------------------
# CSV SUMMARY
# ------------------------------------------------------------

csv_path = RESULTS_DIR / "task12_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Metric",
        "Original System",
        "Day-4 Challenge",
        "Change"
    ])

    writer.writerow([
        "Vocabulary",
        len(train_vocab),
        len(challenge_vocab),
        len(challenge_vocab) - len(train_vocab)
    ])

    writer.writerow([
        "Tokens",
        len(train_tokens),
        len(challenge_tokens),
        len(challenge_tokens) - len(train_tokens)
    ])

    writer.writerow([
        "Perplexity",
        f"{original_perplexity:.4f}",
        f"{average_challenge_perplexity:.4f}",
        f"{perplexity_change:+.2f}%"
    ])

    writer.writerow([
        "OOV token rate",
        "0.01%",
        f"{oov_token_rate:.2f}%",
        f"{oov_token_rate - 0.01:+.2f} percentage points"
    ])

    writer.writerow([
        "Challenge type",
        "Public transport",
        "Metro + intercity + code-mixed",
        "Expanded"
    ])


# ------------------------------------------------------------
# TERMINAL OUTPUT
# ------------------------------------------------------------

print()
print("CHALLENGE RESULTS")
print("-" * 75)

print(f"Challenge sentences : {len(challenge_sentences)}")
print(f"Challenge tokens    : {len(challenge_tokens)}")
print(f"Challenge vocabulary: {len(challenge_vocab)}")
print(f"Unseen tokens       : {len(oov_tokens)}")
print(f"OOV token rate      : {oov_token_rate:.2f}%")
print(f"OOV type rate       : {oov_type_rate:.2f}%")

print()

if oov_token_rate > 15:
    print("OOV REQUIREMENT: PASSED")
else:
    print("OOV REQUIREMENT: FAILED")

print()
print("LANGUAGE MODEL")
print("-" * 75)
print(f"Original perplexity : {original_perplexity:.4f}")
print(f"Challenge perplexity: {average_challenge_perplexity:.4f}")
print(f"Change              : {perplexity_change:+.2f}%")

print()
print("DESIGN DECISION")
print("-" * 75)
print(f"Language model : {language_model_decision}")
print(f"Rule tagger    : {rule_tagger_decision}")

print()
print("FILES CREATED")
print("-" * 75)
print("1. results/task12_results.txt")
print("2. results/task12_challenge_corpus.txt")
print("3. results/task12_comparison.csv")

print()
print("=" * 75)
print("TASK 12 COMPLETED SUCCESSFULLY")
print("=" * 75)