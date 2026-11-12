# ============================================================
# DSA03 NLP ASSIGNMENT 3
# Task 1 - CFG Grammar Verification
# ============================================================

from pathlib import Path
import re

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
GRAMMAR_FILE = BASE_DIR / "data" / "grammar.cfg"
LEXICON_FILE = BASE_DIR / "data" / "lexicon.txt"
QUERIES_FILE = BASE_DIR / "data" / "queries.txt"


# ------------------------------------------------------------
# LOAD GRAMMAR
# ------------------------------------------------------------

def load_grammar():
    rules = []

    with open(GRAMMAR_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "->" in line:
                lhs, rhs = line.split("->", 1)

                lhs = lhs.strip()
                rhs = rhs.strip()

                rules.append((lhs, rhs))

    return rules


# ------------------------------------------------------------
# LOAD LEXICON
# ------------------------------------------------------------

def load_lexicon():
    words = []

    with open(LEXICON_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            words.append(line)

    return words


# ------------------------------------------------------------
# LOAD QUERY SENTENCES
# ------------------------------------------------------------

def load_queries():
    queries = []

    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            queries.append(line)

    return queries


# ------------------------------------------------------------
# EXTRACT NONTERMINALS
# ------------------------------------------------------------

def get_nonterminals(rules):
    return sorted(set(lhs for lhs, rhs in rules))


# ------------------------------------------------------------
# EXTRACT RHS SYMBOLS
# ------------------------------------------------------------

def get_rhs_symbols(rules):
    symbols = set()

    for lhs, rhs in rules:
        for symbol in rhs.split():
            symbols.add(symbol)

    return symbols


# ------------------------------------------------------------
# SENTENCE TYPE CLASSIFICATION
# ------------------------------------------------------------

def classify_sentence(sentence):
    s = sentence.strip()

    if s.endswith("?"):
        if re.match(
            r"^(what|which|where|when|why|how|who|whom)\b",
            s,
            re.IGNORECASE
        ):
            return "Wh-question"
        else:
            return "Yes/No question"

    words = s.lower().split()

    if words and words[0].rstrip(".") in {
        "check",
        "submit",
        "download",
        "book",
        "view"
    }:
        return "Imperative"

    return "Declarative"


# ------------------------------------------------------------
# MAIN VERIFICATION
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("TASK 1 - CFG GRAMMAR VERIFICATION")
    print("=" * 70)

    # Load data
    rules = load_grammar()
    unique_rules = sorted(set(rules))

    lexicon = load_lexicon()
    unique_lexicon = sorted(set(lexicon))

    queries = load_queries()

    # --------------------------------------------------------
    # GRAMMAR INFORMATION
    # --------------------------------------------------------

    nonterminals = get_nonterminals(unique_rules)
    rhs_symbols = get_rhs_symbols(unique_rules)

    print("\nGRAMMAR")
    print("-" * 70)

    print(f"Total rule entries       : {len(rules)}")
    print(f"Unique CFG rules         : {len(unique_rules)}")
    print(f"Nonterminals             : {len(nonterminals)}")

    print("\nNonterminals:")
    print(", ".join(nonterminals))

    # --------------------------------------------------------
    # LEXICON INFORMATION
    # --------------------------------------------------------

    print("\nLEXICON")
    print("-" * 70)

    print(f"Total lexical entries    : {len(lexicon)}")
    print(f"Unique lexical words     : {len(unique_lexicon)}")

    # --------------------------------------------------------
    # QUERY INFORMATION
    # --------------------------------------------------------

    print("\nQUERY SET")
    print("-" * 70)

    print(f"Total query sentences    : {len(queries)}")

    sentence_types = {}

    for query in queries:
        sentence_type = classify_sentence(query)

        sentence_types[sentence_type] = (
            sentence_types.get(sentence_type, 0) + 1
        )

    for sentence_type, count in sentence_types.items():
        print(f"{sentence_type:<25}: {count}")

    # --------------------------------------------------------
    # RECURSIVE RULE CHECK
    # --------------------------------------------------------

    recursive_rules = []

    for lhs, rhs in unique_rules:
        if lhs in rhs.split():
            recursive_rules.append(f"{lhs} -> {rhs}")

    print("\nRECURSIVE RULES")
    print("-" * 70)

    for rule in recursive_rules:
        print(rule)

    # --------------------------------------------------------
    # REQUIRED CATEGORY CHECK
    # --------------------------------------------------------

    required_categories = {
        "S",
        "NP",
        "VP",
        "PP",
        "AdjP",
        "Conj"
    }

    missing_categories = required_categories - set(nonterminals)

    print("\nREQUIRED CATEGORY CHECK")
    print("-" * 70)

    if not missing_categories:
        print("PASS - All required phrase categories are present.")
    else:
        print("FAIL - Missing categories:")
        print(", ".join(sorted(missing_categories)))

    # --------------------------------------------------------
    # ASSIGNMENT REQUIREMENT CHECK
    # --------------------------------------------------------

    print("\nASSIGNMENT REQUIREMENTS")
    print("-" * 70)

    rule_pass = len(unique_rules) >= 40
    lexicon_pass = len(unique_lexicon) >= 60
    query_pass = len(queries) >= 30
    recursion_pass = len(recursive_rules) > 0

    print(
        f"40+ unique grammar rules : "
        f"{'PASS' if rule_pass else 'FAIL'} "
        f"({len(unique_rules)})"
    )

    print(
        f"60+ lexical words        : "
        f"{'PASS' if lexicon_pass else 'FAIL'} "
        f"({len(unique_lexicon)})"
    )

    print(
        f"30+ query sentences      : "
        f"{'PASS' if query_pass else 'FAIL'} "
        f"({len(queries)})"
    )

    print(
        f"Recursive CFG rules      : "
        f"{'PASS' if recursion_pass else 'FAIL'}"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if (
        rule_pass
        and lexicon_pass
        and query_pass
        and recursion_pass
        and not missing_categories
    ):
        print("TASK 1 BASIC STRUCTURE: PASSED")
    else:
        print("TASK 1 BASIC STRUCTURE: NEEDS MODIFICATION")

    print("=" * 70)


if __name__ == "__main__":
    main()