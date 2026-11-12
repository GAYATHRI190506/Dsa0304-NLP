# ============================================================
# DSA03 NLP ASSIGNMENT - TASK 8
# Feature Structures, AVMs, Unification and Earley Integration
# ============================================================

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================================
# 1. FEATURE STRUCTURE / AVM
# ============================================================

FeatureStructure = Dict[str, Any]


def print_avm(fs: FeatureStructure, indent: int = 0) -> None:
    """Pretty-print an Attribute-Value Matrix."""
    space = " " * indent

    if not fs:
        print(space + "[ ]")
        return

    print(space + "[")
    for key, value in fs.items():
        if isinstance(value, dict):
            print(space + f"  {key}:")
            print_avm(value, indent + 4)
        else:
            print(space + f"  {key}: {value}")
    print(space + "]")


# ============================================================
# 2. FEATURE PATH
# ============================================================

def get_feature_path(fs: FeatureStructure, path: List[str]) -> Any:
    """Retrieve a value using a feature path."""
    current = fs

    for feature in path:
        if not isinstance(current, dict):
            return None

        if feature not in current:
            return None

        current = current[feature]

    return current


# ============================================================
# 3. REENTRANCY
# ============================================================

def demonstrate_reentrancy() -> None:
    print("\n" + "=" * 70)
    print("3. REENTRANCY")
    print("=" * 70)

    shared_number = {"VALUE": "singular"}

    fs = {
        "SUBJ": {
            "NUM": shared_number
        },
        "VERB": {
            "NUM": shared_number
        }
    }

    print("Feature structure with shared NUM value:")
    print_avm(fs)

    print("\nReentrant paths:")
    print("SUBJ.NUM ->", get_feature_path(fs, ["SUBJ", "NUM"]))
    print("VERB.NUM ->", get_feature_path(fs, ["VERB", "NUM"]))

    print("\nBoth paths point to the same feature value.")
    print("This represents reentrancy.")


# ============================================================
# 4. UNIFICATION
# ============================================================

def unify(
    fs1: FeatureStructure,
    fs2: FeatureStructure
) -> Optional[FeatureStructure]:
    """
    Unify two feature structures.

    If a feature occurs in both structures:
      - nested dictionaries are recursively unified
      - identical atomic values are compatible
      - conflicting atomic values cause failure
    """

    result = deepcopy(fs1)

    for key, value2 in fs2.items():

        if key not in result:
            result[key] = deepcopy(value2)
            continue

        value1 = result[key]

        if isinstance(value1, dict) and isinstance(value2, dict):
            unified = unify(value1, value2)

            if unified is None:
                return None

            result[key] = unified

        elif isinstance(value1, dict) != isinstance(value2, dict):
            return None

        elif value1 != value2:
            return None

    return result


def run_unification_examples() -> None:
    print("\n" + "=" * 70)
    print("4. UNIFICATION EXAMPLES")
    print("=" * 70)

    # --------------------------------------------------------
    # Example 1: Successful unification
    # --------------------------------------------------------

    fs1 = {
        "NUM": "singular",
        "PER": 3
    }

    fs2 = {
        "NUM": "singular",
        "CASE": "nom"
    }

    result1 = unify(fs1, fs2)

    print("\nExample 1 - Successful unification")
    print("FS1:", fs1)
    print("FS2:", fs2)
    print("Result:", result1)

    # --------------------------------------------------------
    # Example 2: Successful nested unification
    # --------------------------------------------------------

    fs3 = {
        "SUBJ": {
            "NUM": "plural"
        },
        "CASE": "nom"
    }

    fs4 = {
        "SUBJ": {
            "PER": 3
        }
    }

    result2 = unify(fs3, fs4)

    print("\nExample 2 - Nested successful unification")
    print("FS3:", fs3)
    print("FS4:", fs4)
    print("Result:", result2)

    # --------------------------------------------------------
    # Example 3: Failed unification
    # --------------------------------------------------------

    fs5 = {
        "NUM": "singular",
        "PER": 3
    }

    fs6 = {
        "NUM": "plural",
        "PER": 3
    }

    result3 = unify(fs5, fs6)

    print("\nExample 3 - Failed unification")
    print("FS5:", fs5)
    print("FS6:", fs6)
    print("Result:", result3)

    if result3 is None:
        print("UNIFICATION FAILURE: conflicting NUM values.")


# ============================================================
# 5. LEXICAL FEATURE ENTRIES
# ============================================================

LEXICON = {

    "student": {
        "CAT": "N",
        "NUM": "singular",
        "PER": 3,
        "CASE": "nom/acc",
        "SUBCAT": []
    },

    "students": {
        "CAT": "N",
        "NUM": "plural",
        "PER": 3,
        "CASE": "nom/acc",
        "SUBCAT": []
    },

    "the": {
        "CAT": "DET",
        "NUM": "any",
        "PER": "any",
        "CASE": "any",
        "SUBCAT": ["N"]
    },

    "a": {
        "CAT": "DET",
        "NUM": "singular",
        "PER": "any",
        "CASE": "any",
        "SUBCAT": ["N"]
    },

    "needs": {
        "CAT": "V",
        "NUM": "singular",
        "PER": 3,
        "CASE": "none",
        "SUBCAT": ["NP"]
    },

    "need": {
        "CAT": "V",
        "NUM": "plural",
        "PER": "any",
        "CASE": "none",
        "SUBCAT": ["NP"]
    },

    "checks": {
        "CAT": "V",
        "NUM": "singular",
        "PER": 3,
        "CASE": "none",
        "SUBCAT": ["NP"]
    },

    "check": {
        "CAT": "V",
        "NUM": "plural",
        "PER": "any",
        "CASE": "none",
        "SUBCAT": ["NP"]
    }
}


def display_lexical_entries() -> None:
    print("\n" + "=" * 70)
    print("5. LEXICAL FEATURE STRUCTURES")
    print("=" * 70)

    selected_words = [
        "student",
        "students",
        "the",
        "a",
        "needs",
        "need",
        "checks",
        "check"
    ]

    print("\nAt least 5 lexical entries with NUM, PER, CASE and SUBCAT:\n")

    for word in selected_words:
        print(f"WORD: {word}")
        print_avm(LEXICON[word])
        print()


# ============================================================
# 6. FEATURE-BASED GRAMMAR
# ============================================================

def check_np_agreement(
    subject: FeatureStructure,
    verb: FeatureStructure
) -> bool:
    """Check subject-verb number/person agreement."""

    subj_num = subject.get("NUM")
    subj_per = subject.get("PER")

    verb_num = verb.get("NUM")
    verb_per = verb.get("PER")

    # Singular third person -> singular third-person verb
    if subj_num == "singular" and subj_per == 3:
        return verb_num == "singular" and verb_per == 3

    # Plural subject -> plural-compatible verb
    if subj_num == "plural":
        return verb_num == "plural"

    return True


def feature_based_sentence_check(
    subject_word: str,
    verb_word: str,
    object_word: str
) -> bool:

    if subject_word not in LEXICON:
        return False

    if verb_word not in LEXICON:
        return False

    if object_word not in LEXICON:
        return False

    subject = LEXICON[subject_word]
    verb = LEXICON[verb_word]
    obj = LEXICON[object_word]

    if subject.get("CAT") != "N":
        return False

    if verb.get("CAT") != "V":
        return False

    if obj.get("CAT") != "N":
        return False

    if not check_np_agreement(subject, verb):
        return False

    if "NP" not in verb.get("SUBCAT", []):
        return False

    return True


def demonstrate_feature_grammar() -> None:
    print("\n" + "=" * 70)
    print("6. FEATURE-BASED GRAMMAR")
    print("=" * 70)

    test_sentences = [
        ("student", "needs", "student"),
        ("students", "need", "student"),
        ("student", "need", "student"),
        ("students", "needs", "student"),
    ]

    print("\nFeature grammar agreement tests:\n")

    for subject, verb, obj in test_sentences:

        accepted = feature_based_sentence_check(
            subject,
            verb,
            obj
        )

        sentence = f"{subject} {verb} the {obj}"

        print(
            f"{sentence:<40} -> "
            f"{'ACCEPT' if accepted else 'REJECT'}"
        )


# ============================================================
# 7. EARLEY FEATURE CHECK
# ============================================================

@dataclass
class FeatureState:
    lhs: str
    rhs: List[str]
    dot: int
    origin: int
    features: FeatureStructure


class FeatureEarleyParser:
    """
    Small Earley-style demonstration showing how
    feature compatibility can be checked during completion.
    """

    def __init__(self):
        self.chart: List[List[FeatureState]] = []

    def feature_compatible(
        self,
        parent_features: FeatureStructure,
        child_features: FeatureStructure
    ) -> bool:

        result = unify(parent_features, child_features)

        return result is not None

    def complete_with_features(
        self,
        parent: FeatureState,
        child_features: FeatureStructure
    ) -> Optional[FeatureStructure]:

        unified = unify(
            parent.features,
            child_features
        )

        return unified


def demonstrate_earley_feature_integration() -> None:
    print("\n" + "=" * 70)
    print("7. FEATURE CHECKS IN EARLEY COMPLETER")
    print("=" * 70)

    parser = FeatureEarleyParser()

    parent = FeatureState(
        lhs="S",
        rhs=["NP", "VP"],
        dot=1,
        origin=0,
        features={
            "SUBJ": {
                "NUM": "singular",
                "PER": 3
            }
        }
    )

    compatible_child = {
        "NUM": "singular",
        "PER": 3
    }

    incompatible_child = {
        "NUM": "plural",
        "PER": 3
    }

    print("\nParent state:")
    print_avm(parent.features)

    print("\nCompatible child features:")
    print_avm(compatible_child)

    result1 = parser.complete_with_features(
        parent,
        compatible_child
    )

    print("Completion result:")
    print_avm(result1 if result1 else {})

    print("\nIncompatible child features:")
    print_avm(incompatible_child)

    result2 = parser.complete_with_features(
        parent,
        incompatible_child
    )

    if result2 is None:
        print("Completion rejected because feature unification failed.")
    else:
        print("Completion accepted unexpectedly:")
        print_avm(result2)


# ============================================================
# 8. FEATURE PATH EXAMPLES
# ============================================================

def demonstrate_feature_paths() -> None:
    print("\n" + "=" * 70)
    print("8. FEATURE PATH EXAMPLES")
    print("=" * 70)

    fs = {
        "SUBJ": {
            "AGR": {
                "NUM": "singular",
                "PER": 3
            },
            "CASE": "nom"
        },
        "OBJ": {
            "AGR": {
                "NUM": "plural",
                "PER": 3
            },
            "CASE": "acc"
        }
    }

    paths = [
        ["SUBJ", "AGR", "NUM"],
        ["SUBJ", "AGR", "PER"],
        ["SUBJ", "CASE"],
        ["OBJ", "AGR", "NUM"],
        ["OBJ", "CASE"]
    ]

    print("\nFeature paths:")

    for path in paths:
        value = get_feature_path(fs, path)
        print(f"{'.'.join(path):<20} -> {value}")


# ============================================================
# 9. SUMMARY
# ============================================================

def print_summary() -> None:
    print("\n" + "=" * 70)
    print("TASK 8 SUMMARY")
    print("=" * 70)

    print("""
1. AVMs represent linguistic information as attribute-value pairs.
2. Feature paths access nested information such as SUBJ.AGR.NUM.
3. Reentrancy represents shared feature values.
4. Unification combines compatible feature structures.
5. Conflicting features cause unification failure.
6. Lexical entries contain NUM, PER, CASE and SUBCAT information.
7. Feature-based grammar checks agreement and subcategorization.
8. Earley completion can use unification to reject incompatible states.
9. Feature structures reduce the rule explosion of agreement-aware CFGs.
10. The same feature mechanism can be extended to a full NLP parser.
""")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("DSA03 NLP ASSIGNMENT - TASK 8")
    print("FEATURE STRUCTURES, AVMs, UNIFICATION AND EARLEY")
    print("=" * 70)

    demonstrate_reentrancy()

    run_unification_examples()

    display_lexical_entries()

    demonstrate_feature_grammar()

    demonstrate_earley_feature_integration()

    demonstrate_feature_paths()

    print_summary()

    print("\n" + "=" * 70)
    print("TASK 8 IMPLEMENTATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()