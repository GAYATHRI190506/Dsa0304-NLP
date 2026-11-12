# ============================================================
# DSA03 NLP ASSIGNMENT 3
# Task 1 - Constituency Tests
# ============================================================

def show_test(sentence, phrase, replacement, coordinated_phrase):
    print("=" * 70)
    print(f"Sentence : {sentence}")
    print(f"Phrase   : {phrase}")
    print("-" * 70)

    # Substitution test
    substituted = sentence.replace(phrase, replacement, 1)
    print("1. Substitution Test")
    print(f"   Result: {substituted}")
    print("   Observation: The phrase can be replaced by a shorter constituent.")

    # Movement test
    print("\n2. Movement Test")
    print(f"   Moved form: {phrase}, {sentence.replace(phrase, '', 1).strip()}")
    print("   Observation: The phrase can be moved as a unit.")

    # Coordination test
    print("\n3. Coordination Test")
    print(f"   Coordinated phrase: {phrase} and {coordinated_phrase}")
    print("   Observation: The two phrases have the same syntactic category.")

    print()


def main():

    print("=" * 70)
    print("TASK 1 - CONSTITUENCY TESTS")
    print("=" * 70)

    show_test(
        "The student needs the current timetable.",
        "the current timetable",
        "it",
        "the examination schedule"
    )

    show_test(
        "The student needs a hostel room.",
        "a hostel room",
        "it",
        "a library book"
    )

    show_test(
        "The department provides placement information.",
        "placement information",
        "it",
        "examination information"
    )

    print("=" * 70)
    print("CONSTITUENCY TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()