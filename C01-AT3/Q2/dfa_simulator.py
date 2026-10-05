# DFA Simulator - Input from User

# Enter states
states = input("Enter states separated by space: ").split()

# Enter alphabet
alphabet = input("Enter input alphabet separated by space: ").split()

# Enter initial state
initial_state = input("Enter initial state: ")

# Enter final states
final_states = input("Enter final state(s) separated by space: ").split()

# Enter transition table
transition = {}

print("\nEnter transition table:")

for state in states:
    transition[state] = {}
    for symbol in alphabet:
        next_state = input(
            f"Transition({state}, {symbol}) = "
        )
        transition[state][symbol] = next_state

# Number of input strings
n = int(input("\nEnter number of input strings: "))

# Process each string
for i in range(n):
    string = input(f"\nEnter input string {i + 1}: ")

    current_state = initial_state
    path = [current_state]
    valid = True

    for symbol in string:
        if symbol not in alphabet:
            print("Invalid input symbol:", symbol)
            valid = False
            break

        current_state = transition[current_state][symbol]
        path.append(current_state)

    if valid:
        print("Transition Path:")
        print(" → ".join(path))

        if current_state in final_states:
            print("Accepted")
        else:
            print("Rejected")