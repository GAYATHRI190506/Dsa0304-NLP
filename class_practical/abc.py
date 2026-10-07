s = input("Enter string: ")

state = "q0"

for ch in s:
    if state == "q0":
        if ch == "a":
            state = "q1"
        else:
            state = "q0"

    elif state == "q1":
        if ch == "b":
            state = "q2"
        else:
            state = "q0"

    elif state == "q2":
        if ch == "c":
            state = "q3"
        else:
            state = "q0"

    elif state == "q3":
        if ch == "a":
            state = "q1"
        else:
            state = "q0"

if state == "q3":
    print("True")
else:
    print("False")