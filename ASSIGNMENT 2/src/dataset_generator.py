import random
from pathlib import Path

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

RANDOM_SEED = 42
TOTAL_UTTERANCES = 20000

random.seed(RANDOM_SEED)

# ---------------------------------------------------------
# TRANSPORT DATA
# ---------------------------------------------------------

places = [
    "Chennai", "Vellore", "Madurai", "Coimbatore", "Salem",
    "Bengaluru", "Hyderabad", "Vijayawada", "Guntur", "Tirupati",
    "Kanchipuram", "Pondicherry", "Trichy", "Erode", "Nellore",
    "Visakhapatnam", "Warangal", "Mysuru", "Kochi", "Mumbai",
    "Delhi", "Pune", "Bhopal", "Jaipur", "Ahmedabad"
]

days = [
    "today", "tomorrow", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

times = [
    "6:00", "6:30", "7:00", "7:30", "8:00",
    "8:30", "9:00", "10:00", "11:30", "12:00",
    "13:00", "14:30", "16:00", "17:30", "18:00",
    "19:30", "20:00", "21:30", "22:00"
]

fares = [
    "50", "75", "100", "120", "150", "180",
    "200", "250", "300", "350", "400", "500"
]

vehicles = [
    "bus", "train", "express bus", "local train",
    "intercity train", "metro"
]

# ---------------------------------------------------------
# SENTENCE TEMPLATES
# ---------------------------------------------------------

templates = [

    "what time does the {vehicle} leave {source}",
    "what time does the {vehicle} reach {destination}",
    "when is the next {vehicle} to {destination}",
    "when does the {vehicle} from {source} arrive",
    "i need a {vehicle} to {destination}",
    "i want to book a {vehicle} from {source} to {destination}",
    "can i book two tickets to {destination}",
    "please book a ticket from {source} to {destination}",
    "how much is the fare from {source} to {destination}",
    "what is the ticket price from {source} to {destination}",
    "how much does a ticket to {destination} cost",
    "is there a {vehicle} from {source} to {destination}",
    "is the {vehicle} available today",
    "is the {vehicle} available tomorrow",
    "does the {vehicle} run on {day}",
    "which {vehicle} goes to {destination}",
    "which {vehicle} leaves {source} at {time}",
    "i need to travel from {source} to {destination}",
    "i want to travel to {destination} on {day}",
    "can you tell me the timing for {vehicle}",
    "tell me the departure time from {source}",
    "tell me the arrival time at {destination}",
    "what is the route from {source} to {destination}",
    "how long does the journey to {destination} take",
    "how long is the trip from {source} to {destination}",
    "is there a direct {vehicle} to {destination}",
    "do i need to change trains for {destination}",
    "does this {vehicle} stop at {destination}",
    "where can i board the {vehicle}",
    "where is the {vehicle} stand at {source}",
    "which platform is the {vehicle} on",
    "what platform does the {vehicle} leave from",
    "my {vehicle} is cancelled what should i do",
    "is the {vehicle} cancelled today",
    "why was my {vehicle} cancelled",
    "how can i get a refund for my ticket",
    "i want to cancel my ticket",
    "can i get a refund for a cancelled {vehicle}",
    "how do i cancel my booking",
    "can i change my travel date",
    "can i change my ticket to {day}",
    "i booked a ticket to {destination}",
    "i want to check my booking status",
    "please check the status of my ticket",
    "what is the fare for the {vehicle}",
    "is there a cheaper {vehicle} to {destination}",
    "what is the fastest route to {destination}",
    "which route is best from {source} to {destination}",
    "can i travel from {source} to {destination} at {time}",
    "is there a {vehicle} at {time}",
    "does the {vehicle} run at {time}",
    "how many tickets are available",
    "are two tickets available to {destination}",
    "i need three tickets to {destination}",
    "book two tickets from {source} to {destination}",
    "book three tickets from {source} to {destination}",
    "what time should i reach the station",
    "what time should i reach the bus stand",
    "is the station open today",
    "where is the station for {destination}",
    "where is the bus stand in {source}",
    "how can i reach {destination}",
    "how can i travel from {source} to {destination}",
    "what is the distance between {source} and {destination}",
    "is there a night {vehicle} to {destination}",
    "is there a morning {vehicle} to {destination}",
    "is there an evening {vehicle} to {destination}",
    "does the {vehicle} have seats available",
    "are seats available on the {vehicle}",
    "can i reserve a seat to {destination}",
    "can i reserve a seat from {source} to {destination}",
    "what are the rules for cancelling a ticket",
    "what is the cancellation charge",
    "how much refund will i get",
    "when will my refund arrive",
    "my refund has not arrived",
    "i paid for a ticket but did not receive confirmation",
    "my booking confirmation is missing",
    "please check my reservation",
    "can i get a return ticket to {destination}",
    "i need a return ticket from {source}",
    "what is the return fare to {destination}",
    "is there a direct route from {source} to {destination}",
    "which station should i get down at",
    "which stop is closest to {destination}",
    "does this route pass through {destination}",
    "does the {vehicle} stop at {destination}",
    "what time will i reach {destination}",
    "what time will i arrive at {destination}",
    "what time will i leave {source}",
    "is the {vehicle} running late",
    "how late is the {vehicle}",
    "why is the {vehicle} delayed",
    "what is the current status of the {vehicle}",
    "has the {vehicle} departed",
    "has the {vehicle} arrived",
    "please give me the latest update on the {vehicle}",
    "i want information about the {vehicle} to {destination}",
    "give me the schedule for {destination}",
    "give me the bus schedule from {source}",
    "give me the train schedule from {source}",
    "what are the timings from {source} to {destination}",
    "show me buses from {source} to {destination}",
    "show me trains from {source} to {destination}",
    "find a {vehicle} for {destination}",
    "find the next {vehicle} from {source}",
]

# ---------------------------------------------------------
# GENERATE ONE UTTERANCE
# ---------------------------------------------------------

def generate_utterance():
    template = random.choice(templates)

    source = random.choice(places)
    destination = random.choice(places)

    # Avoid source and destination being identical
    while destination == source:
        destination = random.choice(places)

    vehicle = random.choice(vehicles)
    day = random.choice(days)
    time = random.choice(times)

    utterance = template.format(
        source=source,
        destination=destination,
        vehicle=vehicle,
        day=day,
        time=time
    )

    return utterance.lower()


# ---------------------------------------------------------
# GENERATE CORPUS
# ---------------------------------------------------------

def generate_corpus():
    corpus = set()

    while len(corpus) < TOTAL_UTTERANCES:
        corpus.add(generate_utterance())

    return list(corpus)


# ---------------------------------------------------------
# TOKEN COUNT
# ---------------------------------------------------------

def count_tokens(utterances):
    total_tokens = 0

    for utterance in utterances:
        total_tokens += len(utterance.split())

    return total_tokens


# ---------------------------------------------------------
# TRAIN / DEV / TEST SPLIT
# ---------------------------------------------------------

def split_dataset(utterances):

    random.shuffle(utterances)

    total = len(utterances)

    train_end = int(total * 0.80)
    dev_end = int(total * 0.90)

    train = utterances[:train_end]
    dev = utterances[train_end:dev_end]
    test = utterances[dev_end:]

    return train, dev, test


# ---------------------------------------------------------
# SAVE FILE
# ---------------------------------------------------------

def save_file(filename, utterances):

    path = Path(filename)

    with open(path, "w", encoding="utf-8") as file:

        for utterance in utterances:
            file.write(utterance + "\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("PUBLIC TRANSPORT NLP DATASET GENERATOR")
    print("=" * 60)

    print("\nGenerating corpus...")

    corpus = generate_corpus()

    total_tokens = count_tokens(corpus)

    print("\nDataset created successfully!")
    print(f"Number of utterances : {len(corpus)}")
    print(f"Number of word tokens: {total_tokens}")

    train, dev, test = split_dataset(corpus)

    print("\nDataset split:")
    print(f"Training utterances   : {len(train)}")
    print(f"Development utterances: {len(dev)}")
    print(f"Testing utterances    : {len(test)}")

    print("\nToken counts:")
    print(f"Training tokens   : {count_tokens(train)}")
    print(f"Development tokens: {count_tokens(dev)}")
    print(f"Testing tokens    : {count_tokens(test)}")

    # Save files
    save_file("data/corpus.txt", corpus)
    save_file("data/train.txt", train)
    save_file("data/dev.txt", dev)
    save_file("data/test.txt", test)

    print("\nFiles created:")
    print("data/corpus.txt")
    print("data/train.txt")
    print("data/dev.txt")
    print("data/test.txt")

    print("\nFirst 10 sample utterances:")

    for i, utterance in enumerate(corpus[:10], start=1):
        print(f"{i}. {utterance}")

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()