import argparse
import os
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="codex-m")
    args = parser.parse_args()
    dataset_name = args.dataset

    train_file = open(os.path.join(dataset_name, "train.txt"))
    valid_file = open(os.path.join(dataset_name, "valid.txt"))
    test_file = open(os.path.join(dataset_name, "test.txt"))
    output_file = os.path.join(dataset_name, "entity_strings.txt")
    start_separator = ": "
    unique_entities = set()
    files = [train_file, valid_file, test_file]

    for input_file in files:
        for line in tqdm.tqdm(train_file):
            sentence = line[line.find(start_separator)+len(start_separator):]
            split_sentence = sentence.split(" | ")
            unique_entities.add(split_sentence[0].strip())
            split_sentence = split_sentence[1].split(" |\t")
            unique_entities.add(split_sentence[1].strip())

    with open(output_file, "w") as out_file:
        for entity in unique_entities:
            out_file.write(entity + "\n")
