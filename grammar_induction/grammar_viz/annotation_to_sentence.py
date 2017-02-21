import sys
import os
import operator


def main():
    input_dir = sys.argv[1]
    annotation_mapping_file = sys.argv[2]

    annotation_mapping = parse_mapping(annotation_mapping_file)

    sentences = []
    for file in os.listdir(input_dir):
        if file.endswith(".txt") and os.path.basename(file) != "sentences.txt":
            sentences.append(parse_annotation(input_dir + "/" + file, annotation_mapping))

    write_sentences(input_dir, annotation_mapping, sentences)


# create the annotation mapping between
def parse_mapping(file):
    with open(file) as f:
        content = f.readlines()

    annotation_mapping = dict()
    for mapping in content:
        pair = mapping.split(',')
        annotation_mapping[pair[0]] = int(pair[1])

    return annotation_mapping


def parse_annotation(file, annotation_mapping):
    with open(file) as f:
        content = f.readlines()
        print "Opened file: %s" % f.name

    sentence = []
    for line in content:
        # skip the initial line and the start
        if line[:3] == '---' or line[:5] == 'start':
            continue

        identifier = line.split(',')[0]

        if identifier not in annotation_mapping:
            print("Identifier %s not found in mapping" % identifier)
            sys.exit(1)

        sentence.append(annotation_mapping[identifier])

    return sentence


def write_sentences(output_dir, annotation_mapping, sentences):
    with open(output_dir + "/sentences.txt", 'w') as file:
        # write the annotation types
        file.write(str(len(annotation_mapping)) + '\n')

        sorted_mapping = len(annotation_mapping)*[None]

        # build sorted reverse of str->int action dictionary
        for identifier in annotation_mapping:
            sorted_mapping[annotation_mapping[identifier]-1] = identifier

        for i in range(0, len(sorted_mapping)):
            file.write(str(i) + '\n')

        file.write('\n')
        # write each sentence
        file.write(str(len(sentences)) + '\n')
        for sentence in sentences:
            idx = 0
            for word in sentence:
                file.write(str(word - 1) + " [" + str(idx) + " " + str(idx) +"] ")
                idx += 1
            file.write('\n')



if __name__ == '__main__':
    main()