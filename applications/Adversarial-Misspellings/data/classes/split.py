import sys

filename = sys.argv[1]


label_file = open(filename + ".new.labels", 'w')
new_file = open(filename + ".new.txt", 'w')


def read_valid_lines(filename):
    """ignores the neutral reviews
    
    Arguments:
        filename -- data file
    
    Returns:
        lines, tags: list of reviews, and their tags
    """
    
    print ("starting to read ", filename)

    lines, tags = [], []
    with open(filename, 'r') as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            if tag == '0' or tag == '1': tag = '0'
            if tag == '3' or tag == '4': tag = '1'
            if tag == '2': continue
            tags.append(tag)
            lines.append(words)
    return lines, tags

lines, tags = read_valid_lines(filename)
assert len(lines) == len(tags)

for idx in range(len(lines)):
    new_file.write(lines[idx].lower() + "\n")
    label_file.write(tags[idx] + "\n")
