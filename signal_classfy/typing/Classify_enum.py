from enum import Enum

with open(r'D:\project\code\dataset\classes.txt') as file:
    classes = file.read()

def str_to_list(line):
    line = line.replace('\n', '')
    line = line.replace('classes = ', '')
    line = eval(line)
    return line


classes = str_to_list(classes)


class Classify(Enum):
    EightPSK = classes[0],
    OQPSK = classes[1],
    QPSK = classes[2],
    UQPSK = classes[3]
