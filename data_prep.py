import sys
from typing import List
import string
# This file will be used to format the data and labels correctly

def format_string(arr: List[str]) -> str:
    line = sys.argv[1] + "|"
    for entry in arr:
        line += entry + " "
    return line + "\n"

def main():
    lines_to_append = []
    punct_remover = str.maketrans('', '', string.punctuation)
    num_remover = str.maketrans('', '', '012346789')
    with open(sys.argv[2], 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.lower()
            line = line.translate(punct_remover)
            line = line.translate(num_remover)
            sep_line = line.split()
            i = 0
            while i + 15 < len(sep_line):
                lines_to_append.append(format_string(sep_line[i: i + 15]))
                i += 15
    
    with open(sys.argv[3], 'a') as data_file:
        for l in lines_to_append:
            data_file.write(l)

if __name__ == "__main__":
    main()