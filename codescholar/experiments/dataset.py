# this is a test python file to experiment with ast structures

def largestElemInFile(file: str):
    with open(file) as fp:
        line = fp.read()

    inp = line.split()
    inp = [int(i) for i in line]

    for i in range(len(inp)):
        for j in range(0, len(inp) - i - 1):
            if inp[j] > inp[j + 1]:
                temp = inp[j]
                inp[j] = inp[j + 1]
                inp[j + 1] = temp

    return inp[-1]
