import re


def clean(prediction):
    return re.sub(' +', ' ', prediction.split('<|target|>')[-1].strip())


def max_len(prediction, n=15):
    special_char = "|"
    tokens = prediction.split()
    if len(tokens) == 0:
        return None
    output = [tokens[0]]
    counter = 0
    for i in range(len(tokens) - 1):
        if tokens[i + 1] != tokens[i]:
            if tokens[i + 1] == special_char:
                counter += 1
                if counter >= 4:
                    break
            output.append(tokens[i + 1])
            if (tokens[i + 1][-1] == ";"):
                break
    if len(output) > n:
        output = output[:n]
    return ' '.join(output)
