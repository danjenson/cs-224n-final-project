import difflib
import re

import bashlint.bash


def clean(prediction):
    return re.sub(' +', ' ', prediction.split('<|target|>')[-1].strip())


def top_100(prediction):
    top_100 = bashlint.bash.top_100_utilities
    tokens = prediction.split()
    for idx, token in enumerate(tokens):
        matches = difflib.get_close_matches(token, top_100)
        if matches:
            return ' '.join([matches[0]] + tokens[idx + 1:])
    return prediction


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
