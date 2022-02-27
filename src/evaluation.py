import sys

import torch

import metric_utils

cuda = torch.device('cuda')


def post_process(command: str, separator):
    # remove query at the start
    input = command[command.find(separator) + len(separator):].split()
    if len(input) == 0:
        return None
    output = [input[0]]
    for i in range(len(input) - 1):
        if input[i + 1] != input[i]:
            output.append(input[i + 1])
    output = " ".join(output)
    return output


def make_prediction(input, tokenizer, model, separator):
    tokens = tokenizer(input, return_tensors="pt")
    tokens = tokens["input_ids"].clone().detach().cuda()
    prediction = model.generate(
        input_ids=tokens,
        max_length=100,
        do_sample=False,  # greedy
        eos_token_id=tokenizer.eos_token_id,  # halt on newline
        pad_token_id=tokenizer.eos_token_id)
    prediction = ' '.join([
        tokenizer.decode(v, clean_up_tokenization_spaces=False)
        for v in prediction
    ])

    prediction = post_process(prediction, separator)
    return prediction


def test_evaluation(testset, tokenizer, model, separator="cmd: "):
    #    with open(file_path, 'r') as f:
    #        lines = [x.strip() for x in f.readlines()]
    total_scores = 0.0
    sample_size = len(testset)
    #    for line in lines:
    for line in testset:
        query, truth = line.split(separator)
        truth = truth.replace('<|endoftext|>', '')
        prediction = make_prediction(query + separator, tokenizer, model,
                                     separator)
        print(f'\nq: {query}', file=sys.stderr)
        print(f'p: {prediction}', file=sys.stderr)
        print(f't: {truth}', file=sys.stderr)
        confidence = 1
        metric_val = metric_utils.compute_metric(prediction, confidence, truth)
        total_scores += metric_val
    total_scores /= sample_size
    print(f"The averaged score is {total_scores}.")
