import metric_utils

def post_process(command: str):
  # remove query at the start
    input = command[command.find("bash:")+5:].split()
    if len(input) == 0:
        return None
    output = [input[0]]
    for i in range(len(input)-1):
        if input[i+1] != input[i]:
            output.append(input[i+1])
    output = " ".join(output)
    return output

def make_prediction(input, tokenizer, model):
    tokens = tokenizer(input, return_tensors="pt")
    tokens = torch.tensor(tokens["input_ids"], dtype=torch.long)
    prediction = model.generate(
                                input_ids=tokens,
                                max_length=50, # max_length less relevant as we do early stopping
                                do_sample=False, # greedy
                                eos_token_id=198, # halt on newline
                                pad_token_id=tokenizer.eos_token_id
                                )
    prediction = ' '.join([tokenizer.decode(v, clean_up_tokenization_spaces=False) for v in prediction])

    prediction = post_process(prediction)
    return prediction

def file_evaluation(file_path:str, tokenizer, model):
    with open(file_path, 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    total_scores = 0.0
    sample_size = len(lines)
    for line in lines:
        query = line[:line.find("bash:")+5]
        ground_truth = line[line.find("bash:")+6:]
        prediction = make_prediction(query, tokenizer, model)
        confidence = 1.0
        metric_val = metric_utils.compute_metric(prediction, confidence, ground_truth)
        total_scores += metric_val
    total_scores /= sample_size
    print(f"The averaged score is {total_scores}.")