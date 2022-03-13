import numpy as np
import datasets as hfd
import transformers as hft


def build_trainer(cfg):
    model = hft.AutoModelForSeq2SeqLM.from_pretrained(cfg.model.checkpoint)
    tokenizer = hft.AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    collator = hft.DataCollatorForSeq2Seq(tokenizer, model)
    trans = cfg.dataset.translate
    ds = hfd.load_from_disk(cfg.dataset.path)
    ds = ds.map(
        lambda batch: tokenize(tokenizer, batch, trans.source, trans.target),
        batched=True,
    )
    return hft.Seq2SeqTrainer(
        model=model,
        args=hft.Seq2SeqTrainingArguments(**vars(cfg.training)),
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        data_collator=collator,
    )


def tokenize(tokenizer, batch, source, target):
    inputs = tokenizer(batch[source], truncation=True)
    with tokenizer.as_target_tokenizer():
        inputs['labels'] = tokenizer(batch[target],
                                     truncation=True)['input_ids']
    return inputs


def predict(trainer):
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.predict
    pad_id = -100
    res = trainer.predict(trainer.eval_dataset)
    decode = trainer.tokenizer.decode
    output_ids = np.argmax(res.predictions[0], axis=-1)
    return [
        decode(p[np.where(p != pad_id)], skip_special_tokens=True)
        for p in output_ids
    ]
