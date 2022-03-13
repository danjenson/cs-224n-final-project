from types import SimpleNamespace

import datasets as hfd
import transformers as hft


def build_trainer(cfg):
    model = hft.AutoModelForCausalLM.from_pretrained(cfg.model.checkpoint)
    tokenizer = hft.AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    # TODO: could this be the problem?
    # tokenizer.add_special_tokens(
    #     {'additional_special_tokens': ['<|source|>', '<|target|>']})
    # model.resize_token_embeddings(len(tokenizer))
    collator = hft.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trans = cfg.dataset.translate
    ds = hfd.load_from_disk(cfg.dataset.path)
    ds['train'] = ds['train'].map(
        lambda batch: tokenize(tokenizer, batch, trans.source, trans.target),
        batched=True,
    )
    ds['test'] = ds['test'].map(
        lambda batch: tokenize(
            tokenizer, batch, trans.source, trans.target, omit_labels=True),
        batched=True,
    )
    return hft.Trainer(
        model=model,
        args=hft.TrainingArguments(**vars(cfg.training)),
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        data_collator=collator,
    )


def tokenize(
    tokenizer,
    examples,
    source,
    target,
    omit_labels=False,
):
    t = SimpleNamespace(
        bos=tokenizer.bos_token,
        eos=tokenizer.eos_token,
        source='<|source|>',
        target='<|target|>',
    )
    if omit_labels:
        f = lambda source: f'{t.bos} {t.source} {source} {t.target}'
        encoded = list(map(f, examples[source]))
    else:
        f = lambda source, target: f'{t.bos} {t.source} {source} {t.target} {target} {t.eos}'
        encoded = list(map(f, examples[source], examples[target]))
    return tokenizer(encoded, truncation=True)


def predict(trainer):
    preds = []
    for batch in trainer.get_eval_dataloader():
        ps = trainer.model.generate(
            input_ids=batch['input_ids'].clone().detach().cuda(),
            max_length=100,
            do_sample=False,
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.eos_token_id,
        )
        preds.extend([trainer.tokenizer.decode(p) for p in ps])
    return preds
