from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


def prepare_dataset(pairs: List[dict]):
    return Dataset.from_list(pairs)


def finetune_lora(model_name: str, qa_pairs: List[dict], output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1)
    model = get_peft_model(model, config)

    ds = prepare_dataset(qa_pairs)
    ds = ds.map(lambda ex: tokenizer(ex['prompt'], truncation=True), batched=True)

    model.train()
    for epoch in range(3):  # tiny example
        for batch in ds:
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            loss.backward()
            model.optimizer.step()
            model.zero_grad()

    model.save_pretrained(output_dir)
