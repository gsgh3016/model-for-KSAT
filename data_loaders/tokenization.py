def formatting_prompts_func(example, tokenizer):
    output_texts = []
    for i in range(len(example["messages"])):
        output_texts.append(
            tokenizer.apply_chat_template(
                example["messages"][i],
                tokenize=False,
            )
        )
    return output_texts


def tokenize_dataset(processed_dataset, tokenizer):
    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element, tokenizer),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    # 데이터 토큰화
    tokenized_dataset = processed_dataset.map(
        tokenize,
        # remove_columns=list(processed_dataset.features), # 원본
        remove_columns=processed_dataset.column_names,
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing",
    )
    return tokenized_dataset


def prepare_datasets(tokenized_dataset, max_length=1024, test_size=0.1, seed=42):
    # 데이터 분리
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]
    return train_dataset, eval_dataset
