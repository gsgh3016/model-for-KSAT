from peft import LoraConfig
from trl import SFTConfig

from configs import PeftConfig, SftConfig


def create_peft_config(peft_config: PeftConfig) -> LoraConfig:
    return LoraConfig(
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        target_modules=peft_config.target_modules,
        bias=peft_config.bias,
        task_type=peft_config.task_type,
    )


def create_sft_config(sft_config: SftConfig) -> SFTConfig:
    return SFTConfig(
        do_train=sft_config.do_train,
        do_eval=sft_config.do_eval,
        lr_scheduler_type=sft_config.lr_scheduler_type,
        max_seq_length=sft_config.max_seq_length,
        output_dir="outputs/checkpoint",
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        per_device_eval_batch_size=sft_config.per_device_eval_batch_size,
        num_train_epochs=sft_config.num_train_epochs,
        learning_rate=sft_config.learning_rate,
        weight_decay=sft_config.weight_decay,
        logging_strategy=sft_config.logging_strategy,
        logging_steps=sft_config.logging_steps,
        save_strategy=sft_config.save_strategy,
        eval_strategy=sft_config.eval_strategy,
        save_total_limit=sft_config.save_total_limit,
        save_only_model=sft_config.save_only_model,
        report_to=sft_config.report_to,
    )
