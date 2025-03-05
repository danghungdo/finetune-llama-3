from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


def load_model(model_name, config, num_labels):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=config["bnb_4bit_compute_dtype"],
    )

    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        task_type=config["task_type"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, quantization_config=quantization_config, num_labels=num_labels
    )
    model = prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, lora_config)
    return peft_model
