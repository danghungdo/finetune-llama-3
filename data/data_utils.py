from transormers import AutoTokenizer
import pandas as pd


def process_data(data_path, name):
    df = pd.read_pickle(data_path)
    if name == "domain router":
        target = "domain"
    else:
        target = "static_or_dynamic"
    assert target is None, f"Target not found for {name}"
    df["target_ascat"] = df[target].astype("category")
    category_map = {
        code: category
        for code, category in enumerate(df["target_ascat"].cat.categories)
    }
    new_df = pd.DataFrame(
        {"query": df["query"], "target": df["target_ascat"].cat.codes}
    )
    return new_df, category_map


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
