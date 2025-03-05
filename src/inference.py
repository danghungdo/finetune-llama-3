import torch


def predict(model, test_df, tokenizer, max_len):
    test_df["query"] = test_df["query"].fillna("").astype(str)
    sentences = test_df["query"].tolist()

    batch_size = 32

    all_predictions = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_len,
        )

        inputs = {
            key: values.to("cuda") if torch.cuda.is_available() else "cpu"
            for key, values in inputs.items()
        }

        with torch.no_grad():
            predictions = model(**inputs)
            all_predictions.append(predictions["logits"])

    final_predictions = torch.cat(all_predictions, dim=0)
    return final_predictions.argmax(axis=1).cpu().numpy()
