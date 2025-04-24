from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from ds_loader import load_dataset




def generate_extra_column(dataset_name: str,
                          model: AutoModelForCausalLM,
                          tokenizer: AutoTokenizer,
                          device: torch.device,
                          batch_size: int = 100) -> None:
    df = load_dataset(dataset_name)
    extra_column = []

    print(f"Generating text for dataset: {dataset_name}...")

    prompts = [
        f"[INST] Imagine that you are a journalist working in a newspaper. Write an article for the following subject: '{row['statement']}'. Please write it as one continuous block of text, no formatting, no captioning, no headings. [/INST]"
        for _, row in df.iterrows()
    ]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for full_output in decoded:
            assistant_response = full_output.split("[/INST]")[-1].strip()
            extra_column.append(assistant_response.replace('\t', '').replace('\n', ''))
        
    df["articles"] = extra_column
    df["articles"].to_csv(
        f"data/result/articles/{dataset_name}.tsv",
        sep="\t",
        index=False
    )


if __name__ == '__main__':
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.capture_scalar_outputs = False
    torch._dynamo.disable()

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generate_extra_column("test2", model, tokenizer, device)
    generate_extra_column("train2", model, tokenizer, device)
    generate_extra_column("val2", model, tokenizer, device)