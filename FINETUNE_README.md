# Llama Fine-Tuning for `severity` and `repair_cost`

This workflow creates instruction-style training data from `defects_data.csv` and fine-tunes a Llama-family model using LoRA.

## 1) Install dependencies

Use your existing project venv:

```powershell
& "c:/D/Projects/07_Training/Day-2/.venv/Scripts/python.exe" -m pip install --upgrade pip
& "c:/D/Projects/07_Training/Day-2/.venv/Scripts/python.exe" -m pip install torch transformers datasets peft accelerate
```

## 2) Prepare train/validation files

```powershell
& "c:/D/Projects/07_Training/Day-2/.venv/Scripts/python.exe" "c:/D/Projects/07_Training/Day-2/prepare_finetune_data.py" --csv "c:/D/Projects/07_Training/Day-2/defects_data.csv" --out_dir "c:/D/Projects/07_Training/Day-2/training_data"
```

## 3) Start fine-tuning

```powershell
& "c:/D/Projects/07_Training/Day-2/.venv/Scripts/python.exe" "c:/D/Projects/07_Training/Day-2/finetune_llama_lora.py" --train_file "c:/D/Projects/07_Training/Day-2/training_data/train.jsonl" --val_file "c:/D/Projects/07_Training/Day-2/training_data/val.jsonl" --output_dir "c:/D/Projects/07_Training/Day-2/model_outputs"
```

## 4) Notes for Ollama

- Ollama can run custom models built with `Modelfile` and adapters, but adapter compatibility depends on the adapter format.
- The script above outputs a Hugging Face PEFT adapter at `model_outputs/adapter`.
- For production with Ollama, typical flow is: merge adapter into a full model, convert to GGUF, then create with `ollama create`.

If you want, I can add the merge + GGUF conversion scripts next.
