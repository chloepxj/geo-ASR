import typing as tp
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import optuna
from transformers import Wav2Vec2Processor, Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
# from jiwer import wer
import evaluate
from utils import *

# Load your model and processor
model_dir = "wav2vec2-esperanto/"
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model.eval()
model = model.cuda()

#---#
# Prepare validation dataset
val_df = pd.read_csv(DEV_CSV)
# test_df = pd.read_csv(TEST_CSV)
val_dataset = create_audio_dataset(val_df, DATA_PATH)
# test_dataset = create_audio_dataset(test_df, DATA_PATH)
dataset = DatasetDict({
    "val": val_dataset,
    # "test": test_dataset
})

def prepare_dataset(batch):
    audio = batch["audio"]

    # Process the input audio
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    # Process the labels (transcripts) directly using the `text` argument
    batch["labels"] = processor(text=batch["transcript"]).input_ids

    return batch

dataset = dataset.map(prepare_dataset, remove_columns=["transcript", "audio"], num_proc=4)

#---#
# Load language model and decoder
lm_path = "LM/2gram_correct.arpa"  # Update this path to your preferred language model
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
labels = list(sorted_vocab_dict.keys())

decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path=lm_path,
    # kenlm_model_path=None,
)
processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

wer_metric = evaluate.load("wer")

#---#
# Parameter optimization (lm+beam search)
def objective(trial):
    alpha = trial.suggest_float("alpha", 0.0, 2.0)
    beta = trial.suggest_float("beta", 0.0, 2.0)
    # beam_width = trial.suggest_categorical("beam_width", [256, 512, 768])
    # beam_width = trial.suggest_int("beam_width", 0, 256, step=16)
    beam_width=32

    decode_params = {
        "alpha": alpha,
        "beta": beta,
        "beam_width": beam_width
    }

    predictions = []
    references = []
    # Iterate over the entire validation dataset
    for batch in tqdm(dataset["val"]):
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0).cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input_values).logits  # Get logits
    
        # Decode logits
        logits_np = logits.squeeze(0).cpu().numpy()  # Convert logits to numpy
        pred_transcript = processor_with_lm.decode(logits_np, **decode_params).text  # beam search decoding
    
        # Append predictions and references
        predictions.append(pred_transcript)
        references.append(processor.decode(batch["labels"], group_tokens=False))  # Decode reference

    # Compute WER
    wer_score = wer_metric.compute(predictions=predictions, references=references)

    return wer_score


# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

# Log results
print("Best trial:")
print(study.best_trial.params)

# Best is trial 12 with value: 0.0791046674297342.
# Best trial:
# {'alpha': 0.05029241573599162, 'beta': 0.022143561134051075, 'beam_width': 256}