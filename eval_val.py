# accident test (only 2 epoch) got the best test result {'wer': 0.11276841969911276, 'cer': 0.026190371122192312}!!

####
import torch
# import librosa
import numpy as np
import pandas as pd
from pyctcdecode import BeamSearchDecoderCTC, build_ctcdecoder #https://github.com/kensho-technologies/pyctcdecode
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
import evaluate

from utils import *
from tqdm import tqdm

# Load the trained model and processor
model_dir = "wav2vec2-esperanto/"
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model.eval()
model = model.cuda()

# Convert the vocabulary into a list sorted by indices
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
labels = list(sorted_vocab_dict.keys())
print(labels)
# Load the language model (optional)
lm_path = "LM/2gram_correct.arpa"     
# lm_path = "LM/3gram_correct.arpa"     
# lm_path = "LM/4gram_correct.arpa"  
# lm_path = "LM/5gram_correct.arpa"     
decoder = build_ctcdecoder(
    labels = labels,
    kenlm_model_path=lm_path, 
    # kenlm_model_path=None,
)

# create a processor with the decoder
processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder,
)

val_df = pd.read_csv(DEV_CSV)
# test_df = pd.read_csv(TEST_CSV)
val_dataset = create_audio_dataset(val_df, DATA_PATH)
# test_dataset = create_audio_dataset(test_df, DATA_PATH)
dataset = DatasetDict({
    "val": val_dataset,
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

wer_metric = evaluate.load("wer")


#-- decode using processor_with_lm --#
predictions = []
references = []
# Iterate over the entire validation dataset
for batch in tqdm(dataset["val"]):
    input_values = torch.tensor(batch["input_values"]).unsqueeze(0).cuda(non_blocking=True)  
    with torch.no_grad():
        logits = model(input_values).logits  # Get logits
    
    # Decode logits
    logits_np = logits.squeeze(0).cpu().numpy()  # Convert logits to numpy
    pred_transcript = processor_with_lm.decode(logits_np, alpha=1.5, beta=0.9, beam_width=1).text
    # beam_width=1 -> without using beam search, close to greedy decoding

    # Append predictions and references
    predictions.append(pred_transcript)
    references.append(processor.decode(batch["labels"], group_tokens=False))  # Decode reference

# Compute WER
wer_score = wer_metric.compute(predictions=predictions, references=references)
print(f"2gram without BS: WER(val): {wer_score:.5f}")

# Optionally print a few predictions and references for manual inspection
# for i, (pred, ref) in enumerate(zip(predictions[:10], references[:10])):  # Show only the first 10 samples
#     print(f"\nSample {i + 1}:")
#     print(f"Prediction: {pred}")
#     print(f"Reference: {ref}")



# def decode_greedy(batch):
#     with torch.no_grad():
#         input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
#         logits = model(input_values).logits

#     pred_ids = torch.argmax(logits, dim=-1) #naive decoding
#     batch["pred_transcript"] = processor.batch_decode(pred_ids)[0]
#     batch["transcript"] = processor.decode(batch["labels"], group_tokens=False)
#     return batch

# val_results_greedy = dataset["val"].map(decode_greedy, remove_columns=['input_values', 'input_length', 'labels'])
# wer_score_greedy = wer_metric.compute(predictions=val_results_greedy["pred_transcript"], references=val_results_greedy["transcript"])
# print(f"greedy: WER(val): {wer_score:.5f}") # 0.07236








