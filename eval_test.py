import torch
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

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
labels = list(sorted_vocab_dict.keys())

# Load the language model (optional)
# lm_path = "LM/2gram_correct.arpa"  
# lm_path = "LM/3gram_correct.arpa" 
# lm_path = "LM/4gram_correct.arpa" 
lm_path = "LM/5gram_correct.arpa" 

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

test_df = pd.read_csv(TEST_CSV)
test_dataset = create_audio_dataset(test_df, DATA_PATH)
dataset = DatasetDict({
    "test": test_dataset
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


# Prepare for testing the entire test set
predictions = []

# Iterate over the entire test dataset
for batch in tqdm(dataset["test"]):  # Process all files in the test set
    input_values = torch.tensor(batch["input_values"]).unsqueeze(0).cuda(non_blocking=True)  
    with torch.no_grad():
        logits = model(input_values).logits  # Get logits
    
    # Decode logits
    logits_np = logits.squeeze(0).cpu().numpy()  # Convert logits to numpy
    pred_transcript = processor_with_lm.decode(logits_np, alpha=0.048, beta=1.47,beam_width=32).text
    
    # Append predictions
    predictions.append({"file": batch["file"], "transcript": pred_transcript})

# Convert predictions to a DataFrame
df_test_results = pd.DataFrame(predictions)

# Save the results to a CSV
df_test_results.to_csv("test_transcripts.csv", index=False)

# Print a preview of the results
print(df_test_results.head())
