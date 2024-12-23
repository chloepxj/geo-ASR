# import re
import numpy as np
import torch
import evaluate
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    # Wav2Vec2ProcessorWithLM
)
# from pyctcdecode import build_ctcdecoder

from dataclasses import dataclass
from typing import List, Dict, Union
from utils import *

# Constants
DATA_PATH = "data/"
TRAIN_CSV = DATA_PATH + "train_aug.csv"
DEV_CSV = DATA_PATH + "dev.csv"
TEST_CSV = DATA_PATH + "test_release.csv"

TARGET_SAMPLE_RATE = 16000
OUTPUT_DIR = "./wav2vec2-geo"


# Load datasets
dataset = create_data_set(DATA_PATH, TRAIN_CSV, DEV_CSV, TEST_CSV)
dataset = dataset.map(remove_special_characters)

#######################################################################################################
# define the processor

# Vocabulary and processor
tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|"
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=TARGET_SAMPLE_RATE,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch):
    audio = batch["audio"]

    # Process the input audio
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    # Process the labels (transcripts) directly using the `text` argument
    batch["labels"] = processor(text=batch["transcript"]).input_ids

    return batch


dataset = dataset.map(prepare_dataset, remove_columns=["transcript", "audio"])


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# Data collator and evaluation metric
data_collator = DataCollatorCTCWithPadding(processor=processor)
wer_metric = evaluate.load("wer")

# Metrics computation
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# Load pretrained model
model = Wav2Vec2ForCTC.from_pretrained(
    "cpierse/wav2vec2-large-xlsr-53-esperanto",    
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True,
)
model.freeze_feature_encoder()


# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    report_to = "wandb",
    group_by_length=True,
    per_device_train_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=15,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=1000,
  	eval_steps=500,
  	logging_steps=1000,
  	learning_rate=1e-4,
  	weight_decay=0.005,
  	warmup_steps=1000,
    save_total_limit=5, 
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
trainer.train()
# trainer.train(resume_from_checkpoint="wav2vec2-geo/checkpoint-7000")

# Save model and processor
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"Model and processor saved to {OUTPUT_DIR}")
