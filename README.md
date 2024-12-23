# Esperanto Language ASR with Wav2Vec2

This project focuses on training and fine-tuning state-of-the-art automatic speech recognition (ASR) models, specifically wav2vec2, for transcribing Esperanto speech into text. The goal is to improve transcription accuracy using various decoding strategies, data augmentation, and language model integration.

## Environment Setup
1. Create a Conda environment with Python 3.10:
   ```bash
   conda create -n esperanto-asr python=3.10
   ```
2. Activate the environment:
   ```bash
   conda activate esperanto-asr
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Files in the Repository

- **`create_vocab.ipynb`**: Generates the vocabulary for Esperanto ASR by processing text data.

- **`data_augmentation.ipynb`**: Implements data augmentation techniques to enrich the training dataset.

- **`LM_ngram.ipynb`**: Creates an n-gram language model to enhance the ASR system's performance.

- **`train_model.py`**: Fine-tunes the Wav2Vec2 model on Esperanto speech data.

- **`eval_test.py`**: Evaluates the model on the test dataset.

- **`eval_val.py`**: Evaluates the model on the validation dataset.

- **`optimize.py`**: Optimizes hyperparameters to improve model performance.

- **`utils.py`**: Contains utility functions used across various scripts and notebooks.


## Wav2Vec2 Models
The wav2vec2 model is an end-to-end self-supervised speech recognition model developed by Facebook AI. The model consists of two main components:

- **Convolutional Feature Encoder**: Takes raw audio input and generates latent speech representations.

- **Transformer Network**: Processes the encoded representations using self-attention mechanisms to model long-range dependencies in the speech signal.

We fine-tuned the wav2vec2-large-xlsr-53 model, pre-trained on the Common Voice dataset, for the Esperanto language. The model was fine-tuned using the Hugging Face training pipelines and optimized to handle Esperanto-specific speech characteristics.


## Data Preparation
The dataset used for training includes audio files in Esperanto with corresponding transcriptions. The data is processed and prepared using the Datasets library. The following steps were performed:
1. **Loading Audio**: The audio files are loaded and processed into the correct format for input into the ASR models.
2. **Creating Vocabulary**: We generated a custom vocabulary specific to Esperanto using the `create_vocab.ipynb` script.

## Vocabulary Creation
The vocabulary is created by processing Esperanto transcripts and extracting unique characters, adding special tokens for padding and unknown words. This vocabulary is then used for training the models.

## Data Augmentation
Data augmentation is used to enhance the training dataset. This includes:
- **Noise Addition**: Adding Gaussian noise to the audio data.
- **Pitch Shifting**: Modifying the pitch of the audio to simulate different speaking styles.
- **Time Stretching**: Stretching the audio time to simulate variations in speech speed.
- **Audio Shifting**: Randomly shifting the audio along the time axis.

The `data_augmentation.ipynb` script applies these transformations to the dataset.

## Language Modeling
To improve transcription accuracy, we integrated an n-gram language model into the decoding pipeline. The language model was trained on the Esperanto transcripts using the KenLM library. We used this model during decoding to improve results by considering word-level dependencies.

## Language Model Integration
We used the `pyctcdecode` library to integrate the n-gram language model into the beam search decoding process. The language model significantly improved results, especially for longer transcriptions, by adjusting the beam search parameters and balancing the influence of the acoustic and language models.

### Experimental Evaluation
We tested the following configurations:
1. **Greedy Decoding with No Language Model**
2. **Beam Search Decoding with No Language Model**
3. **Language Model Only with No Beam Search**
4. **Language Model with Beam Search**

The performance was evaluated based on the **Word Error Rate (WER)**.

## Optimization
To optimize the models, we used the Optuna library to search for the best hyperparameters (e.g., beam width, alpha, beta) to minimize the WER.

## Results
The best-performing configuration was the **wav2vec2 model with beam search and language model integration**, which outperformed all other configurations in terms of **WER**.