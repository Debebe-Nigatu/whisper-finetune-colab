# Fine-Tuning Whisper-Tiny on MInds14 (English)

Fine-tunes `openai/whisper-tiny` on `PolyAI/minds14` (`en-US`) for automatic speech recognition,
built to match the format the **Hugging Face Audio Course, Unit 5** grader expects.

**Trained model:** [Atrac/my-seq2seq-model](https://huggingface.co/Atrac/my-seq2seq-model)


## What it does

Whisper-tiny is a general-purpose speech recognition model. This project fine-tunes it specifically
on MInds14's English intent-classification audio clips, freezing most of the encoder so only the
decoder's cross-attention layers and the rest of the decoder adapt to the new data.

## Project structure

1. **Load the dataset** — pulls `PolyAI/minds14` (`en-US`, train split) from the Hub.

2. **Trim to the essentials** — keeps only `audio` and `transcription`, renaming the latter to
   `sentence`.

3. **Load the processor** — `WhisperProcessor` for `openai/whisper-tiny`, configured for English
   transcription.

4. **Resample audio** — casts the audio column to match the processor's expected sampling rate.

5. **Extract features & tokenize** — maps each example to `input_features` (log-mel spectrogram),
   `labels` (tokenized transcript), and `input_length` (duration in seconds).

6. **Filter long clips** — drops any audio clip 30 seconds or longer.

7. **Split train/eval** — first 450 examples for training, the remainder for evaluation.

8. **Load the model & freeze the encoder** — `WhisperForConditionalGeneration` loaded pretrained,
   with the encoder frozen except for the decoder's cross-attention (`encoder_attn`) layers, and the
   generation config pinned to English transcription.

9. **Custom data collator** — pads audio features and label sequences per batch, masks label padding
   with `-100` so it's ignored by the loss, and squeezes an extra dimension out of `input_features`.

10. **Load the WER metric** — word error rate via `evaluate`.

11. **Compute metrics** — reports both raw ("orthographic") WER and a normalized WER (using
    Whisper's `BasicTextNormalizer`), skipping empty references.

12. **Check GPU availability**:
    ```
    CUDA available: True
    Device name: Tesla T4
    ```

13. **Log in to the Hugging Face Hub** — authentication happens before the `Trainer` is built, since
    `push_to_hub=True` needs it at construction time, not just when `push_to_hub()` is called.

14–16. **Training arguments & Trainer** — `Seq2SeqTrainer` configured for 40 epochs, batch size 8,
    learning rate 5e-5, fp16, `predict_with_generate=True`, evaluating and pushing to
    `Atrac/my-seq2seq-model`. The processor is saved into the output directory alongside the model
    checkpoints so the full `WhisperProcessor` (feature extractor + tokenizer) ships with the repo.

17. **Train**:
    ```
    TrainOutput(global_step=2280, training_loss=0.0056,
                train_runtime=3107.4s, train_samples_per_second=5.79,
                epoch=40.0)
    ```
    Training loss drops close to zero over 40 epochs on only 450 training examples — a sign of
    heavy overfitting to the training set (see the evaluation result below).

18. **Push to the Hub**:
    ```
    CommitInfo(commit_url='https://huggingface.co/Atrac/my-seq2seq-model/commit/be76f0d7...',
               commit_message='End of training', ...)
    ```

19. **Sanity-check the uploaded model** — reloads it from the Hub, re-runs evaluation, and
    transcribes one real sample end-to-end:
    ```
    {'eval_loss': 0.584, 'eval_wer_ortho': 0.3835, 'eval_wer': 0.3819}
    Normalised WER: 0.3819
    Passing threshold: < 0.37

    Model transcription: I would like to set up a joint account with my partner
    Ground truth:         I would like to set up a joint account with my partner
    ```
    The single sanity-check transcription is a perfect match, but the **normalized WER (0.382)
    is above the course's 0.37 passing threshold** — despite near-zero training loss, the model
    doesn't generalize well to the held-out set, consistent with the overfitting noted above.

## Requirements

Python packages: `transformers`, `datasets`, `evaluate`, `accelerate`, `jiwer`, `soundfile`,
`librosa`, `huggingface_hub`. A CUDA GPU is required (`fp16=True` will fail on CPU); developed on a
single Tesla T4.

## Usage

Open the notebook and run the cells top to bottom. Replace `hub_model_id`/`MY_USERNAME` with your
own Hugging Face username, and log in via `notebook_login()` when prompted — do not hardcode a
token in the notebook.
