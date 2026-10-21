# MusicDPO

This project fine-tunes
[`facebook/musicgen-small`](https://huggingface.co/facebook/musicgen-small)
with preference pairs using LoRA and Direct Preference Optimization (DPO).
The training objective combines:

- DPO loss between preferred and rejected audio tokens; and
- a contrastive CLAP loss between the text prompt and generated audio.

## Workflow

The training workflow is shown below.

![MusicDPO workflow](./figure/model.png)

## Setup

Install the dependencies listed by running:

```bash
pip install -r requirements.txt
```

Download the [training data and CLAP checkpoint](#required-files), or obtain
the corresponding files from the project distribution. The pretrained
MusicGen, EnCodec, and CLAP models are downloaded automatically from Hugging
Face on first use.

Run commands from the `MusicDPO/code/`. GPU execution is recommended
because the trainer uses 8-bit model loading and the paged optimizer through
`bitsandbytes`.

Please note that:

- The pretrained models are downloaded automatically on first use:
  `facebook/musicgen-small`, `facebook/encodec_32khz`, and
  `laion/clap-htsat-unfused`.
- `clap_model_epoch_29.pth` is a local fine-tuned CLAP checkpoint and must be
  placed in the project directory before training.

## Required files

Place the training data and CLAP checkpoint in the project directory:

```text
MusicDPO/
├── code/
│   ├── trainer.py
│   ├── trainer_wo_CLAP.py
│   ├── inference.py
│   └── train.sh
├── music_label_genre.csv
├── train_segments_25sec/
│   └── *.wav
├── output_wav/
│   └── *.wav
└── clap_model_epoch_29.pth
```

The `train_segments_25sec/` and `output_wav/` directories are required by
`code/trainer.py`. The `tokenized_outcome/` and checkpoint directories are
created during training and do not need to exist beforehand.

The CSV must contain at least:

```text
id,prompt
```

For each row, `trainer.py` searches for:

- preferred audio: `train_segments_25sec/{id:04d}_*.wav`
- rejected audio: `output_wav/{prompt}_*.wav`

Audio is resampled to 32 kHz. Only preferred/rejected pairs with the expected
waveform lengths are retained by the current script.

## Training

Start DPO fine-tuning with the default configuration from the project root:

```bash
python code/trainer.py
```

The default training settings are:

```text
epochs:              1
batch size:          4
gradient accumulation: 8
learning rate:       2e-4
DPO beta:            0.1
LoRA rank:           8
LoRA alpha:          8
LoRA dropout:        0.05
```

The trainer:

1. loads preferred/rejected audio and text prompts;
2. converts audio to EnCodec tokens;
3. tokenizes the prompts;
4. applies LoRA adapters to MusicGen query/value projections;
5. computes DPO and CLAP contrastive losses; and
6. saves the adapter after each epoch.

Checkpoints are written as:

```text
checkpoints/
└── musicgen-dpo_epoch_1/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

To train for more epochs or choose another output directory:

```bash
python code/trainer.py \
  --epochs 10 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --lr 2e-4 \
  --save_model_path ./checkpoints/musicgen-dpo \
  --clap_ckpt ./clap_model_epoch_29.pth
```

The first run creates a `tokenized_outcome/` directory containing cached text
and audio-token tensors. Subsequent runs can reuse these tensors when the cache
files match the trainer's expected names and preprocessing configuration.

Optional Weights & Biases logging:

```bash
python code/trainer.py \
  --use_wandb \
  --wandb_project musicgen-dpo \
  --wandb_run_name musicgen-dpo-small
```

## Inference

Generate audio from the fixed prompt in `code/inference.py` and compare the
fine-tuned adapter with the original MusicGen model:

```bash
python code/inference.py \
  --peft_model ./checkpoints/musicgen-dpo_epoch_1 \
  --base_model facebook/musicgen-small \
  --output_dir ./output
```

The script uses the prompt:

```text
Solo Violin in D major with Baroque style
```

It writes both generations to:

```text
output/
├── musicgen_out.wav       # fine-tuned MusicGen with the LoRA adapter
└── musicgen_ft_out.wav    # base MusicGen output
```

The output sample rate is read from the MusicGen configuration. Use the
generated WAV files to compare the preference-trained model with the base
model.

## Reproducing the provided result

From `MusicDPO/`:

```bash
python code/trainer.py \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --save_model_path ./checkpoints/musicgen-dpo

python code/inference.py \
  --peft_model ./checkpoints/musicgen-dpo_epoch_1 \
  --base_model facebook/musicgen-small \
  --output_dir ./output
```

Exact audio output can vary because inference uses sampling
(`do_sample=True`). The training and generation results also depend on the
versions of PyTorch, Transformers, PEFT, CUDA, and the downloaded model
weights.
