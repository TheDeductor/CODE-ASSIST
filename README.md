### AI Disclosure

AI tools, including Claude and Gemini, were used as learning and development assistants. This was not a vibe-coded project; I actively implemented, debugged, and learned through the process, using AI mainly for explanations, troubleshooting, and exploring approaches.

# CODE-ASSIST
Model weights (GGUF): (https://huggingface.co/pranav9999/qwen25coder-student-tutor-gguf/tree/main)


Colab-Notebook Link: (https://colab.research.google.com/drive/1sHqcZhzEb7HfRgpJyPLQ2_7iGk7usO-b?usp=sharing)

A fine-tuned programming assistant for engineering students.

`code-assist` helps students understand programming concepts, debug code, and work through coursework-style questions. The project fine-tunes the open-source Qwen2.5-Coder-3B model with QLoRA on a free Google Colab T4 GPU.

> Status: The training notebook and evaluation results are included. The Gradio demo and model download links can be added when they are available.

## Project Objective

This project was created to build a small AI assistant that can:

- Answer programming and learning-related questions
- Explain code and programming concepts clearly
- Help students identify and fix errors
- Provide step-by-step guidance instead of only returning code
- Avoid confidently inventing answers about clearly nonexistent APIs

## Main Results

The base model and fine-tuned model were evaluated on the same held-out test set of 120 examples.

| Metric | Base model | Fine-tuned model |
| --- | ---: | ---: |
| Test-set perplexity | 5.91 | **1.84** |
| Mean ROUGE-L on a 12-example sample | approximately 0.264 | **approximately 0.374** |

The fine-tuned model produced answers that were more similar to the expected tutoring responses and showed a more patient, instructional style. In the manual comparison, it also handled clearly fake API names more cautiously than the base model.

These results should be interpreted carefully. ROUGE-L was calculated on a small sample, and lower perplexity does not prove that every generated answer is factually correct.

## Model

The base model is [`unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit).

It was selected because it is a relatively small coding model, supports 4-bit loading, and is practical to fine-tune on a free-tier GPU. Its standard transformer architecture also avoided the hardware compatibility issue encountered with Qwen3.5.

## Dataset

The dataset combines programming instruction, debugging, code review, and tutoring examples from the following sources:

- CodeFeedback for code review and debugging-style questions
- Magicoder for general programming instruction
- Python Alpaca for Python-focused questions
- Hand-written examples for the tutoring style, coursework guidance, explanations, and abstention on fake APIs

The final dataset contains 2,009 examples:

- Training: 1,769 examples
- Validation: 120 examples
- Test: 120 examples

Each example is formatted as a `messages` array containing `system`, `user`, and `assistant` messages. The examples were tagged by source and task category. The categories include general programming, debugging, concepts, coursework, style, and abstention.

The test set was kept separate from training so that it could be used for an independent comparison of the base and fine-tuned models.

## Data Preparation

The notebook prepares the data for supervised fine-tuning by:

1. Combining the selected programming and tutoring examples.
2. Removing or excluding unsuitable records.
3. Standardizing each record into the same chat-style `messages` format.
4. Adding source and task-category labels for analysis.
5. Splitting the data into training, validation, and test sets.
6. Applying the model chat template during training and inference.

The prepared files are stored as JSONL files:

```text
data/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

## Fine-Tuning

The model was fine-tuned with QLoRA using Unsloth. The base model was loaded in 4-bit precision, while LoRA adapters were trained on top of the frozen base weights.

| Setting | Value |
| --- | --- |
| Method | QLoRA |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0 |
| Target layers | Attention and MLP projection layers |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Scheduler | Linear |
| Warmup steps | 5 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Training hardware | Google Colab T4 GPU |
| Approximate training time | 40 to 45 minutes |

The notebook selects the precision based on GPU support. On the T4 used for this project, training was performed with `fp16` and not `bf16`.

## Important GPU Problem and Solution

At first, Qwen3.5 was used for the project. During setup, training failed with a dtype error involving `bf16` and `fp16`.

The free Google Colab GPU was an NVIDIA T4. The T4 does not provide the bf16 support required by that setup. As a result, the model could not safely combine its internal bf16 operations with the fp16 operations used elsewhere in the training process.

The project was changed to Qwen2.5-Coder-3B. The final setup used 4-bit loading to reduce VRAM usage and fp16 for computation. This resolved the problem and allowed training to complete on the T4.

The following check was used to inspect bf16 support:

```python
import torch

print(torch.cuda.is_bf16_supported())
```

On the T4 used for this project, this should normally print:

```text
False
```

This was a hardware and model compatibility issue, not simply a missing training argument. It is an important part of the project because model selection must consider both memory requirements and the numerical formats supported by the available GPU.

## Evaluation

The models were tested on the same 120 examples from `test.jsonl`. The evaluation included:

- Test-set loss and perplexity
- ROUGE-L comparison against reference answers
- Manual comparison of base-model and fine-tuned responses

The fine-tuned model improved the measured results. It also showed a more consistent tutoring tone and was more likely to refuse a question about a clearly nonexistent API.

The improvement is not a guarantee of factual accuracy. The fine-tuned model can still produce incorrect supporting details, especially for topics that were not strongly represented in the dataset. This is a known limitation of fine-tuning a small model on a narrow dataset.

## Interface

A simple Gradio `ChatInterface` is included in the notebook. It allows a user to ask programming questions and receive responses from the fine-tuned model.

Demo links:

- Demo video: `Add link here`
- Temporary Gradio demo: `Add link here if still active`
- Fine-tuned model weights: `Add Hugging Face link here if published`

The temporary Gradio link is session-based and may stop working when the Colab runtime ends.


## How to Reproduce

1. Open `notebook/Student_Programming_Assistant_FineTuning.ipynb` in Google Colab.
2. Select a T4 GPU under Runtime, Change runtime type.
3. Upload `train.jsonl`, `val.jsonl`, and `test.jsonl` from the `data` folder.
4. Run the notebook cells from top to bottom.
5. Review the evaluation outputs.
6. Launch the Gradio interface from the final notebook cells.
7. Optionally export the trained model to GGUF for local inference.

Training takes approximately 40 to 45 minutes on a free-tier T4, although the exact time depends on the Colab environment.

## Optional Local Inference

If a GGUF file is exported, it can be used with Ollama or llama.cpp. For Ollama, create a `Modelfile` that points to the exported model and run:

```bash
ollama create code-assist -f Modelfile
ollama run code-assist
```

The GGUF file and a working `Modelfile` are not included unless they are added to the repository separately.

## Limitations

- The dataset is mainly Python-focused, so C++ and Java coverage is more limited.
- The training dataset is small compared with the datasets used for large language models.
- Hallucination reduction is partial. The model may still state incorrect details in an otherwise useful answer.
- No retrieval-augmented generation, persistent conversation memory, or permanent deployment is included in the current version.
- The Gradio demo is temporary unless it is deployed separately.

## Acknowledgements

This project uses Qwen2.5-Coder, Hugging Face tools, PEFT/LoRA concepts, Unsloth, PyTorch, and Gradio. Please follow the license and usage terms of each dependency and model.
