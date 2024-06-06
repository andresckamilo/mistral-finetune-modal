# Mistral Modal Notebook

This repository provides a Python script to set up a Jupyter Lab server for fine-tuning Mistral models using Modal. 

## Features

- **Supported Models:** Easily switch between multiple pre-configured Mistral models.
- **GPU Support:** Configure GPU settings for enhanced performance.
- **Volume Management:** Automatically handle volume creation and seeding with model data.
- **Jupyter Notebook:** Seamlessly run a Jupyter Notebook server with custom authentication.

## Requirements

- **Modal:** Ensure you have Modal installed and configured with the necessary secrets.

## Setup

1. **Clone the Repository**
    ```bash
    git clone https://github.com/andresckamilo/mistral-finetune.git
    cd mistral-finetune
    ```

2. **Configure Modal** 
    Ensure you have the required Modal secrets and configurations set up. This script requires a Modal account.

3. **Update Jupyter Token**
    Update the `JUPYTER_TOKEN` variable in the script to a non-guessable token for Jupyter Notebook authentication.

## Usage

### Running the Application

To run the application, use the command:
```bash
modal run src.main # Default timeout is 6 hours
```

### Supported Models

The script supports the following models:
- "7B Base V3"
- "7B Instruct V3"
- "8x7B Instruct V1"
- "8x22B Instruct V3"
- "8x22B Base V3"

You can change the model by modifying the `model` variable in the script.

## Mistral Model Fine-tuning and LoRA Adapter Upload

Inside the Modal Jupyter server, there is a notebook called `mistral_finetune.ipynb`. This notebook contains scripts and instructions for fine-tuning a Mistral model using the mistral-finetune library and uploading the resulting LoRA adapters to Hugging Face. The structure of the notebook is as follows:

### Table of Contents

1. **Introduction**
2. **Prepare Dataset**
3. **Finetuning**
4. **Inference**

### Introduction

This notebook builds on the work done by the Mistral team, providing a streamlined way to fine-tune Mistral models using Modal. The scripts included handle dataset preparation, model training, and inference.

### Prepare Dataset

To ensure effective training, mistral-finetune has strict requirements for how the training data must be formatted. Check out the required data formatting [here](link_to_data_formatting).

In this example, we use the ultrachat_200k dataset. We load a chunk of the data into Pandas DataFrames, split the data into training and validation sets, and save the data into the required jsonl format for fine-tuning. All datasets and formatting scripts come from the finetuning repository of the Mistral team.

### Finetuning

The fine-tuning process is facilitated by a configuration file that specifies various training parameters and paths to the dataset and model. Ensure the configuration is properly set up before starting the training process.

### Inference

After fine-tuning, you can load the model and use it for generating text. The script provided an example how to perform inference using the fine-tuned model and LoRA adapters.

### Example Inference Script

```python
from mistral_inference.model import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


tokenizer = MistralTokenizer.from_file("mistral_models/mistral-7b-v0-3/tokenizer.model.v3")  # change to extracted tokenizer file
model = Transformer.from_folder("mistral_models/mistral-7b-v0-3")  # change to extracted model dir
model.load_lora("output/mistral-7b-v0-3/checkpoints/checkpoint_000100/consolidated/lora.safetensors")

completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
```

## License

This repository is licensed under the Apache 2.0 License.

## Contact

For any questions or issues, please open an issue on the GitHub repository.
