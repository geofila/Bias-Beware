# Code for "Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations"

## Overview
This repository hosts the implementation code for the attack and evaluation methods discussed in the research paper, "Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations."


![Logo! Bias Beware!](logo.png)
## Installation
Ensure all dependencies are installed prior to running these scripts. Dependencies can be installed using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## Usage
Executing the scripts provided in this repository involves a two-step process:
1. **Run the attack scripts** to generate adversarial perturbations.
2. **Execute the evaluation scripts** to analyze the effects of these perturbations.

### Step 1: Running the Attacks
Use the following commands in your terminal to run the attack scripts for each model:

- **Claude Model**:
  ```bash
  python run_attack_control-claude.py --model_name 'claude_3_5_sonnet_v2' --attack 'your_attack_name' --aws_keys_csv_filename 'path_to_your_csv_file'
  ```

- **Mistral 2 Large Model**:
  ```bash
  python run_attack_control-mistral.py --model_name 'mistral_large_2' --attack 'your_attack_name' --aws_keys_csv_filename 'path_to_your_csv_file'
  ```

- **Llama3.1 Models**:
  ```bash
  # For 8b Model
  python run_attack_control-llama.py --model_name 'llama3.1-8b' --attack 'your_attack_name' --aws_keys_csv_filename 'path_to_your_csv_file' --run_control True

  # For 70b Model
  python run_attack_control-llama.py --model_name 'llama3.1-70b' --attack 'your_attack_name' --aws_keys_csv_filename 'path_to_your_csv_file' --run_control True

  # For 405b Model
  python run_attack_control-llama.py --model_name 'llama3.1-405b' --attack 'your_attack_name' --aws_keys_csv_filename 'path_to_your_csv_file' --run_control True
  ```

Setting the `--run_control` flag to `True` does not implement an attack but shuffles the product rankings to create a control dataset.

These scripts will generate a folder containing the data for each model and attack, saved as `.pickle` files. Each file consists of a list of 100 sublists, each corresponding to a different attack, detailing original data, shuffled products, model input and output, and rankings after applying the verbalizer.


For the defense, just change the `system_prompt` in the `utils.py` file and re-run the attacks.

### Step 2: Running the Evaluation
After generating the attack data, proceed with the evaluation using the following scripts:

```bash
cd evaluation
python table2.py
python mrr.py
python fig3.py
```

## Notes
- Due to file size constraints on OpenReview, the complete results of the attacks, which exceed 200MB each, are not included in the supplementary materials.
- For access to the full attack results, please generate the results using the provided scripts.

## Acknowledgment
The datasets used in this work were obtained from [GitHub - aounon/llm-rank-optimizer](https://github.com/aounon/llm-rank-optimizer/tree/main/data) and [Hugging Face Datasets](https://huggingface.co/datasets/Studeni/AMAZON-Products-2023). Extended product descriptions used in this study are available in the data folder.

