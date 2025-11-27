# Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations

**Accepted at EMNLP 2025 (Main Conference)** | [📄 Paper PDF](https://aclanthology.org/2025.emnlp-main.1140.pdf)

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

## Citation
If you use this code or find our work helpful, please cite our paper:

```bibtex
@inproceedings{filandrianos-etal-2025-bias,
    title = "Bias Beware: The Impact of Cognitive Biases on {LLM}-Driven Product Recommendations",
    author = "Filandrianos, Giorgos  and
      Dimitriou, Angeliki  and
      Lymperaiou, Maria  and
      Thomas, Konstantinos  and
      Stamou, Giorgos",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1140/",
    doi = "10.18653/v1/2025.emnlp-main.1140",
    pages = "22408--22437",
    ISBN = "979-8-89176-332-6",
    abstract = "The advent of Large Language Models (LLMs) has revolutionized product recommenders, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making such manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive evaluation across models of varying scale, we find that certain biases, such as social proof, consistently boost product recommendation rate and ranking, while others, like scarcity and exclusivity, surprisingly reduce visibility. Our results demonstrate that cognitive biases are deeply embedded in state-of-the-art LLMs, leading to highly unpredictable behavior in product recommendations and posing significant challenges for effective mitigation."
}
```

