import numpy as np
import pickle 
import matplotlib.pyplot as plt


attacks_mapping = [
        "social_proof",
        "exclusivity", 
        "attack_scarcity", 
        "attack_discount_framing", 
        "contrast_effect", 
        "decoy_effect", 
        "authority_bias",
        "bizarreness_effect",
        "denominator_neglect",
        "storytelling_effect",
        "identity_signaling",
    
        "social_proof_baseline",
        "exclusivity_baseline",
        "attack_scarcity_baseline",
        "attack_discount_framing_baseline",
        "bizarreness_effect_baseline",
        "contrast_effect_baseline",
        "decoy_effect_baseline",
        "authority_bias_baseline",
        "identity_signaling_baseline",
    
]

def is_recommended(d, item):
    s = []
    for row in d:
        row = row[4]
        row = [r["se_output"] for r in row]
        s.append(item in row)
    return np.sum(s)/len(s)
  
    
def is_first(d, item):
    s = []
    for row in d:
        row = row[4]
        row = [r["se_output"] for r in row]
        
        if item in row:


            if row[0] == item:
                s.append(True)
            else:
                s.append(False)
        else:
            s.append(False)
    return np.sum(s)/len(s)


query_type = "abstract"
product = "coffee_machines"
 


num_of_is_rec_after = {}

for model in ["llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "claude_3_5_sonnet_v2", "mistral_large_2"]:
    num_of_is_rec_after[model] = {}
    for attack_type in attacks_mapping:
        
        isrecafter = []
        for attacked_prd in range (10):

            type_of_control = "control_baseline" if "baseline" in attack_type else "control"
            
            if model == "mistral_large_2":
                with open(f"../outputs_rank_optimizer/{type_of_control}/experiment_{product}_{query_type}_{model}_{type_of_control}_{0}.pickle", "rb") as handle:
                    d = pickle.load(handle)
            else:
                with open(f"../outputs_rank_optimizer/{type_of_control}/experiment_{product}_{query_type}_{model}_{type_of_control}_{attacked_prd}.pickle", "rb") as handle:
                    d = pickle.load(handle)

            with open(f"../outputs_rank_optimizer/{attack_type}/experiment_{product}_{query_type}_{model}_{attack_type}_{attacked_prd}.pickle", "rb") as handle:
                da = pickle.load(handle)

            before_attack, after_attack = [], []
            for i in range (10):
                before_attack.append(is_recommended(d, i))
                after_attack.append(is_recommended(da, i))

            if (np.argmax(after_attack) == attacked_prd and np.argmax(before_attack) != attacked_prd):
                isrecafter.append(attacked_prd)


        num_of_is_rec_after[model][attack_type] = len(isrecafter)

        
plt.style.use('bmh')
        
        
plt.rcParams.update({
        'font.size': 18,        # Default text size
        'axes.titlesize': 18,   # Axes title font size
        'axes.labelsize': 18,   # Axes label size
        'xtick.labelsize': 18,  # X-axis tick label size
        'ytick.labelsize': 18,  # Y-axis tick label size
        'legend.fontsize': 12   # Legend font size
    })


data_dict = num_of_is_rec_after
# plt.style.use('classic')

# Step 1: Gather all unique biases
all_biases = sorted({
    key for model_values in data_dict.values() 
    for key in model_values.keys()
})

# Step 2: Filter out biases that have a total count of 0 across all models
filtered_biases = []
for bias in all_biases:
    total_count = sum(data_dict[model].get(bias, 0) for model in data_dict)
    if total_count != 0:  # Keep bias if total is not zero
        filtered_biases.append(bias)
        
order_biases = []
for att in attacks_mapping:
    if att in filtered_biases:
        order_biases.append(att)
        
filtered_biases = order_biases.copy()

# X positions for each bias
x_positions = np.arange(len(filtered_biases))

# Number of models
models = list(data_dict.keys())
num_models = len(models)

# Define a color map for the models
# colors = plt.cm.tab10(np.linspace(0, 1, num_models))

# Decide on the width of each bar group
bar_width = 1.0 / (num_models + 1)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Step 3: Plot each model's data
for i, model in enumerate(models):
    # Retrieve y-values for the current model from the filtered biases
    y_values = [data_dict[model].get(bias, 0) for bias in filtered_biases]

    # Offset for this model's bars
    offset = (i - num_models / 2) * bar_width + bar_width / 2
    
    ax.bar(
        x_positions + offset,
        y_values,
        width=bar_width,
        label=model.replace("3.1", "").replace("_large_2", "").replace("_3_5_sonnet_v2", "3.5").capitalize(),
#         color=colors[i],
#         edgecolor='black'
    )

for i in range (len(filtered_biases)):
    plt.axvline(i + 0.5, color='lightgray', linestyle='--', dashes=(1, 2))


fig.set_facecolor('white')  # Set the figure background to white
ax.set_facecolor('white')  # Set the axes background to white


# Step 4: Customize plot appearance
ax.set_xticks(x_positions)
ax.set_xticklabels([x.replace("_", " ").replace("attack", "").strip().capitalize().replace(" baseline", "$_{exp}$ ") for x in filtered_biases], rotation=45, ha='right')
ax.set_ylabel('Number of Products')
# ax.set_xlabel('Biases')
# ax.set_title('Bias Counts byx Model')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
# plt.show()
plt.savefig(f"./evaluation_imgs/count_most_recommended_product_before_after_{query_type}_{product}_mistral_max.png")

