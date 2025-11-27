import pickle 
import numpy as np
import math
import matplotlib.pyplot as plt

#MRR: Focuses on the first relevant item retrieved. If multiple relevant items exist, 
# MRR still only considers the first occurrence.
def compute_mrr(relevance_list, ranks):
    reciprocal_ranks = []
    
    # Loop through the list to calculate reciprocal rank for relevant items.
    for idx, relevance in enumerate(relevance_list):
        if relevance == 1:
            reciprocal_ranks.append(1 / (idx + 1))  # rank is 1-based, so use (idx + 1)
            break  # Only the first relevant item is needed for MRR
    
    # If there is no relevant item, return 0 as reciprocal rank
    if not reciprocal_ranks:
        return 0.0
    
    # Mean Reciprocal Rank (MRR) is just the average of the reciprocal ranks
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Average Precision (AP): Measures how well all relevant items are retrieved and averaged over ranks.
# ideally relevant items (attacked) should be ranked higher
# if ranked higher after attack, the attack is successful
def compute_ap(relevance_list, retrieved_rank):
    """
    Calculate Average Precision (AP) for a ranking.

    Parameters:
    - ground_truth (list): Binary list indicating relevance (1 = relevant, 0 = not relevant).
    - retrieved_rank (list): List of item rankings (sorted in order of retrieval).

    Returns:
    - float: The Average Precision (AP).
    """
    relevant_count = 0  # Counter for relevant items found
    precision_at_k = []  # List to store precision at each relevant item
    
    for idx, relevance in enumerate(relevance_list):
        if relevance == 1:
            relevant_count += 1
            precision = relevant_count / (idx + 1)  # Precision at rank k (1-based index)
            precision_at_k.append(precision)
    
    if relevant_count == 0:
        return 0.0  # Return 0 if there are no relevant items
    
    # Average Precision is the mean of precision values at each relevant item
    return sum(precision_at_k) / relevant_count


def precision_recall_f1_at_k(relevance_list, ranked_list, k):
    """
    Calculate Precision@k, Recall@k, and F1@k.

    Parameters:
    - relevance_list (list): Binary list where 1 = relevant, 0 = not relevant.
    - ranked_list (list): List of indices representing the ranking order.
    - k (int): The cutoff rank.

    Returns:
    - tuple: (Precision@k, Recall@k, F1@k)
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Slice the relevance list to consider only the top-k items
    relevance_at_k = relevance_list[:k]

    # Calculate the number of relevant items retrieved at k
    relevant_retrieved_at_k = sum(relevance_at_k)

    # Total relevant items in the full list
    total_relevant = sum(relevance_list)

    # Precision@k: Relevant retrieved / Total retrieved (up to k)
    precision_at_k = relevant_retrieved_at_k / k

    # Recall@k: Relevant retrieved / Total relevant in the list
    recall_at_k = relevant_retrieved_at_k / total_relevant if total_relevant > 0 else 0.0

    # F1-score@k: Harmonic mean of Precision@k and Recall@k
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0.0

    return precision_at_k, recall_at_k, f1_at_k

    # Example usage:
    relevance_list = [1, 0, 1, 0, 1, 1, 0, 0, 1]
    k = 5
    precision, recall, f1 = calculate_precision_recall_f1_at_k(relevance_list, k)
    print(f"Precision@{k}: {precision:.3f}")
    print(f"Recall@{k}: {recall:.3f}")
    print(f"F1-score@{k}: {f1:.3f}")
    return 
    
    
    
model = "mistral_large_2" 
query_type = "abstract"
product = "coffee_machines"




attacks_mapping = [
    "social_proof_baseline",
    "attack_discount_framing",
    "authority_bias_baseline",
    "exclusivity",
    "attack_scarcity",

]    
    

def get_pair(d, da, i):
    return [row["se_output"] for row in d[i][4]], [row["se_output"] for row in da[i][4]]

def find_relevance(targeted_items, rank_list):
    relevance = []
    for i in rank_list:
        if i in targeted_items:
            relevance.append(1)
        else:
            relevance.append(0)
    return relevance

mrr_results_gt, mrr_results_att = {}, {}

for attack_type in attacks_mapping: 
    mrr_results_gt[attack_type] = []
    mrr_results_att[attack_type] = []
    
    for attacked_prd in range (10):

        type_of_control = "control_baseline" if "baseline" in attack_type else "control"


        with open(f"../outputs_rank_optimizer/{type_of_control}/experiment_{product}_{query_type}_{model}_{type_of_control}_{attacked_prd}.pickle", "rb") as handle:
            d = pickle.load(handle)

        with open(f"../outputs_rank_optimizer/{attack_type}/experiment_{product}_{query_type}_{model}_{attack_type}_{attacked_prd}.pickle", "rb") as handle:
            da = pickle.load(handle)



        targeted_items = [attacked_prd]
        mrr_gt, mrr_att = [], []
        for i in range (len(da)):
            (ground_truth, retrieved) = get_pair(d, da, i)

            gt_relevance = find_relevance(targeted_items, ground_truth)
            retrieved_relevance = find_relevance(targeted_items, retrieved)


            mrr = compute_mrr(gt_relevance, ground_truth)
            mrr_gt.append(mrr)


            mrr = compute_mrr(retrieved_relevance, retrieved) # higher value denotes attack was successful
            mrr_att.append(mrr)

        
        mrr_results_gt[attack_type].append(np.mean(mrr_gt))
        mrr_results_att[attack_type].append(np.mean(mrr_att))
        
#         print (f"Attacked Product: {attacked_prd}")
#         print(f"MRR ground truth: {np.mean(mrr_gt):.4f}")
#         print(f"MRR attacked: {np.mean(mrr_att):.4f}")
#         print ("------")




def plot_mrr_results(mrr_results_gt, mrr_results_att, max_cols=5):
    """
    Generate a grid of subplots for each baseline with MRR values 
    before and after the attack. Each row contains up to max_cols subplots.
    """
    
    # Get the baselines (dictionary keys)
    baselines = list(mrr_results_gt.keys())
    n_plots = len(baselines)
    
    # Calculate number of rows (up to max_cols plots per row)
    n_rows = math.ceil(n_plots / max_cols)
    
    # Create subplots
    fig, axes = plt.subplots(
        n_rows, 
        max_cols, 
        figsize=(5 * max_cols, 5 * n_rows), 
        sharey=True
    )
    
    plt.style.use('bmh')

    plt.rcParams.update({
        'font.size': 44,        # Default text size
        'axes.titlesize': 24,   # Axes title font size
        'axes.labelsize': 44,   # Axes label size
        'xtick.labelsize': 44,  # X-axis tick label size
        'ytick.labelsize': 44,  # Y-axis tick label size
        'legend.fontsize': 15   # Legend font size
    })

    
    # Flatten axes for simpler indexing if there are multiple rows
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Loop through each baseline
    for i, baseline in enumerate(baselines):
        # Define custom x-axis labels, mixing numbers and strings
        x_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        x_values = range(1, len(mrr_results_gt[baseline]) + 1)
        
        # Plot bars for "before attack" and "after attack"
        axes[i].bar(
            [x - 0.2 for x in x_values],
            mrr_results_gt[baseline],
            width=0.4,
            label='Before Attack'
        )
        axes[i].bar(
            [x + 0.2 for x in x_values],
            mrr_results_att[baseline],
            width=0.4,
            label='After Attack'
        )
        
        # Add intermediate steps for y-axis ticks
        y_min, y_max = axes[i].get_ylim()  # Get current y-axis limits
        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]  # 5 steps
        axes[i].set_yticks(y_ticks)  # Set custom y-axis ticks
        
        # Adjust tick label font sizes
        axes[i].tick_params(axis='x', labelsize=16)  # X-axis tick font size
        axes[i].tick_params(axis='y', labelsize=16)  # Y-axis tick font size

        
        # Replace underscores in baseline names with spaces for the title
        title = baseline.replace('_', ' ')
        
        # Set custom x-axis labels
        
        axes[i].set_xticks(x_values)
        axes[i].set_xticklabels(x_labels[:len(x_values)])
        
        # Label and title
        ttt = title.replace("attack", "").replace("baseline", "").strip()
        axes[i].set_title(ttt.capitalize())
        if i > 5:
            axes[i].set_xlabel('Product ID', fontsize=22)
            
        if i % 3 == 0:
            axes[i].set_ylabel('MRR', fontsize=22)

           
        fig.set_facecolor('white')  # Set the figure background to white
        
#         axes[i].legend(
#         loc='upper left',  # Position in the top-left corner
#         ncol=2,            # Display legend items in a single row
#         bbox_to_anchor=(0, 1.1),  # Offset the legend outside the plot
#         frameon=False      # Remove the legend border
#     )
#         axes[i].legend()
    
    # Hide any extra subplots if you have fewer baselines than max_cols * n_rows
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    for i in range (len(axes)):
        axes[i].set_facecolor('white')  # Set the axes background to white

    
    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    plt.style.use('bmh')
    plt.savefig(f"./evaluation_imgs/mrr_{model}_{query_type}_{product}_v2.png")

# Example data
    
plt.style.use('bmh')
plot_mrr_results(mrr_results_gt, mrr_results_att)
