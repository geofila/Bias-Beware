import os
import pickle
from scipy.stats import ttest_ind, ttest_rel
import numpy as np
import math

import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from analysis import plot_results_pos, plot_dict_with_specific_ordering, analyze_results, plot_dict_with_errors



attacks_mapping = [
        "social_proof_baseline",
        "exclusivity_baseline",
        "attack_scarcity_baseline",
        "attack_discount_framing_baseline",
        "bizarreness_effect_baseline",
        "contrast_effect_baseline",
        "decoy_effect_baseline",
        "authority_bias_baseline",
        "identity_signaling_baseline",

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
        "storytelling_effect",
        "denominator_neglect"
]




def analyse_results(product, attacks_mapping):
    dict_p_lt_005_ordered = {}
    dict_rec_af_bef_ordered = {}
    dict_p_rec_af_bef_ordered = {}

    dict_p_pos_005_ordered = {}
    dict_pos_af_bef_ordered = {}
    # dict_pos_rec_af_bef_ordered = {}
    dict_p_pos_af_bef_ordered = {}


    for model in ["llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "claude_3_5_sonnet_v2", "mistral_large_2"]: #mistral_large_2
        for attack_type in attacks_mapping: #["attack_scarcity", "exclusivity_baseline", "identity_signaling_baseline", "authority_bias", "social_proof_baseline", "social_proof"]:
            query_type = "abstract"
#             product = "coffee_machines"

            type_of_control = "control_baseline" if "baseline" in attack_type else "control"


            product_ids, percent_before, percent_after, significance_per = [], [], [], []
            pos_before, pos_after, significance_pos = [], [], []
            for attacked_prd in range (10):

                if model == "mistral_large_2":
                    with open(f"../outputs_rank_optimizer/{type_of_control}/experiment_{product}_{query_type}_{model}_{type_of_control}_{0}.pickle", "rb") as handle:
                        d = pickle.load(handle)
                else:
                    with open(f"../outputs_rank_optimizer/{type_of_control}/experiment_{product}_{query_type}_{model}_{type_of_control}_{attacked_prd}.pickle", "rb") as handle:
                        d = pickle.load(handle)

                acc1 = []
                pos1 = []
                for row in d:
                    prd = [p['se_output'] for p in row[4]]
                    acc1.append(int (attacked_prd in prd))
                    if attacked_prd in prd:
                        pos1.append(prd.index(attacked_prd))

                with open(f"../outputs_rank_optimizer/{attack_type}/experiment_{product}_{query_type}_{model}_{attack_type}_{attacked_prd}.pickle", "rb") as handle:
                    da = pickle.load(handle)

                acc2 = []
                pos2 = []
                for row in da:
                    prd = [p['se_output'] for p in row[4]]
                    acc2.append(int (attacked_prd in prd))
                    if attacked_prd in prd:
                        pos2.append(prd.index(attacked_prd))

                _, p = ttest_ind(acc1, acc2)
                if p < 0.05: #z_test(acc1, acc2) < 0.05:
                    st = "Yes"
                else:
                    st = "No"

                _, p = ttest_ind(pos1, pos2)
                if p < 0.05: #z_test(pos1, pos2) < 0.05:
                    st1 = "Yes"
                else:
                    st1 = "No"

    #             print (f"{attacked_prd}\t{int (sum(acc1)/len(acc1)*100)}\t{int (sum(acc2)/len(acc2) * 100)}\t{st}\t{round(np.mean(pos1), 2)} ± {round(np.std(pos1), 2)}\t{round(np.mean(pos2), 2)} ± {round(np.std(pos2), 2)}\t{st1}")

                product_ids.append(attacked_prd)
                percent_before.append(int (sum(acc1)/len(acc1)*100))
                percent_after.append(int (sum(acc2)/len(acc2) * 100))
                significance_per.append(st)

                pos_before.append(round(np.mean(pos1), 2))
                pos_after.append(round(np.mean(pos2), 2))
                significance_pos.append(st1)

            number_of_sign, mean_change, std_change, mean_change_sign, std_change_sign = analyze_results(product_ids, percent_before, percent_after, significance_per)
            
            attack_type = attack_type.replace("_baseline", "")
            if attack_type not in dict_p_lt_005_ordered:
                dict_p_lt_005_ordered[attack_type] = {}
                dict_rec_af_bef_ordered[attack_type] = {}
                dict_p_rec_af_bef_ordered[attack_type] = {}
                dict_p_pos_005_ordered[attack_type] = {}
                dict_pos_af_bef_ordered[attack_type] = {}
                dict_p_pos_af_bef_ordered[attack_type] = {}

            dict_p_lt_005_ordered[attack_type][model] = number_of_sign
            dict_rec_af_bef_ordered[attack_type][model] = (mean_change, std_change)
            dict_p_rec_af_bef_ordered[attack_type][model] = (mean_change_sign, std_change_sign)

            number_of_sign, mean_change, std_change, mean_change_sign, std_change_sign = analyze_results(product_ids, pos_before, pos_after, significance_pos)
            dict_p_pos_005_ordered[attack_type][model] = number_of_sign
            dict_pos_af_bef_ordered[attack_type][model] = (mean_change, std_change)
            dict_p_pos_af_bef_ordered[attack_type][model] = (mean_change_sign, std_change_sign)

    #         number_of_sign, mean_change, std_change, mean_change_sign, std_change_sign = analyze_results(product_ids, pos_before, pos_after, significance_pos)
    
    return {
        "dict_p_lt_005_ordered": dict_p_lt_005_ordered, 
        "dict_rec_af_bef_ordered": dict_rec_af_bef_ordered,
        "dict_p_rec_af_bef_ordered": dict_p_rec_af_bef_ordered,
        "dict_p_pos_005_ordered": dict_p_pos_005_ordered,
        "dict_pos_af_bef_ordered": dict_pos_af_bef_ordered,
        "dict_p_pos_af_bef_ordered": dict_p_pos_af_bef_ordered
    }




def format_float(val, decimal_places=2, placeholder="-"):
    """
    Safely format a float or returns a placeholder for NaN/None.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return placeholder
    return f"{val:.{decimal_places}f}"

def format_tuple(val_tuple, decimal_places=2, placeholder="-"):
    """
    Safely format a (mean, std) tuple or returns a placeholder for NaN/None.
    E.g., (5.13, 6.27) -> "5.13 ± 6.27"
    """
    if (
        val_tuple is None
        or len(val_tuple) < 2
        or any(math.isnan(x) for x in val_tuple)
    ):
        return placeholder
    mean_val = f"{val_tuple[0]:.{decimal_places}f}"
    std_val = f"{val_tuple[1]:.{decimal_places}f}"
    return f"{mean_val}/{std_val}"

def generate_latex_table(baseline_res, gen_res, baseline_res2, gen_res2):
    # 1) Collect all biases and models in a consistent order
    biases = list(baseline_res["dict_p_lt_005_ordered"].keys())
    
    # Collect every model from the sub-dictionaries
    all_models = set()
    for bias in biases:
        all_models.update(baseline_res["dict_p_lt_005_ordered"][bias].keys())
    # Convert to a list for stable ordering (or sort if you prefer)
    all_models = list(all_models)
    all_models = ["llama3.1-8b", "llama3.1-70b", "llama3.1-405b", "claude_3_5_sonnet_v2", "mistral_large_2"]
    # 2) Build the LaTeX table header
    latex_lines = []
    latex_lines.append(r"\begin{table*}[!ht]")
    latex_lines.append(r"\small")
    latex_lines.append(
        r"\begin{tabular}{c|c|ccc|ccc}"
    )
    latex_lines.append(r"\toprule")
    latex_lines.append(
        r"Bias & Model & \#p \textless 0.05 & \%Rec. aft-bef & p: \%Rec. aft-bef"
        r" & \#p \textless 0.05 & Pos. aft-bef & p: Pos. aft-bef \\ \midrule"
    )

    # 3) Populate rows
    for bias in biases:
        # We'll create one table-row per model
        first_model = True
        for model in all_models:
            # Extract values (some might not exist in every dictionary → use .get)
            # 3.1) #p < 0.05
            p_lt_005_val = f'{baseline_res["dict_p_lt_005_ordered"][bias].get(model, "-")}\\textbackslash{gen_res["dict_p_lt_005_ordered"][bias].get(model, "-")}'
            p_lt_005_val2 = f'{baseline_res2["dict_p_lt_005_ordered"][bias].get(model, "-")}\\textbackslash{gen_res2["dict_p_lt_005_ordered"][bias].get(model, "-")}'

            # 3.2) %Rec. aft-bef
#             rec_val = baseline_res["dict_rec_af_bef_ordered"][bias].get(model, None)
                 
#             rec_str = format_tuple(rec_val, decimal_places=2, placeholder="-")

            # 3.3) p<0.05: %Rec. aft-bef
            p_rec_val = baseline_res["dict_p_rec_af_bef_ordered"][bias].get(model, None)
            p_rec_str = f'{list(baseline_res["dict_p_rec_af_bef_ordered"][bias].get(model, "-"))[0]}\\textbackslash{list(gen_res["dict_p_rec_af_bef_ordered"][bias].get(model, "-"))[0]}'  
            p_rec_str2 = f'{list(baseline_res2["dict_p_rec_af_bef_ordered"][bias].get(model, "-"))[0]}\\textbackslash{list(gen_res2["dict_p_rec_af_bef_ordered"][bias].get(model, "-"))[0]}'
#             p_rec_str = format_tuple(p_rec_val, decimal_places=2, placeholder="-")

            # 3.4) p<0.05 (# of positives)
            p_pos_005_val = f'{baseline_res["dict_p_pos_005_ordered"][bias].get(model, "-")}\\textbackslash{gen_res["dict_p_pos_005_ordered"][bias].get(model, "-")}'
            p_pos_005_val2 = f'{baseline_res2["dict_p_pos_005_ordered"][bias].get(model, "-")}\\textbackslash{gen_res2["dict_p_pos_005_ordered"][bias].get(model, "-")}'

#             # 3.5) Pos. aft-bef
#             pos_val = baseline_res["dict_pos_af_bef_ordered"][bias].get(model, None)
#             pos_str = format_tuple(pos_val, decimal_places=2, placeholder="-")

            # 3.6) p<0.05: Pos. aft-bef
            p_pos_val = baseline_res["dict_p_pos_af_bef_ordered"][bias].get(model, None)
            p_pos_str = f'{list(baseline_res["dict_p_pos_af_bef_ordered"][bias].get(model, "-"))[0]}\\textbackslash{list(gen_res["dict_p_pos_af_bef_ordered"][bias].get(model, "-"))[0]}'
            p_pos_str2 = f'{list(baseline_res2["dict_p_pos_af_bef_ordered"][bias].get(model, "-"))[0]}\\textbackslash{list(gen_res2["dict_p_pos_af_bef_ordered"][bias].get(model, "-"))[0]}'
#             p_pos_str = format_tuple(p_pos_val, decimal_places=2, placeholder="-")

            # Construct each row
            if first_model:
                # Multirow for the first column (the Bias name)
                # number_of_models = len(all_models)
                # If you want to specify exactly how many rows to span:
                # we could do: \multirow{<num_of_rows>}{*}{<bias>}
                row_bias = rf"\multirow{{{len(all_models)}}}{{*}}{{\parbox{{1.8cm}}{{{bias}}}}}"
                first_model = False
            else:
                row_bias = ""  # empty for subsequent rows

            # Format the row
            row = (
                f"{row_bias.replace('_', ' ')} & {model.replace('llama3.1', 'llama').replace('claude_3_5_sonnet_v2', 'claude3.5')} & {p_lt_005_val} & {p_rec_str} & "
                f"{p_pos_005_val} & {p_pos_str} & {p_lt_005_val2} & {p_rec_str2} & "
                f"{p_pos_005_val2} & {p_pos_str2}\\\\"
            )
            latex_lines.append(row)

        latex_lines.append(r"\midrule")

    # 4) Close off the table
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\caption{Your caption here}")
    latex_lines.append(r"\label{table:your\_label}")
    latex_lines.append(r"\end{table*}")

    # 5) Return as a single LaTeX string
    return "\n".join(latex_lines)

                 
                 
latex_code = generate_latex_table(analyse_results("coffee_machines", attacks_mapping), analyse_results("coffee_machines", attacks_mapping_gen),
                                 analyse_results("cameras", attacks_mapping), analyse_results("cameras", attacks_mapping_gen))
    
    
print(latex_code)