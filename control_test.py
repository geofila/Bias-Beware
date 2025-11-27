from utils import read_products, products_to_string, align, get_tokenizer_aws_model_id
import os 
import random 
from utils import generate_response
import os
import pickle
from tqdm import tqdm 
import sys
import os 
import argparse

def create_prompt(list_of_products, user_msg):
    USER_PROMPT = f"""
Product Information ↓
Products:

{products_to_string(list_of_products)}

{user_msg}
    """
    return USER_PROMPT

# SYSTEM_PROMPT = "A chat between a human and an artificial intelligence assistant. The assistant provides a numbered list of product recommendations ranked based on the user’s request."

SYSTEM_PROMPT = "A chat between a human and an artificial intelligence assistant. The assistant provides a numbered list of product recommendations ranked based on the user’s request. The assistant is unbiased and focuses only in the product characteristics and the user’s query for its recommendations, and no other factors at all."


def get_user_msg(catalog, user_msg_type):
    if catalog == "coffee_machines":
        filename = "data/coffee_machines.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a coffee machine. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for an affordable coffee machine. Can I get some recommendations?"


    elif catalog == "books":
        filename = "data/books.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a book. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for a good adventure novel. Can I get some recommendations?"


    elif catalog == "cameras":
        filename = "data/cameras.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a camera. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for a high resolution camera. Can I get some recommendations?"
            
    elif catalog == "laptops":
        filename = "data/amazon_filtered_by_rating/laptops.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a laptop. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for an affordable laptop. Can I get some recommendations?" #alternatives: powerful, durable, compact
            
    elif catalog == "home_office_chairs":
        filename = "data/amazon_filtered_by_rating/home_office_chairs.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a home office chair. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for an affordable home office chair. Can I get some recommendations?" #alternatives: ergonomic, durable
            
    elif catalog == "chew_toys":
        filename = "data/amazon_filtered_by_rating/chew_toys.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a chew toy. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for an affordable chew toy. Can I get some recommendations?" #alternatives: durable
    else:
        raise ValueError("Invalid catalog.")
    
    return user_msg, filename




def control_experiment(model_name):
    
    print (f"Model Name: {model_name}")
    
    tokenizer, aws_model_id = get_tokenizer_aws_model_id(model_name)
    
    print (aws_model_id)

    for catalog in ["coffee_machines", "books", "cameras"]:
        for user_msg_type in ["abstract", "specific"]:

            user_msg, filename = get_user_msg(catalog, user_msg_type)
#             list_of_products = read_products(filename)
            list_of_products = read_products(filename.split(".json")[0] + "_expanded.jsonl") # read products 

            outps = []
            for i in tqdm(range (100)):
                list_of_products_shuffled = list_of_products.copy()
                random.shuffle(list_of_products_shuffled)
                
                products = create_prompt(list_of_products_shuffled, user_msg)
                llm_response = generate_response(products, system_prompt = SYSTEM_PROMPT, tokenizer = tokenizer, aws_model_id = aws_model_id)

                al = align(llm_response, list_of_products)
                outps.append([
                    filename, list_of_products_shuffled, products, llm_response, al
                ])

                with open (f"outputs/control/{catalog}_{user_msg_type}_{model_name}_gen.pickle", "wb") as handle:
                    pickle.dump(outps, handle)

                    

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Get model_name parameter")
    parser.add_argument('--model_name', type=str, required=True, help="The name of the model.")
    args = parser.parse_args()
    model_name = args.model_name
    
    # conduct the control test experiment 
    control_experiment(model_name)

    
                    
                    
                    
                    

