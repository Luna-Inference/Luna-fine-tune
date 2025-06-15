# -*- coding: utf-8 -*-
"""
Test script for the finetuned Qwen2.5 LoRA model.
Loads the saved LoRA adapters and runs inference.
"""
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

# Configuration for loading the LoRA model
# These should match the parameters used during saving,
# or be compatible with the saved model.
max_seq_length = 2048
dtype = None  # None for auto detection
load_in_4bit = True # Must match the training/saving configuration

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def run_inference_tests():
    """Loads the LoRA model and runs inference tests."""
    print("Loading LoRA model for inference...")
    # Ensure the model is loaded onto the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, loading model on CPU. This will be very slow.")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="lora_model",  # Assuming 'lora_model' is the save directory
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        # No explicit .to(device) needed here for Unsloth's FastLanguageModel
        # as it handles device placement during from_pretrained and for_inference.
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'lora_model' directory exists and contains the saved model and tokenizer.")
        print("This script should be run after qwen25.py has successfully saved the 'lora_model'.")
        return

    FastLanguageModel.for_inference(model) # Prepares model for faster inference
    print("Model loaded successfully. Running inference tests...\n")

    prompt_instructions = [
        "hello friend I am feeling very down.",
        "Hi there, I'm feeling quite low today.",
        "Hey, feeling really down at the moment.",
        "Hello, I'm not doing so well, feeling down.",
        "Greetings, I'm in a bit of a slump emotionally.",
        "My friend, I'm feeling very sad right now.",
        "I'm feeling down, could use some cheering up.",
        "Just wanted to share that I'm feeling pretty blue.",
        "Feeling a bit under the weather, emotionally speaking.",
        "It's a tough day for me, feeling very down."
    ]

    for i, instruction_text in enumerate(prompt_instructions):
        print(f"--- Test {i+1}: User Feeling Down Variation ---")
        print(f"Instruction: {instruction_text}")
        
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    instruction_text,  # instruction
                    "",  # input - kept empty as per typical use for this kind of prompt
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to(device)

        print("Generating with streamer:")
        text_streamer = TextStreamer(tokenizer, skip_prompt=True) # skip_prompt=True to only see the model's response
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)
        print("\n" + "-" * 30 + "\n")

    print("All 10 inference tests completed.")

if __name__ == "__main__":
    run_inference_tests()
