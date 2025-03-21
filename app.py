import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

# Detect device (use GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer with caching for efficiency
@st.cache_resource
def load_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model = PeftModel.from_pretrained(model, "./gpt2-lora-finetuned-riddles")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-lora-finetuned-riddles")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Function to generate a solution for the given riddle
def generate_solution(riddle):
    if not riddle:
        return "Please enter a riddle."
    
    # Format the input to match the training data style
    input_text = f"Riddle: {riddle}\nAnswer:"
    
    # Tokenize the input and move tensors to the appropriate device
    encoding = tokenizer(input_text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Generate the solution using the model
    with torch.no_grad():
        generated_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,  # Short output since answers are numbers
            do_sample=True,    # Enable sampling to introduce variability
            temperature=0.5,   # Lower temperature for more focused output
            top_p=0.9,         # Use nucleus sampling to filter unlikely tokens
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated output
    decoded_output = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    
    # Log the raw output for debugging
    st.write("Raw Model Output:", decoded_output)
    
    # Extract the solution (text after "Answer:")
    parts = decoded_output.split("\nAnswer:")
    if len(parts) > 1:
        # Take only the first line after "Answer:" and strip whitespace
        solution = parts[1].strip().split("\n")[0].strip()
        # Ensure the solution is a number (or handle non-numeric output)
        if solution.isdigit():
            return solution
        else:
            return "Generated a non-numeric solution: " + solution
    else:
        return "Could not generate a solution."

# Streamlit app interface
st.title("Math Riddle Solver")
st.write("Enter a math riddle, and I'll try to solve it!")
st.write("For example: 'What number, when squared, equals 64?'")

# Input field for the riddle
riddle = st.text_input("Riddle:")

# Button to generate the solution
if st.button("Solve"):
    with st.spinner("Solving the riddle..."):
        solution = generate_solution(riddle)
    st.write("Solution:", solution)