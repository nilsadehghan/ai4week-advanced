from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import chainlit as cl  # For creating interactive chat apps

# Load tokenizer and model with safety tensor format
tokenizer = AutoTokenizer.from_pretrained("rahiminia/manshoorai", use_safetensors=True)
model = AutoModelForCausalLM.from_pretrained("rahiminia/manshoorai", use_safetensors=True)

# Define the main chat function that generates poetic responses
def chat(input_text):
    # Set the persona to answer poetically
    persona = "تو باید سوالات رو به صورت شاعرانه و ادبی پاسخ بدی"
    
    # Combine persona instruction with user input
    prompt = f"{persona}\n{input_text}"
    
    # Tokenize the combined prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate a response using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=100,        # Limit the response length
            do_sample=True,        # Enable sampling for more diverse output
            top_p=0.9,             # Nucleus sampling for randomness
            top_k=40,              # Consider only top 40 tokens at each step
            pad_token_id=tokenizer.eos_token_id  # Avoid padding errors
        )
    
    # Decode the model output into human-readable text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the output to return only the model's response
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):]

    # Remove persona instruction and clean up output
    output_text = output_text.replace(persona, "").strip()
    
    return output_text

# Chainlit event handler: Triggered when chat starts
@cl.on_chat_start
async def start():
    await cl.Message(content="دوست من خوش آمدین.").send()  # Greet the user poetically

# Chainlit event handler: Triggered on each user message
@cl.on_message
async def main(message: cl.Message):
    response = chat(message.content)  # Generate a poetic response
    await cl.Message(content=response).send()  # Send the response to the user
