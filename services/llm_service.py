import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List

LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf" 


class LlamaService:
    def __init__(self, model_name: str, device: str):
        """Initializes the Llama model and pipeline."""
        print(f"Loading LLM: {model_name} on {device}...")
        self.device = device
        

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
  
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512, # Limit generation length for concise answers
                temperature=0.2,    # Low temperature minimizes hallucination
                return_full_text=False # Return only the generated text
            )
            print("Llama model and pipeline loaded successfully.")

        except Exception as e:
            print(f"Failed to load Llama model. Ensure you have the model weights and necessary libraries (transformers, torch) installed: {e}")
            self.pipeline = None
            raise

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates an answer based on the provided context using the Llama model.
        """
        if not self.pipeline:
            return "LLM service is not initialized."


        prompt_template = f"""
        ### Instruction:
        Write a concise, academic answer to the user's question based ONLY on the provided context.
        If the context does not contain the answer, state: "The required information was not found in the retrieved textbook pages."

        ### Input:
        Context:
        ---
        {context}
        ---
        Question: {query}

        ### Response:
        """

        try:
            # Run the generation pipeline
            result = self.pipeline(prompt_template)
            
            # Extract the generated text and clean up the response
            answer = result[0]['generated_text'].strip()
            
            # Clean up residual instruction/prompt text that the model might echo
            if answer.startswith(prompt_template):
                answer = answer.replace(prompt_template, '').strip()

            return answer
        
        except Exception as e:
            return f"LLM Generation Runtime Error: {e}"

# --- Helper function for easy access in app.py ---
def load_llama_service(model_name: str, device: str) -> LlamaService:
    return LlamaService(model_name, device)