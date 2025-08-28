# test_finetuned_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    print("🔄 Loading model and tokenizer...")
    print(f"📁 Model path: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Fix pad token issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",          # Automatically distribute across available devices
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        print(f"🧠 Model type: {model.config.model_type}")
        print(f"📊 Parameters: ~{model.num_parameters() / 1e9:.1f}B")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    """Generate a response from the model"""
    print(f"\n💭 Generating response for: '{prompt[:50]}...'")
    
    try:
        # Encode the prompt with attention mask
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Move to same device as model
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    except Exception as e:
        print(f"❌ Error generating response: {str(e)}")
        return None

def interactive_chat(model, tokenizer):
    """Interactive chat with your fine-tuned model"""
    print("\n" + "="*60)
    print("🤖 INTERACTIVE CHAT WITH YOUR FINE-TUNED MODEL")
    print("="*60)
    print("💡 Tips:")
    print("   • Type 'quit' or 'exit' to stop")
    print("   • Type 'clear' to clear conversation history")
    print("   • Be specific with your questions!")
    print("-"*60)
    
    conversation_history = ""
    
    while True:
        # Get user input
        user_prompt = input("\n🧑 You: ").strip()
        
        # Check for exit commands
        if user_prompt.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        # Clear conversation
        if user_prompt.lower() == 'clear':
            conversation_history = ""
            print("🗑️ Conversation cleared!")
            continue
        
        if not user_prompt:
            continue
        
        # Format prompt (adjust based on your fine-tuning format)
        if conversation_history:
            full_prompt = f"{conversation_history}\nHuman: {user_prompt}\nAssistant:"
        else:
            full_prompt = f"Human: {user_prompt}\nAssistant:"
        
        # Generate response
        response = generate_response(model, tokenizer, full_prompt)
        
        if response:
            print(f"\n🤖 Model: {response}")
            # Add to conversation history
            conversation_history = f"{full_prompt} {response}"
        else:
            print("❌ Sorry, couldn't generate a response.")

def test_sample_prompts(model, tokenizer):
    """Test with some sample prompts"""
    print("\n" + "="*60)
    print("🧪 TESTING WITH SAMPLE PROMPTS")
    print("="*60)
    
    # Sample prompts - adjust based on what you fine-tuned for
    sample_prompts = [
        "What is the difference between a virus and bacteria?",
        "Explain how antibiotics work.",
        "What are the symptoms of diabetes?",
        "How does the immune system work?",
        "What causes high blood pressure?"
    ]
    
    for i, prompt in enumerate(sample_prompts, 1):
        print(f"\n📝 Sample {i}: {prompt}")
        print("-" * 40)
        
        response = generate_response(model, tokenizer, f"Human: {prompt}\nAssistant:")
        
        if response:
            print(f"🤖 Response: {response}")
        else:
            print("❌ No response generated")
        
        print()

def main():
    # CONFIGURATION - UPDATE THIS PATH!
    MODEL_PATH = r"/teamspace/studios/this_studio/qlora_pubmedqa_merged_7B"
    
    print("🚀 FINE-TUNED MODEL TESTER")
    print("=" * 60)
    
    # Load the model
    model, tokenizer = load_model(MODEL_PATH)
    
    if model is None or tokenizer is None:
        print("💡 Please check your model path and try again.")
        return
    
    # Choose what to do
    print("\n🎯 What would you like to do?")
    print("1. Test with sample prompts")
    print("2. Interactive chat")
    print("3. Both")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        test_sample_prompts(model, tokenizer)
    
    if choice in ['2', '3']:
        interactive_chat(model, tokenizer)
    
    print("\n🎉 Testing complete!")

if __name__ == "__main__":
    main()