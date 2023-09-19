import streamlit as st
from transformers import AutoTokenizer, pipeline
import torch

model = "tanzir11/llm:0115fd6c6a9b1deed47a63258c13a0ee65cc3eeb5c20a747bbd651914e602237"
api_token="Replicate API key"


def main():
    tokenizer = AutoTokenizer.from_pretrained(model, api_key=api_token)
    chatbot_pipeline = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device=0,
    )

    st.header("Merger Assistant")

    # Take input from the user
    prompt = st.text_area("Enter Your Message", height=100)

    submit = st.button("Send")

    # When 'Send' button is clicked, execute the chatbot pipeline
    if submit:
        # Generating the response using chatbot pipeline
        sequences = chatbot_pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
        )

        for seq in sequences:
            generated_text = seq['generated_text']
            st.subheader("Generated Response")
            st.write(generated_text)


if __name__ == "__main__":
    main()


