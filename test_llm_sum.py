#!/usr/bin/env python3
"""
Test text-to-text extraction with Flan-T5 (or other HF models)
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json

def main():
    model_name = "google/flan-t5-base"  # You can try large for better accuracy
    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as e:
        print("Failed to load model:", e)
        return

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    # Example doctor bio text
    bio_text = """
    Dr. Jane Smith graduated from Harvard School of Dental Medicine in 2008.
    She has 12 years of experience in pediatric dentistry.
    Born in Boston, MA, she loves volunteering for community dental health programs.
    """

    prompt = (
        "Extract structured information about the doctor in JSON format. "
        "Fields: name, education, experience, hometown, photo_url. "
        "Return only valid JSON.\n\n"
        f"BIO:\n{bio_text}"
    )

    print("\nPrompt sent to LLM:\n", prompt)
    try:
        output = pipe(prompt, max_new_tokens=200)
        text = output[0].get("generated_text") if isinstance(output, list) else str(output)
        print("\nLLM Output:\n", text)

        # Try to parse as JSON
        try:
            data = json.loads(text)
            print("\nParsed JSON:\n", json.dumps(data, indent=2))
        except Exception:
            print("LLM output could not be parsed as JSON.")
    except Exception as e:
        print("Pipeline call failed:", e)

if __name__ == "__main__":
    main()
