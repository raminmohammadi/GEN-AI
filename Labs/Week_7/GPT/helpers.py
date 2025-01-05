def generate(prompt, model, tokenizer, max_length=100):
    """
    Inference phase: Generate text from a given prompt using a fine-tuned GPT-2 model.

    Args:
        prompt (str): The starting text for the generation.
        model (TFGPT2LMHeadModel): The fine-tuned GPT-2 model.
        tokenizer (GPT2Tokenizer): The tokenizer used for encoding and decoding text.
        max_length (int, optional): The maximum length of the generated text. Defaults to 100.

    Returns:
        str: The generated text.
    """

    """Inference phase: Generate text from prompt"""
    input_ids = tokenizer.encode(prompt, return_tensors="tf")
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)