"""Decode SDK token output, removing only a verified terminal EOS token."""
def completion_text(tokenizer, sequence):
    tokens = list(sequence["tokens"])
    eos = tokenizer.eos_token_id
    if tokens and eos is not None and tokens[-1] == eos:
        tokens = tokens[:-1]
    return tokenizer.decode(tokens, skip_special_tokens=False)
