
def compute_seq_rep_n(samples, tokenizer, n=3):
    """compute seq-rep-n metric"""
    n_gram_reps = []
    
    for s in samples:
        n_grams = []
        tokens = tokenizer(s, add_special_tokens=False).input_ids
        for i in range(len(tokens)):
            if i <= len(tokens) - n:
                n_grams.append(tuple(tokens[i:i + n]))

        if len(n_grams) > 0:
            rep = 1 - len(set(n_grams)) / len(n_grams)
        else:
            rep = 0.0
        n_gram_reps.append(rep)
        
    return n_gram_reps