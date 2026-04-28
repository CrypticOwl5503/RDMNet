import torch

# Demonstrate Bug 1: Transposing after softmax breaks probability distributions
attn = torch.randn(2, 4, 4)
attn_softmax = torch.softmax(attn, dim=-1)

print("--- Before the Fix (Bug 1) ---")
print("Row sums of softmaxed attn (should be 1):", attn_softmax.sum(dim=-1)[0].tolist())
attn_buggy = attn_softmax.transpose(-1, -2)
print("Row sums of TRANSPOSED softmaxed attn:", [round(x, 4) for x in attn_buggy.sum(dim=-1)[0].tolist()])
print("Notice how the rows no longer sum to 1. This means the features get arbitrarily scaled up or down!\n")

print("--- After the Fix ---")
attn_fixed = torch.softmax(attn.transpose(-1, -2), dim=-1)
print("Row sums of CORRECT transposed softmax:", [round(x, 4) for x in attn_fixed.sum(dim=-1)[0].tolist()])
print("The rows correctly sum to 1, preserving the scale of your features.")
