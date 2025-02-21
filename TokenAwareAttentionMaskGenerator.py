import torch
import torch.nn as nn
import torch.nn.functional as F

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """Gumbel-Softmax sampling."""
    U = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    y = logits + gumbel_noise
    y = torch.softmax(y / tau, dim=-1)

    if hard:
        y_hard = torch.eq(y, torch.max(y, dim=-1, keepdim=True)[0]).float()
        y = (y_hard - y).detach() + y

    return y

class AttentionMaskGenerator(nn.Module):
    def __init__(self, num_kv_heads, num_token_types, sliding_window_size):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_token_types = num_token_types
        self.sliding_window_size = sliding_window_size

        # Learnable linear layer to generate logits
        self.attention_score_layer = nn.Linear(128, num_token_types) # Assuming input tensor size of 128

    def forward(self, input_tensor, token_types):
        batch_size, seq_len, _ = input_tensor.size()

        # 1. Generate logits for each token type
        logits = self.attention_score_layer(input_tensor) # (batch_size, seq_len, num_token_types)

        # 2. Sample the token types (using Gumbel-Softmax for a differentiable choice)
        token_types_soft = gumbel_softmax(logits, hard=True)
        token_types = torch.argmax(token_types_soft, dim=-1)

        # Initialize the attention mask
        attention_mask = torch.ones((batch_size, self.num_kv_heads, seq_len, seq_len), device=input_tensor.device)

        # Apply token-type specific masks:
        for i in range(seq_len):
            for j in range(seq_len):
                for batch in range(batch_size):
                    if token_types[batch, i] == 0:  # Global Token
                        attention_mask[batch, :, i, j] = 1.0  # Always attend
                    elif token_types[batch, i] == 1:  # Sliding Window Token
                        if j >= i and j <= i + self.sliding_window_size:
                            attention_mask[batch, :, i, j] = 1.0  # Attend within window
                        else:
                            attention_mask[batch, :, i, j] = 0.0 # Do not attend
                    elif token_types[batch, i] == 2:  # Local Token
                        # Find the next global token index
                        next_global_token_index = seq_len
                        for k in range(i + 1, seq_len):
                            if token_types[batch, k] == 0:
                                next_global_token_index = k
                                break

                        if j <= next_global_token_index:
                            attention_mask[batch, :, i, j] = 1.0  # Attend until next global token
                        else:
                            attention_mask[batch, :, i, j] = 0.0  # Do not attend

        # Causal Mask:  Ensure no attending to future tokens
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_tensor.device)).bool()
        attention_mask = attention_mask.masked_fill(~causal_mask, 0)  # fill lower triangular part with 0

        return attention_mask

# Example Usage:
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    input_dim = 128
    num_kv_heads = 8
    num_token_types = 3  # Global, Sliding Window, Local
    sliding_window_size = 3

    # Create a random input tensor
    input_tensor = torch.randn(batch_size, seq_len, input_dim)

    # Dummy token types (replace with your token type selection logic)
    token_types = torch.randint(0, num_token_types, (batch_size, seq_len))

    # Instantiate the attention mask generator
    mask_generator = AttentionMaskGenerator(num_kv_heads, num_token_types, sliding_window_size)

    # Generate the attention mask
    attention_mask = mask_generator(input_tensor, token_types)

    print("Attention Mask Shape:", attention_mask.shape) # Expected (batch_size, num_kv_heads, seq_len, seq_len)
    print("Attention Mask:\n", attention_mask)
