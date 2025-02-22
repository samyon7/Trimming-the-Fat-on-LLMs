import torch
import torch.nn.functional as F

def create_mask(sequence_length, mask_size):
    mask = torch.ones(sequence_length, sequence_length)  
    for i in range(sequence_length):
        mask[i, max(0, i - mask_size):min(sequence_length, i + mask_size + 1)] = 0 # ubah dari 1 ke 0 untuk mengizinkan
        mask[i, :max(0, i - mask_size)] = 1 
        mask[i, min(sequence_length, i + mask_size + 1):] = 1 
    return mask

# 2. Simulasi Pengaruh "Global Token"
def apply_global_token(attention_weights, global_token_weight, global_token_index):

    sequence_length = attention_weights.shape[0]

    for i in range(sequence_length):
        attention_weights[i, global_token_index] += global_token_weight
    return attention_weights

# 3. Regularisasi Gradien (Konsep)
def regularize_global_token_gradient(global_token_gradient, previous_gradients, regularization_factor=0.1):

    if previous_gradients: #jika ada gradien sebelumnya
        avg_previous_gradient = torch.mean(torch.stack(previous_gradients))
        global_token_gradient = (1 - regularization_factor) * global_token_gradient + regularization_factor * avg_previous_gradient

    return global_token_gradient

# Contoh Penggunaan
if __name__ == '__main__':
    # Parameter contoh
    sequence_length = 10
    mask_size = 3
    global_token_index = 0  # Misalnya, token pertama adalah token global
    global_token_weight = 0.5

    # 1. Membuat mask
    mask = create_mask(sequence_length, mask_size)
    print("Mask:\n", mask)

    # 2. Membuat bobot atensi acak (simulasi)
    attention_weights = torch.rand(sequence_length, sequence_length)
    print("\nAttention Weights (Sebelum Global Token):\n", attention_weights)

    # 3. Menerapkan pengaruh global token
    attention_weights = apply_global_token(attention_weights, global_token_weight, global_token_index)
    print("\nAttention Weights (Setelah Global Token):\n", attention_weights)

    # 4. Contoh Regularisasi Gradien (Hanya ilustrasi)
    global_token_gradient = torch.rand(1)  # Gradien untuk token global
    previous_gradients = [torch.rand(1), torch.rand(1)]  # Gradien dari iterasi sebelumnya
    regularized_gradient = regularize_global_token_gradient(global_token_gradient, previous_gradients)
    print("\nGlobal Token Gradient (Sebelum Regularisasi):", global_token_gradient)
    print("Global Token Gradient (Setelah Regularisasi):", regularized_gradient)
