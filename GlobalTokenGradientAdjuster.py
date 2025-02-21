import numpy as np

def calculate_gradient_loss(attention_map, gradient_attention_map, global_token_status, i, k_plus_1, k_minus_l):

    if global_token_status == 0:  # Gi = 0
        gradient = 0.0
        for m in range(i, attention_map.shape[0]):  #m >= i
            for n in range(0, i):  #n < i
                gradient += attention_map[m, n] * gradient_attention_map[m, n]
    else:  # Gi = 1
        gradient = 0.0
        for m in range(i, k_plus_1):  # k_{i+1} > m >= i
            for n in range(k_minus_l, i):  # i > n >= k_{i-l}
                gradient += gradient_attention_map[m, n]
    return gradient

def adjust_global_token_gradient(global_token_gradient, gradient_loss):
    return global_token_gradient - gradient_loss

# Contoh penggunaan:
attention_map = np.random.rand(5, 5)  # Contoh matriks perhatian
gradient_attention_map = np.random.rand(5, 5) # Contoh gradient dari matriks perhatian
global_token_status = 0  # Token ke-2 bukan Global Token
i = 2  # Indeks token yang sedang diproses
k_plus_1 = 4 # index global token berikutnya
k_minus_l = 0 # index global token terakhir di local sequence yang berakhir di i

# Hitung gradient dαiG-
gradient_loss = calculate_gradient_loss(attention_map, gradient_attention_map, global_token_status, i, k_plus_1, k_minus_l)
print(f"Nilai gradient dαiG-: {gradient_loss}")

# Contoh nilai gradient Global Tokens sebelum penyesuaian
global_token_gradient = 0.5

# Sesuaikan gradient Global Tokens
adjusted_gradient = adjust_global_token_gradient(global_token_gradient, gradient_loss)
print(f"Gradient Global Tokens yang disesuaikan: {adjusted_gradient}")
