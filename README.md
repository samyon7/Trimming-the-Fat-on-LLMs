# Trimming-the-Fat-on-LLMs

# 🔥 **Dynamic Attention Done Right**  

Welcome to the **MultiheadAttention** class — a straight-up fire implementation of multihead self-attention with a *next-level twist*: **NAtS (Neural Attention Token Selector)**.  
This beast dynamically classifies tokens into **Global**, **Local**, and **Sliding Window** categories and builds a custom attention mask on the fly.  

Let’s break this down mathematically and keep it 💯 academic yet chill.  

---

## 💡 **Key Concepts**  

### 🔥 Multihead Self-Attention (MHA)  
We got the OG transformer vibes here — splitting the embedding into `num_heads` partitions. For each head:  

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$  

where \(X \in \mathbb{R}^{L \times B \times d}\) (seq_len, batch_size, embed_dim).  
Heads run in parallel because we like that speed ⏩.  

---

### 🧠 **Scaled Dot-Product Attention**  
We compute:  

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$  

with \(M\) as the attention mask. The scaling by \(\sqrt{d_k}\) prevents that “too spicy” softmax explosion. 🌶️  

---

## 🚀 **But Wait... We Got NAtS!**  
This ain’t your regular attention layer. We dynamically score each token type:  

$$
P(\text{token\_type}) = \text{softmax}(XW^{\text{type}})
$$  

where \(W^{\text{type}}\) maps embeddings into a \(\mathbb{R}^{d \times 3}\) space — representing our holy trinity:  
1. **Global** (Yo, I attend to everything) 🌎  
2. **Local** (Just vibin’ with the neighborhood) 🏡  
3. **Sliding Window** (Rollin' through time, one frame at a time) 🎞️  

---

## 🏗️ **Dynamic Mask Construction**  
Based on token probs, we build the attention mask \(M\):  

- **Global Tokens**:  
  $$
  M_{i,j} = 0 \quad \forall j \leq i \quad \text{(full context access)}  
  $$  

- **Local Tokens**:  
  $$
  M_{i,j} = -\infty \quad \text{if } j < i \quad \text{(no peeking too far back)}  
  $$  

- **Sliding Window Tokens (window=3)**:  
  $$
  M_{i,j} = 
  \begin{cases} 
  0 & \text{if } |i-j| \leq 3 \\ 
  -\infty & \text{otherwise} 
  \end{cases}
  $$  

This mask keeps the attention focused, efficient, and context-aware, like a well-tuned Markov chain 🔗.  

---

## 🎛️ **Code Walkthrough**  

### `__init__()` — *Setting up the party* 🎉  
- Splits embeddings into heads (`embed_dim // num_heads`)  
- Sets up linear projections for Q, K, V  
- Adds a token-type scorer (NAtS brain)  
- Drops out (literally) for regularization  

---

### `forward()` — *Where the magic happens* ✨  
1. **Projection**: Compute Q, K, V from inputs.  
2. **Reshaping**: Turn \(B, L, D\) into \(B \times H, L, d_h\).  
3. **Token Typing**: Predict token types dynamically:  

   $$
   P(t) = \text{softmax}(XW^{\text{type}})
   $$  

4. **Dynamic Masking**: Generate attention mask via `construct_nats_attn_mask()`.  
5. **Scaled Dot-Product**: Compute attention and apply dropout.  
6. **Output Projection**: Concatenate heads and project back.  

---

### `construct_nats_attn_mask()` — *NAtS in action* 🧩  
This function loops through each token and dynamically assigns who it can attend to based on learned probabilities. Think of it like a bouncer letting only certain tokens into the VIP section. 🎟️  

---

### `scaled_dot_product_attention()` — *Classic transformer math* 🧮  
Executes the OG attention formula:  

$$
\alpha_{ij} = \frac{\exp\left(\frac{(QK^T)_{ij}}{\sqrt{d_k}}\right)}{\sum_j \exp\left(\frac{(QK^T)_{ij}}{\sqrt{d_k}}\right)}
$$  

Multiplies by \(V\) to get the final output:  

$$
\text{Output} = \alpha V
$$  

---

## 🏃 **Why This Rocks**  
- 🧠 **Dynamic Attention**: Adaptive masking = smarter token dependencies.  
- 🚀 **Efficient Context Use**: Focuses compute where it matters.  
- 🔥 **Flexible Architecture**: Just tweak the number of token types or thresholds.  

---

## 🎯 **TL;DR**  
This code implements a multihead attention mechanism that doesn’t just blindly attend everywhere. Instead, it **learns** what matters, dynamically shifting between global, local, and sliding-window attention modes.  

Special credits for my brother and my friend, Jo. Dani and Wanto

---
