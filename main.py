import numpy as np
from collections import Counter

with open("text.txt", "r", encoding="utf-8") as f:     #сам текст пушити не буду
    text = f.read().lower()

text = ''.join([ch for ch in text if ch.isalpha() or ch == ' '])

letters = sorted(list(set(text)))
m = len(letters)
char_to_idx = {ch: i for i, ch in enumerate(letters)}

counts_single = Counter(text)
total_chars = len(text)
p = np.array([counts_single[ch] / total_chars for ch in letters])

bigram = np.zeros((m, m))
for i in range(len(text) - 1):
    c1, c2 = text[i], text[i+1]
    bigram[char_to_idx[c1]][char_to_idx[c2]] += 1

row = bigram.sum(axis=1, keepdims=True)
Q = np.divide(bigram, row, out=np.zeros_like(bigram), where=row!=0)

p_next = Q.T @ p

print("max absolute deviation (||Q^T * p - p||_max):", np.max(np.abs(p_next - p)))
if np.allclose(p_next, p, atol=1e-3):
    print("correct")
