import numpy as np

# Given embeddings for the sentence
embedding_the = np.array([0.2, -0.5])
embedding_sky = np.array([-0.7, 0.3])
embedding_is = np.array([0.1, 0.6])
embedding_blue = np.array([0.9, 0.2])

# Given weight matrices
W_Q = np.array([[0.1, 0.2], [0.3, 0.4]])
W_K = np.array([[0.5, 0.6], [0.7, 0.8]])
W_V = np.array([[0.9, 1.0], [1.1, 1.2]])

# Calculate Q, K, and V for each word
Q_the = np.dot(embedding_the, W_Q)
K_the = np.dot(embedding_the, W_K)
V_the = np.dot(embedding_the, W_V)

Q_sky = np.dot(embedding_sky, W_Q)
K_sky = np.dot(embedding_sky, W_K)
V_sky = np.dot(embedding_sky, W_V)

Q_is = np.dot(embedding_is, W_Q)
K_is = np.dot(embedding_is, W_K)
V_is = np.dot(embedding_is, W_V)

Q_blue = np.dot(embedding_blue, W_Q)
K_blue = np.dot(embedding_blue, W_K)
V_blue = np.dot(embedding_blue, W_V)

# Calculate attention scores for each word
attention_scores_the = np.dot(Q_the, K_the) / np.sqrt(2)
attention_scores_sky = np.dot(Q_sky, K_sky) / np.sqrt(2)
attention_scores_is = np.dot(Q_is, K_is) / np.sqrt(2)
attention_scores_blue = np.dot(Q_blue, K_blue) / np.sqrt(2)

# Apply softmax to get attention weights for each word
attention_weights_the = np.exp(attention_scores_the) / np.sum(np.exp(attention_scores_the))
attention_weights_sky = np.exp(attention_scores_sky) / np.sum(np.exp(attention_scores_sky))
attention_weights_is = np.exp(attention_scores_is) / np.sum(np.exp(attention_scores_is))
attention_weights_blue = np.exp(attention_scores_blue) / np.sum(np.exp(attention_scores_blue))

# Calculate weighted sum of values for each word
weighted_sum_the = np.dot(attention_weights_the, V_the)
weighted_sum_sky = np.dot(attention_weights_sky, V_sky)
weighted_sum_is = np.dot(attention_weights_is, V_is)
weighted_sum_blue = np.dot(attention_weights_blue, V_blue)

# Print results
print("Attention Scores (the):", attention_scores_the)
print("Attention Weights (the):", attention_weights_the)
print("Weighted Sum (the):", weighted_sum_the)

print("Attention Scores (sky):", attention_scores_sky)
print("Attention Weights (sky):", attention_weights_sky)
print("Weighted Sum (sky):", weighted_sum_sky)

print("Attention Scores (is):", attention_scores_is)
print("Attention Weights (is):", attention_weights_is)
print("Weighted Sum (is):", weighted_sum_is)

print("Attention Scores (blue):", attention_scores_blue)
print("Attention Weights (blue):", attention_weights_blue)
print("Weighted Sum (blue):", weighted_sum_blue)
