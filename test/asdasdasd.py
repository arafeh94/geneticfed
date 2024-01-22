import torch
import torch.nn as nn

# Sample data
num_classes = 3
batch_size = 4

# Assuming your model outputs raw logits (before softmax)
logits = torch.tensor([
    [0, 100.0, 0],
    [0, 100.0, 0],
    [100.0, 0, 0],
    [0, 100.0, 0]
])

# Ground truth labels (class indices)
target_labels = torch.tensor([0, 1, 0, 1])
# Define the CrossEntropyLoss without class weights
criterion = nn.CrossEntropyLoss()

# Calculate the loss
loss = criterion(logits, target_labels)
# Print the loss
print("Loss without class weights:", loss.item())

# Define class weights (higher weight for class 2)
class_weights = torch.tensor([1.0, 100.0, 1.0])

# Define the CrossEntropyLoss with class weights
criterion_with_weights = nn.CrossEntropyLoss(weight=class_weights)

# Calculate the loss with class weights
loss_with_weights = criterion_with_weights(logits, target_labels)

# Print the loss with class weights
print("Loss with class weights:", loss_with_weights.item())
