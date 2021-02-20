import torch


def accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    correct_predictions = (predicted == target).sum().item()
    accuracy_result = correct_predictions / target.size(0)
    return accuracy_result
