import torch


def node_right_value(W, B):
    return torch.sum(torch.abs(W), dim=1) + torch.abs(B) # Weights for the incoming connections (node to the right)

def node_left_value(W, B):
    return torch.sum(torch.abs(W), dim=0) + torch.abs(B) # Weights for the outgoing connections (node to the left)

def entropic_loss_right(W, B, alpha=1.0, beta = 1.0):
    right_values = node_right_value(W, B)
    normalized_values = right_values / torch.sum(right_values)
    if beta == 1.0:
        loss = -torch.sum(normalized_values * torch.log(normalized_values + 1e-10))
    else:
        loss = beta/(1-beta) * torch.log(torch.norm(normalized_values, p=beta) + 1e-10)
    return alpha * loss

def entropic_loss_left(W, B, alpha=1.0, beta = 1.0):
    left_values = node_left_value(W, B)
    normalized_values = left_values / torch.sum(left_values)
    if beta == 1.0:
        loss = -torch.sum(normalized_values * torch.log(normalized_values + 1e-10))
    else:
        loss = beta/(1-beta) * torch.log(torch.norm(normalized_values, p=beta) + 1e-10)
    return alpha * loss

def entropic_loss(W_l, W_r, B, alpha=1.0, beta=1.0): # Difference between this and cumm_entropic_loss is that this one computes the loss for both left and right nodes separately
    loss_l = entropic_loss_left(W_r, B, alpha, beta)
    loss_r = entropic_loss_right(W_l, B, alpha, beta)
    return loss_l + loss_r

def cumm_entropic_loss(W_l, W_r, B, alpha=1.0): # Difference between this and entropic_loss is that this one assigns the node weight from the left and right before computing the loss
    values = node_right_value(W_l, B) + node_left_value(W_r, B)
    normalized_values = values / torch.sum(values)
    loss = -torch.sum(normalized_values * torch.log(normalized_values + 1e-10))  # Add small constant to avoid log(0)
    return loss * alpha