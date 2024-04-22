import torch

def load_balancing(
        gate_logits: torch.Tensor,
        num_experts: torch.Tensor = None,
        top_k=2,
        layer_mode="sum") -> float:
   
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[-1].device
        layer_gate_logits = torch.stack(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
            )

    # at this point, layer_gate_logits has shape of (layer, bs*seq, num_exp)
    routing_weights = torch.nn.functional.softmax(layer_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_masks = torch.nn.functional.one_hot(selected_experts, num_experts).sum(-2).float()

    # https://arxiv.org/pdf/2403.07816.pdf Section 3.3
    routing_weights = routing_weights.mean(1)
    routing_masks = routing_masks.mean(1)
    loss_lb = num_experts * torch.sum(routing_weights * routing_masks, dim=-1)
    
    if layer_mode == "sum":
        loss_lb = loss_lb.sum()
    elif layer_mode == "average":
        loss_lb = loss_lb.mean()
    else:
        raise NotImplementedError
    
    return loss_lb