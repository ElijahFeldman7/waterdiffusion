import torch

def batched_multi_fft_convolution_idx(
    W: torch.Tensor,
    convolution_idx: torch.Tensor,
    start_at: int = None,
    end_at: int = None
) -> torch.Tensor:
    """
    Batched k-way 1D convolution via FFT using arbitrary relative indices.

    Args:
        W (torch.Tensor): shape (B, S, V)
        convolution_idx (torch.Tensor): 1D-Tensor of length k, relative time offsets
        start_at (int, optional): Start index for the output sequence. If None, computed from convolution_idx.
        end_at (int, optional): End index for the output sequence. If None, computed from convolution_idx.

    Returns:
        torch.Tensor: shape (B, S', k*V), where
          S' = S - (max(convolution_idx) - min(convolution_idx)),
          padded and computed via FFT, with dtype of W
    """
    if W.dim() != 3:
        raise ValueError(f"W must be 3D, got shape {tuple(W.shape)}")

    B, S, V = W.shape
    idx = convolution_idx
    # determine valid output positions
    if not start_at:
        start_at = max(-int(idx.min()), 0)
    if not end_at:
        end_at = S - max(int(idx.max()), 0)
    S_out = end_at - start_at

    # compute absolute positions tensor: (B, S_out, k)
    base = torch.arange(start_at, end_at, device=W.device).unsqueeze(1)  # (S_out, 1)
    positions = base + idx.unsqueeze(0)                                 # (S_out, k)
    positions = positions.unsqueeze(0).expand(B, -1, -1)               # (B, S_out, k)

    # gather the k slices per timestep: (B, S_out, k, V)
    Wg = torch.gather(
        W.unsqueeze(2).expand(-1, -1, idx.size(0), -1),
        dim=1,
        index=positions.unsqueeze(-1).expand(-1, -1, -1, V)
    )

    # merge batch and time dims
    N = B * S_out
    k = idx.numel()
    Wm = Wg.contiguous().view(N, k, V)  # (N, k, V)

    # FFT-based convolution steps
    conv_len = k * V
    Nfft = 1 << ((conv_len - 1).bit_length())

    # FFT along feature axis
    F = torch.fft.rfft(Wm.float(), n=Nfft, dim=2)  # (N, k, Nfft//2+1)
    F_prod = F.prod(dim=1)                         # (N, Nfft//2+1)
    full = torch.fft.irfft(F_prod, n=Nfft, dim=1)  # (N, Nfft)
    valid = full[:, :conv_len]                     # (N, conv_len)

    # reshape back to (B, S_out, k*V)
    Pb = valid.view(B, S_out, conv_len).to(W.dtype)
    return Pb


def compute_prob_of_min(probs, convolution_kernel, start_at=None, end_at=None):
    """Compute the probability of the minimum tokens across the kernel.
   
    Args:
        probs (torch.Tensor): Tensor of shape (B, S, V) containing probabilities.
                              B is the batch size, S is the sequence length, and V is the vocabulary size.
        convolution_kernel (torch.Tensor or list): 1D tensor or list of integers representing offsets for convolution.
    Returns:
        torch.Tensor: Tensor of shape (B, valid_length, V) containing the computed probabilities.
                      valid_length is determined by the convolution kernel and sequence length.
   
    """
    B, S, V = probs.shape
    device = probs.device
    dtype = probs.dtype

    if not isinstance(convolution_kernel, torch.Tensor):
        convolution_kernel = torch.tensor(convolution_kernel, device=device, dtype=torch.long)
    else:
        convolution_kernel = convolution_kernel.to(device=device, dtype=torch.long)

    min_offset = int(convolution_kernel.min())
    max_offset = int(convolution_kernel.max())

    if start_at is None:
        start_at = max(-min_offset, 0)
    if end_at is None:
        end_at = S - max(max_offset, 0)
    

    if start_at >= end_at: # No valid tokens to process
        return torch.empty((B, 0, V), device=device, dtype=dtype)


    target_tok_indices = torch.arange(start_at, end_at, device=device) # e.g., [t1, t2, ..., tN]

    source_s_indices = target_tok_indices[:, None] + convolution_kernel[None, :]


    batch_indices_mesh = torch.arange(B, device=device)[:, None, None] # (B, 1, 1)
    source_s_indices_mesh = source_s_indices[None, :, :]              # (1, num_valid_toks, K)
    
    probs_to_minhash = probs[batch_indices_mesh, source_s_indices_mesh, :]


    survival_functions = 1.0 - torch.cumsum(probs_to_minhash, dim=-1)
    
    survival_function_prod = torch.prod(survival_functions, dim=2)

    ones_to_prepend = torch.ones((B, target_tok_indices.shape[0], 1), device=device, dtype=dtype)
    
    survival_function_cat = torch.cat((ones_to_prepend, survival_function_prod), dim=-1)
    
    out_probs_vectorized = survival_function_cat[..., :-1] - survival_function_cat[..., 1:]
    
    return out_probs_vectorized

def offset_unfold(
    x: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Mimic the torch unfold operation with arbitrary relative offsets.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L).
        offsets (torch.Tensor): 1-D tensor with relative positions.

    Returns:
        torch.Tensor: Output tensor of shape (B, valid_length, len(offsets)).
        with `out[..., i, j, ...] == x[..., start_i + offsets[j], ...]`
        for every valid starting position `i`.
    """
    
    B, L = x.shape
    
    start_at = max(-min(offsets), 0)
    end_at = L - max(max(offsets), 0)
    valid_length = end_at - start_at
    
    token_indices = torch.arange(start_at, end_at, device=x.device)

    positions = token_indices.unsqueeze(1) + offsets.unsqueeze(0)
    
    batch_indices = torch.arange(B, device=x.device).repeat_interleave(
        valid_length * len(offsets)
    )
    
    flat_positions = positions.flatten().repeat(B)
    
    gathered = x[batch_indices, flat_positions]
    out = gathered.view(B, valid_length, len(offsets))
    
    return out