import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal, Optional
from utils import add_gumbel_noise, get_num_transfer_tokens, top_k_sampling_with_logging, log_topk_for_selected, get_key_based_parity, check_watermark_compliance


class WatermarkGenerator:
    """Text generator with various watermarking strategies."""

    def __init__(self, model, tokenizer, device='cuda', mask_id=126336, private_key=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.mask_id = mask_id
        self.private_key = private_key

    @torch.no_grad()
    def generate_original(self, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
                         block_length: int = 32, temperature: float = 0., cfg_scale: float = 0.,
                         remasking: Literal['low_confidence', 'random'] = 'low_confidence') -> torch.Tensor:
        """
        Original generation function with argmax token selection.

        Args:
            prompt: Input prompt tensor of shape (1, L)
            steps: Sampling steps, less than or equal to gen_length
            gen_length: Generated answer length
            block_length: Block length for semi-autoregressive generation
            temperature: Categorical distribution sampling temperature
            cfg_scale: Unsupervised classifier-free guidance scale
            remasking: Remasking strategy ('low_confidence' or 'random')
        """
        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != self.mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == self.mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                mask_index = (x == self.mask_id)

                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    @torch.no_grad()
    def generate_multinomial(self, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
                           block_length: int = 32, temperature: float = 0., cfg_scale: float = 0.,
                           remasking: Literal['low_confidence', 'random'] = 'low_confidence',
                           top_k: int = 3, verbose: bool = False) -> torch.Tensor:

        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != self.mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == self.mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                mask_index = (x == self.mask_id)

                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = top_k_sampling_with_logging(logits_with_noise, k=top_k)

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x

    @torch.no_grad()
    def generate_watermark_greedy(self, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
                                 block_length: int = 32, temperature: float = 0., cfg_scale: float = 0.,
                                 remasking: Literal['low_confidence', 'random'] = 'low_confidence') -> torch.Tensor:

        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != self.mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            for i in range(steps):
                mask_index = (x == self.mask_id)
                block_mask = mask_index[0, block_start:block_end]
                gen_positions = (block_mask.nonzero(as_tuple=False).squeeze(-1) + block_start)
                if gen_positions.numel() == 0:
                    break

                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                    selected_positions = []
                    matched = []
                    unmatched = []

                    for pos in gen_positions:
                        pos = pos.item()
                        token_id = x0[0, pos].item()
                        prob = x0_p[0, pos].item()

                        if check_watermark_compliance(pos + 1, token_id, self.private_key):
                            matched.append((pos, prob))
                        else:
                            unmatched.append((pos, prob))

                    matched.sort(key=lambda x: x[1], reverse=True)
                    unmatched.sort(key=lambda x: x[1], reverse=True)

                    if matched:
                        selected_positions.append(matched[0][0])  
                    elif unmatched:
                        selected_positions.append(unmatched[0][0])  

                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    selected_positions = gen_positions.tolist()[:1]
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
                x0 = torch.where(mask_index, x0, x)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for pos in selected_positions:
                    transfer_index[0, pos] = True
                x[transfer_index] = x0[transfer_index]

        return x

    @torch.no_grad()
    def generate_watermark_multinomial(self, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
                                     block_length: int = 32, temperature: float = 0., cfg_scale: float = 0.,
                                     remasking: Literal['low_confidence', 'random'] = 'low_confidence',
                                     top_k: int = 3, verbose: bool = False) -> torch.Tensor:

        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != self.mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            for i in range(steps):
                mask_index = (x == self.mask_id)
                block_mask = mask_index[0, block_start:block_end]
                gen_positions = (block_mask.nonzero(as_tuple=False).squeeze(-1) + block_start)
                if gen_positions.numel() == 0:
                    break

                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = top_k_sampling_with_logging(logits_with_noise, k=top_k)

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                    selected_positions = []
                    matched = []
                    unmatched = []

                    for pos in gen_positions:
                        pos = pos.item()
                        token_id = x0[0, pos].item()
                        prob = x0_p[0, pos].item()

                        if check_watermark_compliance(pos + 1, token_id, self.private_key):
                            matched.append((pos, prob))
                        else:
                            unmatched.append((pos, prob))

                    matched.sort(key=lambda x: x[1], reverse=True)
                    unmatched.sort(key=lambda x: x[1], reverse=True)

                    if matched:
                        selected_positions.append(matched[0][0])  
                    elif unmatched:
                        selected_positions.append(unmatched[0][0])  

                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    selected_positions = gen_positions.tolist()[:1]
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
                x0 = torch.where(mask_index, x0, x)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for pos in selected_positions:
                    transfer_index[0, pos] = True
                x[transfer_index] = x0[transfer_index]

        return x


    def _get_token_selection(self, logits_with_noise: torch.Tensor, strategy: Literal['greedy', 'multinomial'],
                           top_k: int = 3) -> torch.Tensor:
        """
        Apply token selection strategy.

        Args:
            logits_with_noise: Logits with noise applied
            strategy: Selection strategy ('greedy' or 'multinomial')
            top_k: Number of top tokens for multinomial sampling

        Returns:
            Selected token indices
        """
        if strategy == 'greedy':
            return torch.argmax(logits_with_noise, dim=-1)
        elif strategy == 'multinomial':
            return top_k_sampling_with_logging(logits_with_noise, k=top_k)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")


    @torch.no_grad()
    def generate_beam_search(self, prompt: torch.Tensor, steps: int = 128, gen_length: int = 128,
                           block_length: int = 32, temperature: float = 0., cfg_scale: float = 0.,
                           remasking: Literal['low_confidence', 'random'] = 'low_confidence',
                           beam_size: int = 5, sampling_strategy: Literal['greedy', 'multinomial'] = 'greedy',
                           top_k: int = 3, enable_watermark: bool = True) -> torch.Tensor:

        if beam_size == 1:
            if not enable_watermark:
                # No watermarking, use base methods
                if sampling_strategy == 'greedy':
                    return self.generate_original(prompt, steps, gen_length, block_length,
                                                temperature, cfg_scale, remasking)
                elif sampling_strategy == 'multinomial':
                    return self.generate_multinomial(prompt, steps, gen_length, block_length,
                                                   temperature, cfg_scale, remasking, top_k)
            else:
                if sampling_strategy == 'greedy':
                    return self.generate_watermark_greedy(prompt, steps, gen_length, block_length,
                                                        temperature, cfg_scale, remasking)
                elif sampling_strategy == 'multinomial':
                    return self.generate_watermark_multinomial(prompt, steps, gen_length, block_length,
                                                             temperature, cfg_scale, remasking, top_k)

        if not enable_watermark:
            if sampling_strategy == 'greedy':
                return self.generate_original(prompt, steps, gen_length, block_length,
                                            temperature, cfg_scale, remasking)
            elif sampling_strategy == 'multinomial':
                return self.generate_multinomial(prompt, steps, gen_length, block_length,
                                               temperature, cfg_scale, remasking, top_k)

        x = torch.full((1, prompt.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != self.mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            for i in range(steps):
                mask_index = (x == self.mask_id)
                block_mask = mask_index[0, block_start:block_end]
                gen_positions = (block_mask.nonzero(as_tuple=False).squeeze(-1) + block_start)
                if gen_positions.numel() == 0:
                    break

                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

                x0 = self._get_token_selection(logits_with_noise, sampling_strategy, top_k)

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

                    selected_positions = []
                    matched = []
                    unmatched = []

                    for pos in gen_positions:
                        pos = pos.item()
                        token_id = x0[0, pos].item()
                        prob = x0_p[0, pos].item()

                        if check_watermark_compliance(pos + 1, token_id, self.private_key):
                            matched.append((pos, prob))
                        else:
                            unmatched.append((pos, prob))

                    matched.sort(key=lambda x: x[1], reverse=True)
                    unmatched.sort(key=lambda x: x[1], reverse=True)

                    if matched:
                        top_candidates = matched[:beam_size]

                        best_pos = None
                        best_match_count = -1

                        # Evaluate each candidate by simulating future generation
                        for pos, _ in top_candidates:
                            x_sim = x.clone()
                            x_sim[0, pos] = x0[0, pos]
                            mask_sim = mask_index.clone()
                            mask_sim[0, pos] = False

                            with torch.no_grad():
                                logits_sim = self.model(x_sim).logits
                                logits_with_noise_sim = add_gumbel_noise(logits_sim, temperature=temperature)

                                pred_ids_sim = torch.argmax(logits_with_noise_sim, dim=-1)

                            next_matched_count = 0

                            for future_pos in gen_positions:
                                fpos = future_pos.item()
                                if fpos == pos:
                                    continue

                                token_id = pred_ids_sim[0, fpos].item()

                                if check_watermark_compliance(fpos + 1, token_id, self.private_key) and (token_id not in [126081, self.mask_id]):
                                    next_matched_count += 1

                            if next_matched_count > best_match_count:
                                best_match_count = next_matched_count
                                best_pos = pos

                        selected_positions.append(best_pos)

                    elif unmatched:
                        selected_positions.append(unmatched[0][0]) 

                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                    selected_positions = gen_positions.tolist()[:1]
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
                x0 = torch.where(mask_index, x0, x)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for pos in selected_positions:
                    transfer_index[0, pos] = True

                x[transfer_index] = x0[transfer_index]

        return x