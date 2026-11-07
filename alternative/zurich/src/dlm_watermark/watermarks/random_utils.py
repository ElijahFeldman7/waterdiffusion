import torch


class OnTheFlyGreenlist:
    """We implement a stateless RNG to map (hash,token) to a greenlist value deterministically.
    This allows to not have to pre-compute the greenlist (faster but too high memory footprint) but do it at runtime in a vectorized manner.

    The logic is generate a uniform -> CDF-transform into the desired distribution.
    """

    def __init__(
        self,
        hash_size: int,
        vocab_size: int,
        mode: str = "gaussian",
        distrib_params: dict = {},
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.idx_offset = max(vocab_size, hash_size)
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.distrib_params = distrib_params

        self._C1 = torch.tensor(
            0x9E3779B97F4A7C15 - (1 << 64), dtype=torch.int64, device=device
        )
        self._C2 = torch.tensor(
            0xBF58476D1CE4E5B9 - (1 << 64), dtype=torch.int64, device=device
        )
        self._C3 = torch.tensor(
            0x94D049BB133111EB - (1 << 64), dtype=torch.int64, device=device
        )
        self._MASK53 = torch.tensor((1 << 53) - 1, dtype=torch.int64, device=device)


    def _splitmix64(self, z: torch.LongTensor) -> torch.LongTensor:
        z = z + self._C1
        z = (z ^ (z >> 30)) * self._C2
        z = (z ^ (z >> 27)) * self._C3
        return z ^ (z >> 31)

    def _uniform_from_keys(self, keys: torch.LongTensor) -> torch.Tensor:
        keys = keys.to(dtype=torch.int64, device=self.device)
        z = self._splitmix64(keys)
        u = (z & self._MASK53).to(torch.float64) / float(1 << 53)
        return u 

    def _get_final_distribution(self, U: torch.Tensor) -> torch.Tensor:
        """Get the final distribution based on the mode."""
        if self.mode == "gaussian":
            return self._normal_from_uniform(U)
        elif self.mode == "bernoulli":
            p = self.distrib_params["gamma"]
            return self._bernoulli_from_uniform(U, p)
        elif self.mode == "lognormal":
            return self._lognormal_from_uniform(U)
        elif self.mode == "uniform":
            return U
        else:
            raise ValueError(f"Unknown distribution mode: {self.mode}")

    def _normal_from_uniform(self, u: torch.Tensor) -> torch.Tensor:
        """Convert uniform [0,1) to normal N(0,1) using the inverse CDF (quantile function)."""
        normal_torch = torch.distributions.Normal(0, 1)
        norm01 = normal_torch.icdf(u)
        return norm01.to(self.dtype)

    def _bernoulli_from_uniform(self, u: torch.Tensor, p: float) -> torch.Tensor:
        """Convert uniform [0,1) to Bernoulli with probability p.

        Warning: Unlike default KGW, we do not enforce constraints on the covariance.
        """
        return (u < p).to(self.dtype)
    
    def _lognormal_from_uniform(self, u: torch.Tensor) -> torch.Tensor:
        """Convert uniform [0,1) to log-normal distribution."""
        normal = self._normal_from_uniform(u)
        lognorm01 = torch.exp(normal) 
        return lognorm01

    def _lookup_flat(
        self, h_flat: torch.LongTensor, v_flat: torch.LongTensor
    ) -> torch.Tensor:
        phi = h_flat + v_flat * self.idx_offset + int(self.seed)  
        u = self._uniform_from_keys(phi)
        final_distribution = self._get_final_distribution(u)

        return final_distribution.to(self.dtype)

    def lookup(
        self, h_inds: torch.LongTensor, v_inds: torch.LongTensor
    ) -> torch.Tensor:
        h_inds = h_inds.to(dtype=torch.int64, device=self.device)
        v_inds = v_inds.to(dtype=torch.int64, device=self.device)
        h_b, v_b = torch.broadcast_tensors(h_inds, v_inds)

        flat = self._lookup_flat(h_b.flatten(), v_b.flatten())
        return flat.view(h_b.shape)
    
    def sample(self, shape: tuple):
        """Sample shape from the distribution."""
        
        u = torch.rand(shape, dtype=self.dtype, device=self.device)

        final_distribution = self._get_final_distribution(u)
        
        return final_distribution
