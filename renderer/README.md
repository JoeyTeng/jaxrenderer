# Key Implementations of the Renderer

## Implementation Details

### Benchmarks

- Effect of back-face culling (very significant), see [Colab](https://colab.research.google.com/drive/1KOjeemLDPxMf-8H0ZpLgpd1DSXuFyNSK?usp=sharing)
- Effect of batched vs non-batched rendering on triangles. Non-batched pure `fori_loop`-based implementation is consistently faster, see [Colab](https://colab.research.google.com/drive/13p3US19TrVOTtLFkGKgg08KYzMb4747w?usp=sharing). This may be due to the extra GPU memory allocation required in `vmap`-ed implementation
- Effect of memory donation to suggest memory reuse, see [Colab](https://colab.research.google.com/drive/1VT7nvHV7au2oncMUjbZVNkPjZm0cN1wM?usp=sharing). The improvement is marginal.

### Why use `NamedTuple` but not `dataclasses`, etc

See experiment code [here in Colab](https://colab.research.google.com/drive/19b4VpAevvTVj_Ry9tEj88Q91ECXsku6z?usp=sharing). Basically tuples are well supported (and thus `NamedTuple`) by JAX as a [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) out of box, but `dataclass` is not.
