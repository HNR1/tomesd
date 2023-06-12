import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        torch.set_printoptions(precision=7)
        #print('metric', metric[0,0,:4])
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)
        print('score', scores[0,0,:4])
        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        print('max', node_max[0,100:104], node_idx[0,100:104])
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge

'''
hidden tensor([ 0.0954363,  0.3521819, -0.1353313, -0.0054430], device='cuda:0')
score tensor([0.7133359, 0.7488449, 0.6783174, 0.6451056], device='cuda:0')
max tensor([0.7488449, 0.7886865, 0.7980256, 0.8085438], device='cuda:0')
tensor([1, 2, 2, 3], device='cuda:0')
hidden tensor([ 0.0516011, -0.8971321,  0.0351196, -0.7181754], device='cuda:0')
score tensor([0.7151191, 0.6919521, 0.7272021, 0.7164218], device='cuda:0')
max tensor([0.7859746, 0.7893220, 0.7628933, 0.8069919], device='cuda:0')
tensor([22,  2, 14,  4], device='cuda:0')
hidden tensor([-0.1359314, -0.1188316,  0.1302420,  0.1066350], device='cuda:0')

hidden tensor([ 0.0954363,  0.3521819, -0.1353313, -0.0054430], device='cuda:0')
score tensor([0.7133359, 0.7488449, 0.6783174, 0.6451056], device='cuda:0')
max tensor([0.7488449, 0.7886865, 0.7980256, 0.8085438], device='cuda:0')
tensor([1, 2, 2, 3], device='cuda:0')
hidden tensor([ 0.0516009, -0.8971320,  0.0351195, -0.7181755], device='cuda:0')
score tensor([0.7151190, 0.6919520, 0.7272020, 0.7164217], device='cuda:0')
max tensor([0.7859744, 0.7893220, 0.7628933, 0.8069919], device='cuda:0')
tensor([22,  2, 14,  4], device='cuda:0')
hidden tensor([-0.1359320, -0.1188321,  0.1302425,  0.1066353], device='cuda:0')
'''
