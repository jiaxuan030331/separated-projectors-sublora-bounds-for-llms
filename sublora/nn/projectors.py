import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
import random
from functools import partial
import inspect
import re
import math

from sublora.nn.linear_operator_base import (
    Lazy,
    LazyKron,
    ConcatLazy,
    LazyPerm,
    LinearOperator,
    LazyDirectSum,
)


_DEFAULT_SEED = 137


def _getchainattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes:
        obj = getattr(obj, a)
    return obj


def _delchainattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    try:
        delattr(obj, attributes[-1])
    except AttributeError:
        raise


def _setchainattr(obj, attr, value):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attributes[-1], value)


def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.contiguous().view(t.numel()))
    return torch.cat(flatList)


def unflatten_like(vector, likeTensorList):
    outList = []
    i = 0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i: i + n].view(tensor.shape))
        i += n
    return outList


class QuantizingWrapper(nn.Module):
    def __init__(self, net, centroids, assignments):
        super().__init__()
        self.subspace_params = deepcopy(net.subspace_params)
        _delchainattr(net, "subspace_params")

        self._forward_net = [net]
        self.centroids = [centroids]
        self.assignments = assignments

    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        _setchainattr(self._forward_net[0], "subspace_params", self.subspace_params)
        return self._forward_net[0](*args, **kwargs)


class FixedPytorchSeed(object):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.pt_rng_state = torch.random.get_rng_state()
        self.cuda_rng_state = torch.cuda.get_rng_state_all()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, *_):
        torch.random.set_rng_state(self.pt_rng_state)
        torch.cuda.set_rng_state_all(self.cuda_rng_state)


class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)


class IDModule(nn.Module):
    """Intrinsic dimensionality wrapper module..
    Takes in the network, a projector (a function(D,d)-> projection LinearOperator),
    and the target intrinsic dimensionality.

    Example usage:
    id_net = IDModule(net, lambda D,d: LazyRandom(D,d), 1000)
    """

    def __init__(self, net, projector, dimension=1000):
        super().__init__()

        self.d = dimension
        self._forward_net = [net]
        initnet = deepcopy(net)
        for orig_name, orig_p in initnet.named_parameters():
            if orig_p.requires_grad:
                _delchainattr(net, orig_name)
        aux = [(n, p) for n, p in initnet.named_parameters() if p.requires_grad]
        self.names, self.trainable_initparams = zip(*aux)
        self.trainable_initparams = [param for param in self.trainable_initparams]
        self.names = list(self.names)
        self.D = sum([param.numel() for param in self.trainable_initparams])
        self.subspace_params = nn.Parameter(torch.zeros(self.d))
        self.P = projector(self.D, self.d, self.trainable_initparams, self.names)

    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        self.trainable_initparams = [
            param.to(*args, **kwargs) for param in self.trainable_initparams
        ]
        return super().to(*args, **kwargs)
    
    
    def get_num_params(self, only_trainable=False):
        if only_trainable:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return n_params
        else:
            n_params = sum(p.numel() for p in self.parameters())    
            return n_params
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, correct_bias, adam_epislon, no_decay_bias):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Separate gating params (will get a special LR multiplier)
        gating_param_names = [pn for pn in param_dict.keys() if pn.startswith('gating_params')]
        gating_params = [param_dict[pn] for pn in gating_param_names]

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        optim_groups = []
        if not no_decay_bias:
            # all params decayed except gating handled separately
            normal_params = [p for pn, p in param_dict.items() if not pn.startswith('gating_params')]
            optim_groups.append({'params': normal_params, 'weight_decay': weight_decay})
            if gating_params:
                optim_groups.append({'params': gating_params, 'weight_decay': weight_decay, 'lr': learning_rate * self.gating_lr_mult})
            print("using all params (with gating params grouped if present)")
        else:
            # split into decayed and non-decayed, then move gating params into dedicated groups
            decay_params = [p for pn, p in param_dict.items() if (p.dim() >= 2 and not pn.startswith('gating_params'))]
            nodecay_params = [p for pn, p in param_dict.items() if (p.dim() < 2 and not pn.startswith('gating_params'))]

            if decay_params:
                optim_groups.append({'params': decay_params, 'weight_decay': weight_decay})
            if nodecay_params:
                optim_groups.append({'params': nodecay_params, 'weight_decay': 0.0})

            # gating params: separate into decay vs nodecay depending on dim
            if gating_params:
                gating_decay = [p for p in gating_params if p.dim() >= 2]
                gating_nodecay = [p for p in gating_params if p.dim() < 2]
                if gating_decay:
                    optim_groups.append({'params': gating_decay, 'weight_decay': weight_decay, 'lr': learning_rate * self.gating_lr_mult})
                if gating_nodecay:
                    optim_groups.append({'params': gating_nodecay, 'weight_decay': 0.0, 'lr': learning_rate * self.gating_lr_mult})

            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=adam_epislon, **extra_args)

        print(f"using fused AdamW: {use_fused}")
        if gating_param_names:
            print(f"gating params grouped ({len(gating_param_names)} tensors), gating_lr_mult={self.gating_lr_mult}")

        return optimizer

    def forward(self, *args, **kwargs):
        flat_projected_params = self.P @ self.subspace_params
        unflattened_params = unflatten_like(
            flat_projected_params, self.trainable_initparams
        )
        iterables = zip(self.names, self.trainable_initparams, unflattened_params)
        for p_name, init, proj_param in iterables:
            p = init + proj_param.view(*init.shape)
            _setchainattr(self._forward_net[0], p_name, p)
        return self._forward_net[0](*args, **kwargs)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self._forward_net[0].config.block_size else idx[:, -self._forward_net[0].config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class RandomMultiply(torch.autograd.Function):
    CHUNK_MAX = 2e8

    @staticmethod
    def forward(ctx, v, D, d, seed):
        ctx.info = (D, d, seed)
        ctx.dtype = v.dtype  # Save dtype for backward pass
        with FixedPytorchSeed(seed):
            D_chunks = int(np.ceil((D * d) / RandomMultiply.CHUNK_MAX))
            D_chunksize = D // D_chunks
            D_tot = 0
            Pv_chunks = []
            while D_tot < D:
                D_chunk = min(D_chunksize, D - D_tot)
                D_tot += D_chunk
                Pv_chunks.append(
                    torch.randn(D_chunk, d, device=v.device, dtype=v.dtype) @ v / np.sqrt(D)
                )
            Pv = torch.cat(Pv_chunks, dim=0)
        return Pv

    @staticmethod
    def backward(ctx, grad_output):

        D, d, seed = ctx.info
        dtype = grad_output.dtype  # Use grad_output's dtype directly
        grad_in = 0.0
        with FixedPytorchSeed(seed):
            D_chunks = int(np.ceil((D * d) / RandomMultiply.CHUNK_MAX))
            D_chunksize = D // D_chunks
            split_grad_outs = torch.split(grad_output, D_chunksize, dim=0)
            for grad_out in split_grad_outs:
                grad_in += (
                    torch.randn(grad_out.shape[0], d, device=grad_output.device, dtype=dtype).T
                    @ grad_out
                    / np.sqrt(D)
                )
        return grad_in, None, None, None


class LazyRandom(LinearOperator):
    def __init__(self, D, d, params, names, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        self.info = (D, d, seed)

    def _matvec(self, v):
        return RandomMultiply.apply(v, *self.info)

    def _matmat(self, v):
        return RandomMultiply.apply(v, *self.info)

    def __repr__(self):
        D, d, seed = self.info
        return f"LazyRandom({D}, {d}, seed={seed})"


class LazyRandomQR(LinearOperator):
    def __init__(self, D, d, params, names, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        self.info = (D, d, seed)
        self.P = torch.randn(D, d)
        self.P, _ = torch.linalg.qr(self.P, mode="reduced")
        self._cached_P = None  # Cache for device/dtype converted matrix

    def _get_P(self, v):
        """Get P matrix on the same device and dtype as v, with caching."""
        target_device = v.device
        target_dtype = v.dtype

        if self._cached_P is not None:
            if self._cached_P.device == target_device and self._cached_P.dtype == target_dtype:
                return self._cached_P

        # Convert and cache
        self._cached_P = self.P.to(device=target_device, dtype=target_dtype)
        return self._cached_P

    def _matvec(self, v):
        return self._get_P(v) @ v

    def _matmat(self, v):
        return self._get_P(v) @ v

    def __repr__(self):
        D, d, seed = self.info
        return f"LazyRandomQR({D}, {d}, seed={seed})"


class LazyOneSidedKron(LinearOperator):
    def __init__(self, D, d, params, names, order=2, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        self.seed = seed
        assert np.floor(D ** (1 / order)) == D ** (1 / order)
        self.order = order

    def _matvec(self, v):
        seed = self.seed
        k = int(self.shape[0] ** (1 / self.order))
        out_tensor = torch.zeros(*(self.order * [k]), device=v.device)
        for i in range(self.order):
            Pvi = RandomMultiply.apply(v, k, self.shape[-1], seed) / (
                np.sqrt(self.order) * np.sqrt(k) ** (self.order - 1)
            )
            # unsqueeze all axes except i
            for j in range(self.order):
                if j != i:
                    Pvi = Pvi.unsqueeze(j)
            out_tensor += Pvi
            # re randomize/ advance the seed
            with FixedPytorchSeed(seed):
                seed = int(torch.randint(high=2**31, size=(1,))[0])
        return out_tensor


def RoundedKron(D, d, params, names, order=2, seed=_DEFAULT_SEED):
    rounded_D = int(np.floor(D ** (1 / order))) ** order
    with FixedPytorchSeed(seed):
        fitting_kron = LazyOneSidedKron(rounded_D, d, params, names, order, seed)
        perm = torch.randperm(D)
        if rounded_D == D:
            return LazyPerm(perm) @ fitting_kron
        else:
            newseed = int(torch.randint(high=2**31, size=(1,))[0])
            leftover_random = LazyRandom(D - rounded_D, d, params, names, newseed) * (
                1 / np.sqrt(D / (D - rounded_D))
            )
            return LazyPerm(perm) @ ConcatLazy([fitting_kron, leftover_random])


def RoundedDoubleKron(D, d, params, names, order=2, seed=_DEFAULT_SEED):
    rounded_D = int(np.floor(D ** (1 / order)))
    rounded_d = int(np.floor(d ** (1 / order)))

    with FixedPytorchSeed(seed):
        seed = int(torch.randint(high=2**31, size=(1,))[0])
        Rs = []
        for i in range(order):
            Rs.append(LazyRandom(rounded_D, rounded_d, params, names, seed))
            seed = int(torch.randint(high=2**31, size=(1,))[0])
        RkR = LazyKron(Rs)
        if rounded_D**order == D or rounded_d**order == d:
            extra = Lazy(
                torch.randn(D - rounded_D**order, d - rounded_d**order) / np.sqrt(D)
            )
        else:
            extra = LazyRandom(
                D - rounded_D**order, d - rounded_d**order, params, names, seed
            )

        M = LazyDirectSum([RkR, extra])
        perm = torch.randperm(D)

    return LazyPerm(perm) @ M


def RoundedDoubleKronQR(D, d, params, names, order=2, seed=_DEFAULT_SEED):
    rounded_D = int(np.floor(D ** (1 / order)))
    rounded_d = int(np.floor(d ** (1 / order)))

    with FixedPytorchSeed(seed):
        seed = int(torch.randint(high=2**31, size=(1,))[0])
        Rs = []
        for i in range(order):
            Rs.append(LazyRandomQR(rounded_D, rounded_d, params, names, seed))
            seed = int(torch.randint(high=2**31, size=(1,))[0])
        RkR = LazyKron(Rs)
        if rounded_D**order == D or rounded_d**order == d:
            extra = Lazy(
                torch.randn(D - rounded_D**order, d - rounded_d**order) / np.sqrt(D)
            )
        else:
            extra = LazyRandom(
                D - rounded_D**order, d - rounded_d**order, params, names, seed
            )

        M = LazyDirectSum([RkR, extra])
        perm = torch.randperm(D)

    return LazyPerm(perm) @ M


def FiLMLazyRandom(D, d, params, names, seed=_DEFAULT_SEED):
    def bn_or_fc(name):
        return (
            ("bn" in name)
            or ("fc" in name)
            or ("norm" in name)
            or ("classifier" in name)
        )

    return FilterLazyRandom(D, d, params, names, bn_or_fc, seed)


class FilterLazyRandom(LinearOperator):
    def __init__(self, D, d, params, names, condition, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        i = 0
        ids = []
        for name, param in zip(names, params):
            if condition(name):
                ids.append(np.arange(i, i + param.numel()))
            i += param.numel()
        self.ids = np.concatenate(ids)
        assert len(ids) > 0
        assert i == D
        self.dense_random = LazyRandom(len(self.ids), d, params, names, seed)
        print(D, len(self.ids), d)

    def _matvec(self, v):
        filtered_v_params = self.dense_random @ v
        out = torch.zeros(self.shape[0], device=v.device, dtype=v.dtype)
        out[self.ids] = filtered_v_params
        return out


class LazySTFiLMRDKronQR(LinearOperator):
    def __init__(self, D, d, params, names, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        def condition_fn1(x): return x.find('bn') >= 0
        def condition_fn2(x): return not condition_fn1(x)
        ids1 = find_locations_from_condition(names, params, condition_fn1)
        ids2 = find_locations_from_condition(names, params, condition_fn2)
        self.bn_d = len(ids1)
        self.ids = np.argsort(np.concatenate([ids1, ids2]))
        self.P = RoundedDoubleKronQR(D - self.bn_d, d - self.bn_d, params, names)

    def _matvec(self, v):
        return self._matmat(v)

    def _matmat(self, v):
        v1, v2 = v[:self.bn_d], v[self.bn_d:]
        output = torch.concat([v1, self.P @ v2])
        return output[self.ids]


def find_locations_from_condition(names, params, condition_fn):
    i, ids = 0, []
    for name, param in zip(names, params):
        if condition_fn(name):
            ids.append(np.arange(i, i + param.numel()))
        i += param.numel()
    ids = np.concatenate(ids)
    return ids


def find_all_batch_norm(net):
    leaf_criteria = (nn.BatchNorm1d, nn.BatchNorm2d)

    class Counter:
        count = 0

        def count_params_in_module(self, x):
            print(x)
            for y in list(x.parameters()):
                self.count += y.numel()

    counter = Counter()
    # TODO: check if this is the correct way to pass modules
    selective_apply(list(net.modules())[0], counter, leaf_criteria)
    return counter.count


def is_leaf(module, leaf_criteria):
    no_children_att = not hasattr(module, 'children')
    no_children = not list(module.children())
    is_leaf_criteria = isinstance(module, leaf_criteria)
    return no_children_att or no_children or is_leaf_criteria


def selective_apply(module, counter, leaf_criteria):
    if is_leaf(module, leaf_criteria):
        if isinstance(module, leaf_criteria):
            counter.count_params_in_module(module)
    else:
        for c in module.children():
            selective_apply(c, counter, leaf_criteria)


def CombinedRDKronFiLM(D, d, params, names, seed=_DEFAULT_SEED):
    rdkron = RoundedDoubleKron(D, d, params, names, seed=seed)
    FiLM = FiLMLazyRandom(D, d, params, names, seed=seed)

    return (rdkron + FiLM) * (1 / np.sqrt(2))


def CombinedRDKronQRFiLM(D, d, params, names, seed=_DEFAULT_SEED):
    rdkronqr = RoundedDoubleKronQR(D, d, params, names, seed=seed)
    FiLM = FiLMLazyRandom(D, d, params, names, seed=seed)

    return (rdkronqr + FiLM) * (1 / np.sqrt(2))


class SparseOperator(LinearOperator):
    def __init__(self, D, d, params, names, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))
        s = np.sqrt(D)
        with FixedNumpySeed(seed):
            number_nonzero = np.random.binomial(D * d, 1.0 / s)
            # print(number_nonzero)
            nonzero_indices = np.random.choice(D * d, number_nonzero)
            nonzero_indices2d = np.stack(
                np.unravel_index(nonzero_indices, (D, d)), axis=0
            )
            # sample values from +-1
            nonzero_values = np.random.choice([-1, 1], number_nonzero) / np.sqrt(s)
            self.V = torch.sparse_coo_tensor(
                nonzero_indices2d, nonzero_values, size=(D, d)
            ).float()

    def _matvec(self, x):
        assert x.shape[0] == self.shape[-1], f"{x.shape[0]} != {self.shape[-1]}"
        return self.V.to(x.device) @ x


class FastfoodOperator(LinearOperator):
    # Source: https://discuss.pytorch.org/t/fast-walsh-hadamard-transform/19341
    class FWHT(torch.autograd.Function):
        @staticmethod
        def transform(x):
            bit = dd = x.size(-1)
            result = x.detach().cpu().numpy()

            for _ in range(int(np.log2(dd))):
                bit >>= 1
                for i in range(dd):
                    if i & bit == 0:
                        j = i | bit
                        temp = np.copy(result[..., i])
                        result[..., i] += result[..., j]
                        result[..., j] = temp - result[..., j]

            result /= np.sqrt(dd)
            return torch.from_numpy(result).to(x.device)

        @staticmethod
        def forward(_, inputs):
            return FastfoodOperator.FWHT.transform(inputs)

        @staticmethod
        def backward(_, grad_outputs):
            return FastfoodOperator.FWHT.transform(grad_outputs)

    def __init__(self, D, d, params, names, scale=1, seed=_DEFAULT_SEED):
        super().__init__(None, (D, d))

        self.D = D
        self.real_d = d
        self.d = 2 ** np.ceil(np.log2(d)).astype(int)
        self.sigma = scale
        blocks = np.ceil(self.D / self.d).astype(int)
        with FixedPytorchSeed(seed):
            self.S = torch.rand(blocks, self.d)
            self.G = torch.randn(blocks, self.d)
            self.B = 2 * (torch.rand(blocks, self.d) > 0.5).float() - 1
            self.Pi = torch.randperm(self.d)

    def _matvec(self, x):
        """Implicit P @ x
        Assumed x is 1-D tensor.
        """
        device = x.device

        pad = self.d - self.real_d
        if pad > 0:
            x = torch.cat([x, torch.zeros(pad, device=device)], dim=-1)

        GPiHBx = (
            self.G.to(device)
            * FastfoodOperator.FWHT.apply(self.B.to(device) * x)[..., self.Pi]
        )
        SHGPiHBx = self.S.to(device) * FastfoodOperator.FWHT.apply(GPiHBx)
        result = SHGPiHBx.flatten()[: self.D] / (self.sigma * np.sqrt(self.d))
        return result


def create_intrinsic_model(
    base_net,
    ckpt_path=None,
    intrinsic_mode="dense",
    intrinsic_dim=1000,
    seed=None,
    device=None,
    allocation_config=None,
):
    if seed is None:
        raise ValueError(
            "Missing seed. Randomized projections will not be reproducible!"
        )

    net = None
    
    # Check if we should use StructuredIDModule
    use_structured = allocation_config is not None and allocation_config.get('mode') in ['fixed', 'learned']

    if intrinsic_mode == "dense":
        if use_structured:
            net = StructuredIDModule(
                base_net, 
                partial(LazyRandom, seed=seed), 
                dimension=intrinsic_dim,
                allocation_config=allocation_config
            )
        else:
            class DenseIDNet(IDModule):
                def __init__(self, net, dimension=1000, seed=None, **_):
                    super().__init__(
                        net, partial(LazyRandom, seed=seed), dimension=dimension
                    )
            net = DenseIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    elif intrinsic_mode == "sparse":
        if use_structured:
            net = StructuredIDModule(
                base_net, 
                partial(SparseOperator, seed=seed), 
                dimension=intrinsic_dim,
                allocation_config=allocation_config
            )
        else:
            class SparseIDNet(IDModule):
                def __init__(self, net, dimension=1000, seed=None, **_):
                    super().__init__(
                        net, partial(SparseOperator, seed=seed), dimension=dimension
                    )
            net = SparseIDNet(base_net, dimension=intrinsic_dim, seed=seed)


    elif intrinsic_mode == "fastfood":
        class FastfoodIDNet(IDModule):
            def __init__(self, net, dimension=1000, seed=None, **_):
                super().__init__(
                    net, partial(FastfoodOperator, seed=seed), dimension=dimension
                )
        net = FastfoodIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    elif intrinsic_mode == "rkron":
        class RoundedKronIDNet(IDModule):
            def __init__(self, net, dimension=1000, order=2, seed=None, **_):
                super().__init__(
                    net,
                    partial(RoundedKron, order=order, seed=seed),
                    dimension=dimension,
                )
        net = RoundedKronIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    elif intrinsic_mode == "rdkron":
        class RoundedDoubleKronIDNet(IDModule):
            def __init__(self, net, dimension=1000, order=2, seed=None, **_):
                super().__init__(
                    net,
                    partial(RoundedDoubleKron, order=order, seed=seed),
                    dimension=dimension,
                )
        net = RoundedDoubleKronIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    elif intrinsic_mode == "rdkronqr":
        if use_structured:
            net = StructuredIDModule(
                base_net, 
                partial(RoundedDoubleKronQR, order=2, seed=seed), 
                dimension=intrinsic_dim,
                allocation_config=allocation_config
            )
        else:
            class RoundedDoubleKronQRIDNet(IDModule):
                def __init__(self, net, dimension=1000, order=2, seed=None, **_):
                    super().__init__(
                        net,
                        partial(RoundedDoubleKronQR, order=order, seed=seed),
                        dimension=dimension,
                    )
            net = RoundedDoubleKronQRIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    elif intrinsic_mode == "film":
        class FiLMIDNet(IDModule):
            def __init__(self, net, dimension=1000, seed=None, **_):
                super().__init__(
                    net, partial(FiLMLazyRandom, seed=seed), dimension=dimension
                )
        net = FiLMIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    elif intrinsic_mode == "filmrdkron":
        class FiLMRDKronIDNet(IDModule):
            def __init__(self, net, dimension=1000, seed=None, **_):
                super().__init__(
                    net, partial(CombinedRDKronFiLM, seed=seed), dimension=dimension
                )
        net = FiLMRDKronIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    elif intrinsic_mode == "filmrdkronqr":
        class FiLMRDKronQRIDNet(IDModule):
            def __init__(self, net, dimension=1000, seed=None, **_):
                super().__init__(
                    net, partial(CombinedRDKronQRFiLM, seed=seed), dimension=dimension
                )
        net = FiLMRDKronQRIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    elif intrinsic_mode == "stfilmkronqr":
        class STFiLMRDKronQRIDNet(IDModule):
            def __init__(self, net, dimension=1000, seed=None, **_):
                super().__init__(
                    net, partial(LazySTFiLMRDKronQR, seed=seed), dimension=dimension
                )
        net = STFiLMRDKronQRIDNet(base_net, dimension=intrinsic_dim, seed=seed)

    else:
        raise NotImplementedError

    if ckpt_path is not None:
        weights = torch.load(ckpt_path)
        if "subspace_params" in weights:
            net.load_state_dict(weights)
        else:
            tmp = {}
            tmp["subspace_params"] = weights["module.subspace_params"]
            net.load_state_dict(tmp)
    return net


class StructuredIDModule(nn.Module):
    """
    Structured Intrinsic Dimensionality Module.
    Allows for asymmetric allocation of subspace dimensions to different parameter groups (e.g. A vs B matrices).
    """
    def __init__(self, net, projector_factory, dimension=1000, allocation_config=None):
        super().__init__()
        self.d = dimension
        self.allocation_config = allocation_config or {}
        self.mode = self.allocation_config.get('mode', 'uniform') # uniform, fixed, learned
        self.ratio = self.allocation_config.get('ratio', 0.5) # for fixed, d_B / (d_A + d_B)
        # gating_scale controls the steepness of the sigmoid used for soft masking.
        # You can provide either:
        #  - 'gating_scale' in `allocation_config` to set an explicit scale, or
        #  - 'gating_fraction' in `allocation_config` to request that the sigmoid
        #    transition span a fraction of the per-layer block size `d_per_layer`.
        # If neither is provided, we fall back to a reasonable default of 5.
        self.gating_scale = self.allocation_config.get('gating_scale', None)
        self.gating_fraction = self.allocation_config.get('gating_fraction', None)
        # initialization std for gating params (small noise to break symmetry)
        self.gating_init_std = float(self.allocation_config.get('gating_init_std', 0.01))
        # multiplier for gating learning rate (creates a separate optimizer group)
        self.gating_lr_mult = float(self.allocation_config.get('gating_lr_mult', 5.0))
        # Optional gating annealing configuration. If provided, should be a dict with keys:
        #   'start': initial scale (float) or None to use current computed scale
        #   'end': final scale (float)
        #   'total_steps': number of steps over which to anneal (int)
        #   'mode': 'linear' or 'cosine' (default 'linear')
        #   'auto_step': if True, automatically step the annealer on each forward() call (default True)
        self.gating_anneal = self.allocation_config.get('gating_anneal', None)
        # Internal anneal step counter
        self._anneal_step = 0
        
        self._forward_net = [net]
        
        # Extract parameters and remove from net
        initnet = deepcopy(net)
        for orig_name, orig_p in initnet.named_parameters():
            if orig_p.requires_grad:
                _delchainattr(net, orig_name)
        
        self.all_params = [(n, p) for n, p in initnet.named_parameters() if p.requires_grad]
        
        # Compatibility attributes
        self.names = [n for n, p in self.all_params]
        self.trainable_initparams = [p for n, p in self.all_params]
        
        # Group parameters
        self.param_groups = self._group_parameters(self.all_params)
        
        # Merge 'other' into 'A' to ensure they are projected
        self.param_groups['A']['items'].extend(self.param_groups['other']['items'])
        
        # Initialize subspace parameters as a SINGLE tensor for compatibility with quantize_model
        self.subspace_params = nn.Parameter(torch.empty(self.d))
        # Kaiming-style initialization for a 1-D subspace vector: approximate
        # kaiming_uniform by treating the vector as a (1, d) weight with fan_in=d.
        # bound = sqrt(3) * gain / sqrt(fan_in), with gain=1.0 for linear.
        fan_in = max(1, int(self.d))
        gain = nn.init.calculate_gain('linear')
        bound = math.sqrt(3.0) * gain / math.sqrt(fan_in)
        nn.init.uniform_(self.subspace_params, a=-bound, b=bound)

        self.projectors = {} # Not nn.ModuleDict because projectors are not Modules
        self.gating_params = nn.ParameterDict() # For learned gating
        
        # Store parameter lists for reconstruction
        self.params_A = [p for n, p in self.param_groups['A']['items']]
        self.names_A = [n for n, p in self.param_groups['A']['items']]
        self.params_B = [p for n, p in self.param_groups['B']['items']]
        self.names_B = [n for n, p in self.param_groups['B']['items']]
        
        if self.mode == 'learned':
            # Layer-wise learned gating
            # We assume d is split equally among layers for the base allocation
            
            # First, identify layers and misc params
            layers = set()
            misc_params_exist = False
            
            for name, _ in self.all_params:
                match = re.search(r'\.h\.(\d+)\.', name)
                if match:
                    layers.add(int(match.group(1)))
                else:
                    misc_params_exist = True
            
            self.layers = sorted(list(layers))
            self.has_misc = misc_params_exist
            
            num_groups = len(self.layers) + (1 if self.has_misc else 0)

            # Compute per-group (per-layer + misc) parameter counts so we can
            # allocate intrinsic dimensions proportionally rather than equally.
            group_D = []
            group_keys = []

            for layer_idx in self.layers:
                layer_key = str(layer_idx)
                layer_params = [p for n, p in self.param_groups['A']['items'] if f'.h.{layer_idx}.' in n] + [p for n, p in self.param_groups['B']['items'] if f'.h.{layer_idx}.' in n]
                D_layer = sum(p.numel() for p in layer_params)
                group_keys.append(layer_key)
                group_D.append(max(0, D_layer))

            if self.has_misc:
                misc_params = [p for n, p in self.param_groups['A']['items'] if not re.search(r'\.h\.(\d+)\.', n)] + [p for n, p in self.param_groups['B']['items'] if not re.search(r'\.h\.(\d+)\.', n)]
                D_misc = sum(p.numel() for p in misc_params)
                group_keys.append('misc')
                group_D.append(max(0, D_misc))

            total_group_D = sum(group_D) if sum(group_D) > 0 else num_groups

            # Allocate integer intrinsic dims per group proportional to parameter counts.
            # Ensure at least 1 per group, then distribute remainder.
            raw_alloc = [max(1, int(round(self.d * (d_i / total_group_D)))) for d_i in group_D]
            alloc_sum = sum(raw_alloc)
            # fix rounding errors: adjust by distributing difference
            diff = self.d - alloc_sum
            i = 0
            while diff != 0:
                idx = i % len(raw_alloc)
                if diff > 0:
                    raw_alloc[idx] += 1
                    diff -= 1
                else:
                    if raw_alloc[idx] > 1:
                        raw_alloc[idx] -= 1
                        diff += 1
                i += 1

            # Map allocations to layer keys for later use
            self.d_alloc_map = {k: int(v) for k, v in zip(group_keys, raw_alloc)}
            # Keep a list in the same order as self.layers followed by misc (if present)
            self.d_alloc_order = [self.d_alloc_map[k] for k in group_keys]

            # For backward compatibility, set a nominal d_per_layer as the integer mean
            self.d_per_layer = int(sum(self.d_alloc_order) // max(1, len(self.d_alloc_order)))

            # If gating_scale wasn't explicitly provided, compute it from gating_fraction
            # so that the sigmoid transition width is approximately `gating_fraction * d_per_layer`.
            # Use the mean per-group allocation for the computation so gating_scale is reasonable.
            mean_d_group = max(1, self.d_per_layer)
            if self.gating_scale is None:
                if self.gating_fraction is not None:
                    try:
                        f = float(self.gating_fraction)
                    except Exception:
                        f = None
                    if f is not None and f > 0:
                        # avoid division by zero, clamp minimum fraction
                        f = max(f, 1e-4)
                        computed = 9.2 / (f * mean_d_group)
                        # keep the scale in a reasonable numeric range
                        self.gating_scale = float(max(0.01, min(computed, 1e3)))
                    else:
                        # fallback default
                        self.gating_scale = 5.0
                else:
                    # fallback default if neither gating_scale nor fraction provided
                    self.gating_scale = 5.0
            
            # Create params for each layer using the computed per-group allocation
            for idx, layer_idx in enumerate(self.layers):
                layer_key = str(layer_idx)

                # Find params for this layer
                layer_params_A = [p for n, p in self.param_groups['A']['items'] if f'.h.{layer_idx}.' in n]
                layer_names_A = [n for n, p in self.param_groups['A']['items'] if f'.h.{layer_idx}.' in n]
                layer_params_B = [p for n, p in self.param_groups['B']['items'] if f'.h.{layer_idx}.' in n]
                layer_names_B = [n for n, p in self.param_groups['B']['items'] if f'.h.{layer_idx}.' in n]

                if not layer_params_A and not layer_params_B:
                    continue

                # Gating parameter for this layer (small random init to break symmetry)
                self.gating_params[layer_key] = nn.Parameter(torch.randn(1) * self.gating_init_std)

                # Projectors: use the allocated d for this group
                d_alloc = self.d_alloc_order[idx]
                D_A = sum(p.numel() for p in layer_params_A)
                D_B = sum(p.numel() for p in layer_params_B)

                if D_A > 0:
                    self.projectors[f'{layer_key}_A'] = projector_factory(D_A, d_alloc, layer_params_A, layer_names_A)
                if D_B > 0:
                    self.projectors[f'{layer_key}_B'] = projector_factory(D_B, d_alloc, layer_params_B, layer_names_B)

            # Create params for misc group (if present). The misc allocation is last in d_alloc_order
            if self.has_misc:
                layer_key = 'misc'

                # Find params NOT in any layer
                layer_params_A = [p for n, p in self.param_groups['A']['items'] if not re.search(r'\.h\.(\d+)\.', n)]
                layer_names_A = [n for n, p in self.param_groups['A']['items'] if not re.search(r'\.h\.(\d+)\.', n)]
                layer_params_B = [p for n, p in self.param_groups['B']['items'] if not re.search(r'\.h\.(\d+)\.', n)]
                layer_names_B = [n for n, p in self.param_groups['B']['items'] if not re.search(r'\.h\.(\d+)\.', n)]

                self.gating_params[layer_key] = nn.Parameter(torch.randn(1) * self.gating_init_std)

                d_alloc = self.d_alloc_order[-1]
                D_A = sum(p.numel() for p in layer_params_A)
                D_B = sum(p.numel() for p in layer_params_B)

                if D_A > 0:
                    self.projectors[f'{layer_key}_A'] = projector_factory(D_A, d_alloc, layer_params_A, layer_names_A)
                if D_B > 0:
                    self.projectors[f'{layer_key}_B'] = projector_factory(D_B, d_alloc, layer_params_B, layer_names_B)

        else:
            # Fixed Global Split
            # ratio is d_B / d
            self.d_B = int(self.d * self.ratio)
            self.d_A = self.d - self.d_B
            
            D_A = sum(p.numel() for p in self.params_A)
            D_B = sum(p.numel() for p in self.params_B)
            
            if D_A > 0 and self.d_A > 0:
                self.projectors['A'] = projector_factory(D_A, self.d_A, self.params_A, self.names_A)
            
            if D_B > 0 and self.d_B > 0:
                self.projectors['B'] = projector_factory(D_B, self.d_B, self.params_B, self.names_B)

    def _group_parameters(self, all_params):
        groups = {
            'A': {'items': []},
            'B': {'items': []},
            'other': {'items': []}
        }
        
        for n, p in all_params:
            if 'lora_A' in n:
                groups['A']['items'].append((n, p))
            elif 'lora_B' in n:
                groups['B']['items'].append((n, p))
            else:
                groups['other']['items'].append((n, p))
        return groups

    def forward(self, *args, **kwargs):
        if self.mode == 'learned':
            # If annealing is configured and auto_step enabled, advance the annealer.
            if self.gating_anneal is not None and self.gating_anneal.get('auto_step', True):
                try:
                    self.step_gating_anneal()
                except Exception:
                    # be robust: don't break forward pass if annealer misconfigured
                    pass
            # Layer-wise learned gating with Soft Masking

            # Helper to process a group
            def process_group(layer_key, alpha, gamma, d_alloc):
                # Soft Masking using the actual allocation size for this group
                indices = torch.arange(d_alloc, device=alpha.device, dtype=alpha.dtype)
                k = gamma * d_alloc
                mask = torch.sigmoid(self.gating_scale * (k - indices))

                alpha_B = alpha * mask
                alpha_A = alpha * (1 - mask)

                if f'{layer_key}_A' in self.projectors:
                    proj = self.projectors[f'{layer_key}_A']
                    # Defensive check: ensure projector expects the same input dim as alpha
                    expected_d = None
                    try:
                        expected_d = proj.shape[-1]
                    except Exception:
                        pass
                    if expected_d is None:
                        P_attr = getattr(proj, 'P', None)
                        if P_attr is not None and hasattr(P_attr, 'shape'):
                            expected_d = P_attr.shape[1]

                    if expected_d is not None and alpha_A.numel() != int(expected_d):
                        raise RuntimeError(
                            f"Projector input dim mismatch for {layer_key}_A: "
                            f"projector expects d={expected_d}, alpha length={alpha_A.numel()}, "
                            f"d_alloc={d_alloc}, d_alloc_order={self.d_alloc_order}, "
                            f"projectors={list(self.projectors.keys())}"
                        )

                    flat_A = proj @ alpha_A
                    if layer_key == 'misc':
                        layer_params_A = [p for n, p in self.param_groups['A']['items'] if not re.search(r'\.h\.(\d+)\.', n)]
                        layer_names_A = [n for n, p in self.param_groups['A']['items'] if not re.search(r'\.h\.(\d+)\.', n)]
                    else:
                        layer_params_A = [p for n, p in self.param_groups['A']['items'] if f'.h.{layer_key}.' in n]
                        layer_names_A = [n for n, p in self.param_groups['A']['items'] if f'.h.{layer_key}.' in n]
                    self._set_params(flat_A, layer_names_A, layer_params_A)

                if f'{layer_key}_B' in self.projectors:
                    proj = self.projectors[f'{layer_key}_B']
                    # Defensive check: ensure projector expects the same input dim as alpha
                    expected_d = None
                    try:
                        expected_d = proj.shape[-1]
                    except Exception:
                        pass
                    if expected_d is None:
                        P_attr = getattr(proj, 'P', None)
                        if P_attr is not None and hasattr(P_attr, 'shape'):
                            expected_d = P_attr.shape[1]

                    if expected_d is not None and alpha_B.numel() != int(expected_d):
                        raise RuntimeError(
                            f"Projector input dim mismatch for {layer_key}_B: "
                            f"projector expects d={expected_d}, alpha length={alpha_B.numel()}, "
                            f"d_alloc={d_alloc}, d_alloc_order={self.d_alloc_order}, "
                            f"projectors={list(self.projectors.keys())}"
                        )

                    flat_B = proj @ alpha_B
                    if layer_key == 'misc':
                        layer_params_B = [p for n, p in self.param_groups['B']['items'] if not re.search(r'\.h\.(\d+)\.', n)]
                        layer_names_B = [n for n, p in self.param_groups['B']['items'] if not re.search(r'\.h\.(\d+)\.', n)]
                    else:
                        layer_params_B = [p for n, p in self.param_groups['B']['items'] if f'.h.{layer_key}.' in n]
                        layer_names_B = [n for n, p in self.param_groups['B']['items'] if f'.h.{layer_key}.' in n]
                    self._set_params(flat_B, layer_names_B, layer_params_B)

            # Process layers using variable allocations from d_alloc_order
            cumulative_idx = 0
            for i, layer_idx in enumerate(self.layers):
                layer_key = str(layer_idx)
                d_alloc = self.d_alloc_order[i]
                start_idx = cumulative_idx
                end_idx = start_idx + d_alloc
                alpha = self.subspace_params[start_idx:end_idx]
                gamma = torch.sigmoid(self.gating_params[layer_key])
                process_group(layer_key, alpha, gamma, d_alloc)
                cumulative_idx = end_idx

            # Process misc
            if self.has_misc:
                layer_key = 'misc'
                d_alloc = self.d_alloc_order[-1]  # misc is always last
                start_idx = cumulative_idx
                end_idx = start_idx + d_alloc
                alpha = self.subspace_params[start_idx:end_idx]
                gamma = torch.sigmoid(self.gating_params[layer_key])
                process_group(layer_key, alpha, gamma, d_alloc)
                    
        else:
            # Fixed Global Split
            # First d_A params for A, rest for B
            if 'A' in self.projectors:
                alpha_A = self.subspace_params[:self.d_A]
                flat_A = self.projectors['A'] @ alpha_A
                self._set_params(flat_A, self.names_A, self.params_A)
                
            if 'B' in self.projectors:
                alpha_B = self.subspace_params[self.d_A:] # Rest is for B
                flat_B = self.projectors['B'] @ alpha_B
                self._set_params(flat_B, self.names_B, self.params_B)

        return self._forward_net[0](*args, **kwargs)

    def _set_params(self, flat_params, names, init_params):
        unflattened = unflatten_like(flat_params, init_params)
        for p_name, init, proj_param in zip(names, init_params, unflattened):
            p = init + proj_param.view(*init.shape)
            _setchainattr(self._forward_net[0], p_name, p)

    # --- Gating annealing helpers ---
    def set_gating_scale(self, value: float):
        """Set the gating sigmoid scale (slope/temperature inverse).

        Args:
            value: positive float for sigmoid multiplier.
        """
        try:
            self.gating_scale = float(value)
        except Exception:
            raise ValueError("gating_scale must be a numeric value")

    def get_gating_scale(self) -> float:
        return float(self.gating_scale)

    def step_gating_anneal(self, step: int = None):
        """Advance or set the annealer and update `self.gating_scale`.

        If `step` is None, the internal counter `_anneal_step` is incremented by one.
        Otherwise `_anneal_step` is set to `step`.

        The `gating_anneal` config must provide `total_steps` and `end` at minimum.
        Supported modes: 'linear', 'cosine'.
        """
        cfg = self.gating_anneal
        if cfg is None:
            return self.gating_scale

        if step is None:
            self._anneal_step += 1
        else:
            self._anneal_step = int(step)

        start = cfg.get('start', None)
        end = cfg.get('end', None)
        total = cfg.get('total_steps', None)
        mode = cfg.get('mode', 'linear')

        if total is None or end is None:
            # Nothing to do if required keys missing
            return self.gating_scale

        # If start is not provided, use current gating_scale as start
        if start is None:
            start = float(self.gating_scale)

        # clamp step
        t = min(1.0, max(0.0, float(self._anneal_step) / float(total))) if total > 0 else 1.0

        if mode == 'linear':
            new_scale = float(start) + (float(end) - float(start)) * t
        elif mode == 'cosine':
            # smooth cosine schedule from start -> end
            cos_t = 0.5 * (1.0 - math.cos(math.pi * t))
            new_scale = float(start) + (float(end) - float(start)) * cos_t
        else:
            # default to linear if unknown
            new_scale = float(start) + (float(end) - float(start)) * t

        # ensure numeric sanity
        self.gating_scale = float(max(1e-6, min(new_scale, 1e6)))
        return self.gating_scale

    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        # Move init params
        for group in self.param_groups.values():
            for i, (n, p) in enumerate(group['items']):
                group['items'][i] = (n, p.to(*args, **kwargs))
        
        # Update stored lists
        self.params_A = [p for n, p in self.param_groups['A']['items']]
        self.params_B = [p for n, p in self.param_groups['B']['items']]
        
        # Update compatibility list
        self.trainable_initparams = [p.to(*args, **kwargs) for p in self.trainable_initparams]
        
        return super().to(*args, **kwargs)
    
    def get_num_params(self, only_trainable=False):
        if only_trainable:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return n_params
        else:
            n_params = sum(p.numel() for p in self.parameters())
            return n_params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self._forward_net[0].config.block_size else idx[:, -self._forward_net[0].config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, correct_bias, adam_epislon, no_decay_bias):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Separate gating params (will get a special LR multiplier)
        gating_param_names = [pn for pn in param_dict.keys() if pn.startswith('gating_params')]
        gating_params = [param_dict[pn] for pn in gating_param_names]

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        optim_groups = []
        if not no_decay_bias:
            # all params decayed except gating handled separately
            normal_params = [p for pn, p in param_dict.items() if not pn.startswith('gating_params')]
            optim_groups.append({'params': normal_params, 'weight_decay': weight_decay})
            if gating_params:
                optim_groups.append({'params': gating_params, 'weight_decay': weight_decay, 'lr': learning_rate * self.gating_lr_mult})
            print("using all params (with gating params grouped if present)")
        else:
            # split into decayed and non-decayed, then move gating params into dedicated groups
            decay_params = [p for pn, p in param_dict.items() if (p.dim() >= 2 and not pn.startswith('gating_params'))]
            nodecay_params = [p for pn, p in param_dict.items() if (p.dim() < 2 and not pn.startswith('gating_params'))]

            if decay_params:
                optim_groups.append({'params': decay_params, 'weight_decay': weight_decay})
            if nodecay_params:
                optim_groups.append({'params': nodecay_params, 'weight_decay': 0.0})

            # gating params: separate into decay vs nodecay depending on dim
            if gating_params:
                gating_decay = [p for p in gating_params if p.dim() >= 2]
                gating_nodecay = [p for p in gating_params if p.dim() < 2]
                if gating_decay:
                    optim_groups.append({'params': gating_decay, 'weight_decay': weight_decay, 'lr': learning_rate * self.gating_lr_mult})
                if gating_nodecay:
                    optim_groups.append({'params': gating_nodecay, 'weight_decay': 0.0, 'lr': learning_rate * self.gating_lr_mult})

            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=adam_epislon, **extra_args)

        print(f"using fused AdamW: {use_fused}")
        if gating_param_names:
            print(f"gating params grouped ({len(gating_param_names)} tensors), gating_lr_mult={self.gating_lr_mult}")

        return optimizer
