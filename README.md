# triton-but-documented

A documented version of Triton. See screenshots below for examples. Mostly changes heavily to document over 

`triton.language.make_tensor_descriptor(base, shape, strides, block_shape, padding_option='zero')`


# After

<img width="1054" height="1590" alt="CleanShot 2026-03-01 at 20 13 03@2x" src="https://github.com/user-attachments/assets/4a5ef207-36e2-4d12-8f28-27d8f5f03509" />
```python
import torch
import triton
import triton.language as tl

@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    # Create the tensor descriptor
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )

    # Calculate offsets based on program ID
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK

    # Load block using descriptor
    value = desc.load([moffset, noffset])

    # Store computed result back using descriptor
    desc.store([moffset, noffset], tl.abs(value))

# TMA descriptors require aligned global memory allocation
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

M, N = 256, 256
x = torch.randn(M, N, device="cuda")
M_BLOCK, N_BLOCK = 32, 32

# Grid configuration matches block tiling
grid = (M // M_BLOCK, N // N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```

# Before

<img width="1612" height="1136" alt="CleanShot 2026-03-01 at 20 13 27@2x" src="https://github.com/user-attachments/assets/732971b1-68d4-46ad-bffd-e905b326b372" />

```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )

    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK

    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))

# TMA descriptors require a global memory allocation
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

M, N = 256, 256
x = torch.randn(M, N, device="cuda")
M_BLOCK, N_BLOCK = 32, 32
grid = (M / M_BLOCK, N / N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```
