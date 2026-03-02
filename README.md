# triton-but-documented

A documented version of [Triton](https://github.com/triton-lang/triton). All 173 public APIs now have NumPy-style documentation generated via LLM (Qwen3.5-397B), using runtime introspection of the actual Triton source code as context. Based on Triton `3.6.0+gitc72bdad9`.

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

---

## What changed

### 103 existing APIs — rewritten documentation

The existing docs were thin autodoc stubs pulled from docstrings. Every one has been rewritten with proper NumPy-style docs: summary, parameters, returns, notes, and examples.

### 70 new APIs — previously undocumented

These were exported in `__all__` but had zero documentation in the Sphinx site. Now they do.

---

## Rewritten APIs (103)

### `triton` (4)

| API | Kind |
|-----|------|
| `Config` | class |
| `autotune` | function |
| `heuristics` | function |
| `jit` | function |

### `triton.language` (94)

| API | Kind |
|-----|------|
| `abs` | function |
| `advance` | function |
| `arange` | function |
| `argmax` | jit_function |
| `argmin` | jit_function |
| `associative_scan` | function |
| `assume` | function |
| `atomic_add` | function |
| `atomic_and` | function |
| `atomic_cas` | function |
| `atomic_max` | function |
| `atomic_min` | function |
| `atomic_or` | function |
| `atomic_xchg` | function |
| `atomic_xor` | function |
| `broadcast` | function |
| `broadcast_to` | function |
| `cast` | function |
| `cat` | function |
| `cdiv` | jit_function |
| `ceil` | function |
| `clamp` | function |
| `cos` | function |
| `cumprod` | jit_function |
| `cumsum` | jit_function |
| `debug_barrier` | function |
| `device_assert` | function |
| `device_print` | function |
| `div_rn` | function |
| `dot` | function |
| `dot_scaled` | function |
| `erf` | function |
| `exp` | function |
| `exp2` | function |
| `expand_dims` | function |
| `fdiv` | function |
| `flip` | jit_function |
| `floor` | function |
| `fma` | function |
| `full` | function |
| `gather` | function |
| `histogram` | function |
| `inline_asm_elementwise` | function |
| `interleave` | jit_function |
| `join` | function |
| `load` | function |
| `load_tensor_descriptor` | function |
| `log` | function |
| `log2` | function |
| `make_block_ptr` | function |
| `make_tensor_descriptor` | function |
| `max` | jit_function |
| `max_constancy` | function |
| `max_contiguous` | function |
| `maximum` | function |
| `min` | jit_function |
| `minimum` | function |
| `multiple_of` | function |
| `num_programs` | function |
| `permute` | function |
| `program_id` | function |
| `rand` | jit_function |
| `randint` | jit_function |
| `randint4x` | jit_function |
| `randn` | jit_function |
| `range` | class |
| `ravel` | jit_function |
| `reduce` | function |
| `reshape` | function |
| `rsqrt` | function |
| `sigmoid` | jit_function |
| `sin` | function |
| `softmax` | jit_function |
| `sort` | jit_function |
| `split` | function |
| `sqrt` | function |
| `sqrt_rn` | function |
| `static_assert` | function |
| `static_print` | function |
| `static_range` | class |
| `store` | function |
| `store_tensor_descriptor` | function |
| `sum` | jit_function |
| `swizzle2d` | jit_function |
| `tensor` | class |
| `tensor_descriptor` | class |
| `topk` | jit_function |
| `trans` | function |
| `umulhi` | function |
| `view` | function |
| `where` | function |
| `xor_sum` | jit_function |
| `zeros` | jit_function |
| `zeros_like` | jit_function |

### `triton.testing` (5)

| API | Kind |
|-----|------|
| `Benchmark` | class |
| `assert_close` | function |
| `do_bench` | function |
| `do_bench_cudagraph` | function |
| `perf_report` | function |

---

## Newly Documented APIs (70)

### `triton` (17)

| API | Kind | Description |
|-----|------|-------------|
| `AsyncCompileMode` | class | Context manager for asynchronous kernel compilation |
| `CompilationError` | class | Raised when kernel compilation fails |
| `FutureKernel` | class | Handle for an asynchronously compiled kernel |
| `InterpreterError` | class | Raised during interpreter-mode execution errors |
| `JITFunction` | class | Core class wrapping `@triton.jit` decorated functions |
| `KernelInterface` | class | Abstract base class for kernel interfaces |
| `MockTensor` | class | Lightweight tensor mock for testing |
| `OutOfResources` | class | Raised when a kernel exceeds GPU resource limits |
| `TensorWrapper` | class | Wraps tensors with custom dtype for specialization |
| `TritonError` | class | Base exception class for Triton errors |
| `cdiv` | constexpr_function | Ceiling division |
| `compile` | function | Compile a Triton kernel to GPU binary |
| `constexpr_function` | function | Decorator for compile-time evaluated functions |
| `must_use_result` | function | Decorator that warns if return value is unused |
| `next_power_of_2` | constexpr_function | Smallest power of 2 >= n |
| `reinterpret` | function | Reinterpret a tensor's dtype without changing bits |
| `set_allocator` | function | Set custom memory allocator for Triton |

### `triton.language` (46)

| API | Kind | Description |
|-----|------|-------------|
| `PropagateNan` | enum | Controls NaN propagation in reductions (`NONE`, `ALL`) |
| `TRITON_MAX_TENSOR_NUMEL` | constant | Maximum number of elements in a tensor (1048576) |
| `add` | function | Element-wise addition |
| `mul` | function | Element-wise multiplication |
| `sub` | function | Element-wise subtraction |
| `map_elementwise` | function | Apply a function element-wise across tensors |
| `to_tensor` | function | Convert a value to a tensor |
| `squeeze` | jit_function | Remove length-1 dimensions |
| `unsqueeze` | jit_function | Add a length-1 dimension |
| `reduce_or` | jit_function | Bitwise OR reduction |
| `bitonic_merge` | jit_function | Bitonic merge for sorting networks |
| `rand4x` | jit_function | Generate 4 uniform random floats |
| `randn4x` | jit_function | Generate 4 normal random floats |
| `pair_uniform_to_normal` | jit_function | Box-Muller transform |
| `philox` | jit_function | Philox PRNG counter-based RNG |
| `philox_impl` | jit_function | Low-level Philox implementation |
| `uint_to_uniform_float` | jit_function | Convert uint to uniform float in [0,1) |
| `dtype` | class | Triton data type descriptor |
| `constexpr` | class | Compile-time constant wrapper |
| `constexpr_type` | class | Type for constexpr values |
| `const` | class | Constant type marker |
| `block_type` | class | Block (tensor) type descriptor |
| `pointer_type` | class | Pointer type descriptor |
| `condition` | class | Conditional execution context |
| `slice` | class | Slice type for tensor indexing |
| `tuple` | class | Triton tuple type |
| `pi32_t` | pointer_type_instance | Pointer to int32 type alias |
| `void` | dtype_instance | Void type |
| `int1` | dtype_instance | 1-bit integer (boolean) |
| `int8` | dtype_instance | 8-bit signed integer |
| `int16` | dtype_instance | 16-bit signed integer |
| `int32` | dtype_instance | 32-bit signed integer |
| `int64` | dtype_instance | 64-bit signed integer |
| `uint8` | dtype_instance | 8-bit unsigned integer |
| `uint16` | dtype_instance | 16-bit unsigned integer |
| `uint32` | dtype_instance | 32-bit unsigned integer |
| `uint64` | dtype_instance | 64-bit unsigned integer |
| `float16` | dtype_instance | 16-bit IEEE 754 float |
| `bfloat16` | dtype_instance | 16-bit brain floating point |
| `float32` | dtype_instance | 32-bit IEEE 754 float |
| `float64` | dtype_instance | 64-bit IEEE 754 float |
| `float8e4b15` | dtype_instance | 8-bit float (4-bit exponent, bias 15) |
| `float8e4nv` | dtype_instance | 8-bit float (NVIDIA E4M3) |
| `float8e4b8` | dtype_instance | 8-bit float (4-bit exponent, bias 8) |
| `float8e5` | dtype_instance | 8-bit float (5-bit exponent) |
| `float8e5b16` | dtype_instance | 8-bit float (5-bit exponent, bias 16) |

### `triton.testing` (7)

| API | Kind | Description |
|-----|------|-------------|
| `Mark` | class | Benchmark parameter marker |
| `cuda_memcheck` | function | Run kernel under cuda-memcheck |
| `get_dram_gbps` | function | Get peak DRAM bandwidth in GB/s |
| `get_max_simd_tflops` | function | Get peak SIMD TFLOPS |
| `get_max_tensorcore_tflops` | function | Get peak Tensor Core TFLOPS |
| `nvsmi` | function | Query nvidia-smi properties |
| `set_gpu_clock` | function | Set GPU clock frequency for benchmarking |

---

## How it was built

1. **Runtime introspection** — `inspect.getsource()`, `inspect.signature()`, `inspect.getdoc()` on the live Triton objects (including unwrapping `JITFunction.fn` and `ConstexprFunction.fn`)
2. **LLM generation** — Each API's source code + full source file sent to Qwen3.5-397B (262k context) via SGLang, generating NumPy-style RST documentation
3. **Sphinx build** — Generated RST placed into `docs/python-api/generated/`, index files updated, built with `make html`
