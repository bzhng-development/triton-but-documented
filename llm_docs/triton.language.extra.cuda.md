# triton.language.extra.cuda

CUDA-specific intrinsics — SM clock, thread/warp queries, and custom FP8 conversions.

*8 APIs documented.*

---

## triton.language.extra.cuda

### triton.language.extra.cuda.convert_custom_float8_sm70

```python
convert_custom_float8_sm70(arg, dst_ty, fp_downcast_rounding=None, _semantic=None)
```

**`triton.language.extra.cuda.convert_custom_float8_sm70(arg, dst_ty, fp_downcast_rounding=None)`**

    Convert a tensor to custom float8 format for SM70 (Compute Capability 7.0) GPUs.

    Parameters
    ----------
    arg : tensor
        The input tensor to convert.
    dst_ty : tl.dtype
        The target float8 data type. Supported types include `fp8e4nv`, `fp8e4b8`, 
        `fp8e4b15`, `fp8e5`, and `fp8e5b16`.
    fp_downcast_rounding : str, optional
        The rounding mode for downcasting floating-point values. Supported values are 
        `"rtne"` (round to nearest, ties to even) and `"rtz"` (round towards zero). 
        If not specified, defaults to `"rtne"`.

    Returns
    -------
    tensor
        A tensor with the same shape as `arg` but converted to the specified float8 
        destination type.

    Notes
    -----
    This function is specifically designed for NVIDIA GPUs with Compute Capability 7.0 
    (SM70). It disables the `has_minx2` optimization that is available on newer 
    architectures (SM80+). For SM80 and later, consider using 
    :py`triton.language.extra.cuda.convert_custom_float8_sm80()` instead.

    The conversion follows IEEE 754 rounding rules when `fp_downcast_rounding` is 
    specified. Behavior is undefined if the destination type cannot represent the 
    input values (e.g., overflow to infinity).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         input = tl.load(input_ptr + offsets)
         # Convert fp32 to fp8e4nv for SM70 GPUs
         output = tl.extra.cuda.convert_custom_float8_sm70(input, tl.float8e4nv)
         tl.store(output_ptr + offsets, output)
```

---

### triton.language.extra.cuda.convert_custom_float8_sm80

```python
convert_custom_float8_sm80(arg, dst_ty, fp_downcast_rounding=None, _semantic=None)
```

## triton.language.extra.cuda.convert_custom_float8_sm80


**`convert_custom_float8_sm80(arg, dst_ty, fp_downcast_rounding=None, _semantic=None)`**

    Convert a tensor to custom FP8 format for SM80+ GPUs.

    This function converts floating-point values to custom 8-bit floating-point
    formats supported on NVIDIA Ampere (SM80) and later architectures. It uses
    minx2 scaling for improved numerical range.

    Parameters
    ----------
    arg : tl.tensor
        Input tensor to convert. Must be a floating-point type (fp16, bf16, fp32).
    dst_ty : tl.dtype
        Destination FP8 type. Supported types: `fp8e4nv`, `fp8e4b8`, 
        `fp8e4b15`, `fp8e5`, `fp8e5b16`.
    fp_downcast_rounding : str, optional
        Rounding mode for downcasting. Supported values: `"rtne"` 
        (round to nearest, ties to even), `"rtz"` (round towards zero).
        Default is None (uses hardware default).
    _semantic : internal, optional
        Internal semantic parameter. Do not set manually.

    Returns
    -------
    tl.tensor
        Tensor converted to the specified FP8 format.

    Notes
    -----
    This function is specific to NVIDIA GPUs with compute capability 8.0 or 
    higher (Ampere, Ada, Hopper, Blackwell). The `has_minx2=True` parameter
    enables minx2 scaling which provides better numerical range for certain
    workloads.

    For optimal results, ensure the input values are within the representable
    range of the target FP8 format to minimize overflow/underflow.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import convert_custom_float8_sm80

     @triton.jit
     def fp8_convert_kernel(x_ptr, y_ptr, M: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * M
         x = tl.load(x_ptr + offset + tl.arange(0, M))
         # Convert fp32 to fp8e4nv with round-to-nearest-even
         y = convert_custom_float8_sm80(x, tl.float8e4nv, fp_downcast_rounding="rtne")
         tl.store(y_ptr + offset + tl.arange(0, M), y)
```

---

### triton.language.extra.cuda.gdc_launch_dependents

```python
gdc_launch_dependents(_semantic=None)
```

## triton.language.extra.cuda.gdc_launch_dependents


**`gdc_launch_dependents(_semantic=None)`**

   Signal that dependent kernels may launch once all program instances complete.

   This PTX instruction provides a synchronization point for programmatic
   dependent launch, allowing the runtime to schedule subsequent kernels
   once all instances of the current kernel have reached this point or
   terminated.

### Parameters
_semantic
    Internal Triton semantic object. Do not pass this argument directly;
    it is automatically provided when called within a `@triton.jit`
    decorated kernel.

### Returns
None
    This function emits a PTX instruction and does not return a value.

### Notes
This function emits the `griddepcontrol.launch_dependents` PTX
instruction, which is part of CUDA's parallel synchronization and
communication instructions.

Key behaviors:

* Only the first call has effect; repeated calls are no-ops.
* Safe to execute even when programmatic dependent launch is disabled.
* Should be treated as a hint to the runtime system for kernel scheduling.
* All program instances must call this function or complete for dependent
  kernels to launch.

This is a CUDA-specific operation and will not work on other GPU
backends.

See the `NVIDIA PTX documentation`_ for more details on grid dependency
control instructions.

.. _NVIDIA PTX documentation: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol

### Examples
```python
import triton
import triton.language as tl

@triton.jit
def producer_kernel(...):
    # Kernel computation
    ...
    # Signal that dependent kernels may launch
    tl.extra.cuda.gdc_launch_dependents()

@triton.jit
def consumer_kernel(...):
    # Depends on producer_kernel completion
    ...

# Launch with programmatic dependent launch enabled
producer_kernel[grid](...)
consumer_kernel[grid](...)  # Will launch after producer signals
```

---

### triton.language.extra.cuda.gdc_wait

```python
gdc_wait(_semantic=None)
```

## triton.language.extra.cuda.gdc_wait


**`gdc_wait()`**

   Insert a grid dependency control wait barrier.

   This blocking instruction waits for all instructions in a prior kernel to
   complete before continuing execution. Ensures memory operations before the
   wait are visible to subsequent instructions.

   Parameters
   ----------
   None

   Returns
   -------
   None

   Notes
   -----
   This instruction implements CUDA's `griddepcontrol.wait` PTX instruction.
   It provides cross-kernel synchronization by ensuring all memory writes from
   prior kernels are visible before proceeding.

   The instruction is safe to execute even when programmatic dependent launch
   is disabled. Use this when you need explicit synchronization between
   independent kernel launches that access shared memory locations.

   For example, if a prior kernel writes to address `x`, the new values will
   be visible in this kernel after calling `gdc_wait()`.

   References
   ----------
   NVIDIA PTX Documentation:
   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel_with_sync(...):
       # Wait for prior kernel to complete
       tl.extra.cuda.gdc_wait()
       # Now safe to read memory written by prior kernel
       value = tl.load(ptr)
       ...
```

---

### triton.language.extra.cuda.globaltimer

```python
globaltimer(_semantic=None)
```

### globaltimer


**`globaltimer()`**

    Returns the CUDA global timer value.

    Reads the hardware global timer on CUDA GPUs. This is a cycle counter
    that can be used for performance measurement and profiling.

    Parameters
    ----------
    _semantic : None, optional
        Internal Triton parameter. Do not pass directly.

    Returns
    -------
    timer : tl.tensor
        A tensor of type `tl.int64` containing the global timer value.

    Notes
    -----
    This function is CUDA-specific and uses inline PTX assembly to read
    the `%globaltimer` special register. The operation is not pure
    (has side effects) as it reads hardware state.

    The timer value represents GPU clock cycles and can be used for
    measuring kernel execution time or profiling code sections.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(timer_ptr, BLOCK: tl.constexpr):
         timer = tl.extra.cuda.globaltimer()
         tl.store(timer_ptr, timer)

     # Launch kernel
     timer = torch.empty(1, dtype=torch.int64, device="cuda")
     kernel[(1,)](timer, BLOCK=1024)
```

---

### triton.language.extra.cuda.num_threads

```python
num_threads(_semantic=None)
```

## triton.language.extra.cuda.num_threads


**`num_threads()`**

   Returns the total number of threads in a thread block.

   This is computed as `num_warps * 32`, where each warp contains 32
   threads on CUDA architectures.

> **Note:**
      This function can only be called inside a `@triton.jit` decorated
      kernel function. The `_semantic` parameter is handled automatically
      by the Triton compiler.

   Returns
   -------
   constexpr
      A compile-time constant representing the total thread count.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(
       ptr,
       BLOCK_SIZE: tl.constexpr,
   ):
       # Get total number of threads in the block
       n_threads = tl.num_threads()
       
       # Thread ID within the block
       thread_id = tl.arange(0, BLOCK_SIZE)
       
       # Each thread processes one element
       mask = thread_id < n_threads
       tl.store(ptr + thread_id, thread_id, mask=mask)
```

---

### triton.language.extra.cuda.num_warps

```python
num_warps(_semantic=None)
```

## triton.language.extra.cuda.num_warps

**`num_warps()`**

   Returns the number of warps per block for the current kernel.

   Returns
   -------
   constexpr
      The number of warps configured for the kernel at compile time.

   Notes
   -----
   This function returns a compile-time constant representing the number of
   CUDA warps (groups of 32 threads) that will execute each program instance.
   The value is determined by the kernel launch configuration and cannot be
   changed at runtime.

   Common values are powers of 2 (1, 2, 4, 8, 16, 32). The optimal value
   depends on the kernel's resource requirements (registers, shared memory)
   and the target GPU's multiprocessor capacity.

   This is a CUDA-specific function. For other backends, use the appropriate
   backend-specific API.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(...):
       num_warps = tl.extra.cuda.num_warps()
       # num_warps is a constexpr, can be used in compile-time logic
       if num_warps == 4:
           # Specialized code for 4 warps per block
           ...
```

---

### triton.language.extra.cuda.smid

```python
smid(_semantic=None)
```

## smid


**`smid()`**

   Returns the Streaming Multiprocessor (SM) identifier for the current GPU thread block.

   Parameters
   ----------
   None

   Returns
   -------
   smid : tl.tensor
      A scalar tensor containing the SM ID as a 32-bit integer (`int32`).

   Notes
   -----
   This function uses PTX inline assembly to read the `%smid` special register.
   The SM ID identifies which Streaming Multiprocessor is executing the current
   thread block. This can be useful for debugging or for SM-aware optimizations.

   The function is pure and has no side effects. It must be called within a
   :py`@triton.jit <triton.jit>()` decorated kernel function.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(ptr, BLOCK_SIZE: tl.constexpr):
       sm_id = tl.extra.cuda.smid()
       # Use sm_id for debugging or optimization
       tl.store(ptr, sm_id)

   # Launch kernel
   block_size = 128
   grid = (1,)
   kernel[grid](output_ptr, block_size)
```

---
