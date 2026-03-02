# triton.runtime

Triton runtime — autotuner, driver interface, caching, and kernel configuration.

*15 APIs documented.*

---

## triton.runtime

### triton.runtime.Autotuner

```python
Autotuner(fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None, post_hook=None, prune_configs_by: 'Optional[Dict]' = None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False)
```

## triton.runtime.Autotuner

**`triton.runtime.Autotuner(fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None, post_hook=None, prune_configs_by=None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False)`**

   Auto-tuner wrapper for Triton JIT kernels that benchmarks multiple configurations
   to find the optimal one at runtime.

   The autotuner evaluates different kernel configurations (e.g., block sizes,
   num_warps, num_stages) and caches the best performing configuration based on
   tuning keys. Subsequent calls with the same key values reuse the cached result.

   Parameters
   ----------
   fn : triton.JITFunction
       The JIT-compiled kernel function to auto-tune.
   arg_names : list of str
       List of argument names for the kernel function.
   configs : list of triton.Config
       List of configurations to benchmark. Each config specifies meta-parameters
       like BLOCK_SIZE, num_warps, num_stages, etc.
   key : list of str
       Argument names that form the tuning key. When these argument values change,
       all configs are re-evaluated.
   reset_to_zero : list of str, optional
       Argument names whose corresponding tensors will be zeroed before running
       each configuration. Prevents side effects from multiple kernel executions.
   restore_value : list of str, optional
       Argument names whose corresponding tensors will be restored to their
       original values after benchmarking completes.
   pre_hook : callable, optional
       Function called before kernel execution. Signature:
       `pre_hook(kwargs, reset_only=False)`. Overrides default reset/restore hooks.
   post_hook : callable, optional
       Function called after kernel execution. Signature:
       `post_hook(kwargs, exception)`. Overrides default restore hooks.
   prune_configs_by : dict, optional
       Dictionary for config pruning with optional fields:
       
       * 'perf_model': callable that predicts runtime for configs
       * 'top_k': number of configs to benchmark (int or float <= 1.0)
       * 'early_config_prune': callable to filter configs before benchmarking
       
       The early_config_prune function signature:
       `early_config_prune(configs, named_args, **kwargs) -> list of Config`
   warmup : int, optional
       Warmup time in milliseconds for benchmarking (deprecated).
   rep : int, optional
       Repetition count for benchmarking (deprecated).
   use_cuda_graph : bool, default False
       Whether to use CUDA graphs for benchmarking (deprecated).
   do_bench : callable, optional
       Custom benchmark function. Signature: `do_bench(kernel_call, quantiles)`.
       Defaults to Triton's internal benchmarker.
   cache_results : bool, default False
       Whether to persist autotuning results to disk cache.

   Attributes
   ----------
   configs : list of triton.Config
       List of configurations to evaluate.
   keys : tuple of str
       Tuning key argument names.
   cache : dict
       Runtime cache mapping tuning keys to best configurations.
   arg_names : list of str
       Kernel argument names.
   best_config : triton.Config
       The best configuration found during autotuning.

   Methods
   -------
   run(*args, **kwargs)
       Execute the kernel with the best configuration for the given arguments.
   prune_configs(kwargs)
       Filter configurations using perf_model and early_config_prune.
   warmup(*args, **kwargs)
       Warmup the kernel for all pruned configurations.

   Notes
   -----
   When multiple configurations are provided, the autotuner benchmarks each one
   and caches the fastest. The tuning key determines when to re-benchmark:
   if key argument values change, all configs are evaluated again.

   If `reset_to_zero` or `restore_value` are specified, default pre/post hooks
   are installed to manage tensor state. Custom hooks override this behavior.

   Disk caching (when `cache_results=True`) persists tuning results across
   runs. Cache invalidation considers Triton version, backend hash, kernel IR,
   environment variables, and tuning key.

   The environment variable `TRITON_PRINT_AUTOTUNING=1` enables verbose output
   showing autotuning time and selected configuration.

   Deprecated parameters (warmup, rep, use_cuda_graph) trigger warnings and use
   legacy benchmarking paths. Prefer `do_bench` for custom benchmarking.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.autotune(
       configs=[
           triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
           triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=8),
           triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=8),
       ],
       key=['x_size'],
       reset_to_zero=['output_ptr'],
   )
   @triton.jit
   def kernel(x_ptr, output_ptr, x_size, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offs < x_size
       x = tl.load(x_ptr + offs, mask=mask)
       tl.store(output_ptr + offs, x, mask=mask)

   # First call triggers autotuning
   kernel[x_grid, y_grid](x_ptr, output_ptr, x_size=1024)

   # Subsequent calls with same x_size use cached best config
   kernel[x_grid, y_grid](x_ptr, output_ptr, x_size=1024)

   # Different x_size triggers re-benchmarking
   kernel[x_grid, y_grid](x_ptr, output_ptr, x_size=2048)

.. code-block:: python

   # Advanced usage with config pruning
   @triton.autotune(
       configs=[...],
       key=['M', 'N', 'K'],
       prune_configs_by={
           'perf_model': lambda M, N, K, BLOCK_SIZE: ...,
           'top_k': 3,  # Only benchmark top 3 configs
       },
       cache_results=True,  # Persist results to disk
   )
   @triton.jit
   def matmul_kernel(...):
       ...
```

---

### triton.runtime.Config

```python
Config(kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None)
```

## class Config


.. autoclass:: Config

   Represents a possible kernel configuration for the auto-tuner to try.

   Parameters
   ----------
   kwargs : dict
      Dictionary of meta-parameters to pass to the kernel as keyword arguments.
   num_warps : int, optional
      Number of warps to use for the kernel when compiled for GPUs. Default is 4.
      Each warp contains 32 threads, so `num_warps=8` yields 256 threads.
   num_stages : int, optional
      Number of stages for software-pipelining loops. Default is 3. Mostly useful
      for matrix multiplication workloads on SM80+ GPUs.
   num_ctas : int, optional
      Number of blocks in a block cluster. Default is 1. SM90+ only.
   maxnreg : int, optional
      Maximum number of registers one thread can use. Corresponds to PTX
      `.maxnreg` directive. Not supported on all platforms.
   pre_hook : callable, optional
      Function called before the kernel is executed. Receives kernel arguments
      as parameters.
   ir_override : str, optional
      Filename of a user-defined IR file (`.ttgir`, `.llir`, `.ptx`, or
      `.amdgcn`).

   Attributes
   ----------
   kwargs : dict
      Dictionary of meta-parameters passed to the kernel.
   num_warps : int
      Number of warps for kernel parallelization.
   num_stages : int
      Number of software-pipelining stages.
   num_ctas : int
      Number of blocks in a block cluster.
   maxnreg : int or None
      Maximum register count per thread.
   pre_hook : callable or None
      Pre-execution hook function.
   ir_override : str or None
      Path to user-defined IR file.

   Notes
   -----
   `Config` objects are typically used with `triton.autotune()` to specify
   different kernel configurations for performance tuning. The auto-tuner will
   benchmark each configuration and select the best one.

   When multiple configurations are tested, kernel side-effects may occur multiple
   times. Use `reset_to_zero` and `restore_value` parameters in
   `triton.autotune()` to manage tensor state.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.autotune(
       configs=[
           triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
           triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
           triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
       ],
       key=['x_size']
   )
   @triton.jit
   def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
       # kernel implementation
       pass

.. code-block:: python

   # Create a config with custom pre_hook
   def my_pre_hook(args):
       print(f"Running with args: {args}")

   config = triton.Config(
       kwargs={'BLOCK_SIZE': 512},
       num_warps=8,
       num_stages=2,
       pre_hook=my_pre_hook
   )
```

---

### triton.runtime.Heuristics

```python
Heuristics(fn, arg_names, values) -> 'None'
```

## class triton.runtime.Heuristics

Decorator wrapper that computes meta-parameter values dynamically before kernel execution.

### Parameters
fn : JITFunction
    The underlying JIT-compiled kernel function.
arg_names : list[str]
    List of argument names from the kernel function signature.
values : dict[str, Callable[[dict[str, Any]], Any]]
    Dictionary mapping meta-parameter names to functions that compute their values.
    Each function receives a dictionary of kernel arguments and returns the computed value.

### Attributes
fn : JITFunction
    The wrapped kernel function.
values : dict[str, Callable]
    The heuristic value computation functions.
arg_names : list[str]
    The kernel argument names.

### Notes
This class is typically instantiated via the `triton.heuristics()` decorator rather
than directly. It allows dynamic computation of meta-parameter values based on kernel
input arguments, providing an alternative to auto-tuning when performance tuning is
prohibitively expensive or not applicable.

The heuristic functions are called before kernel execution with a dictionary containing
all positional and keyword arguments. The returned values are injected as keyword
arguments to the kernel.

### Examples
```python
 # Compute BLOCK_SIZE as the smallest power-of-2 >= x_size
 @triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
 @triton.jit
 def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
     # BLOCK_SIZE is automatically computed based on x_size
     ...

 # Multiple heuristics can be combined
 @triton.heuristics(values={
     'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n']),
     'NUM_WARPS': lambda args: 4 if args['n'] < 1024 else 8
 })
 @triton.jit
 def kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr):
     ...
```

---

### triton.runtime.InterpreterError

```python
InterpreterError(error_message: Optional[str] = None)
```

**`InterpreterError(error_message: Optional[str] = None)`**

    Exception raised when an error occurs during Triton interpreter execution.

    Parameters
    ----------
    error_message : Optional[str], optional
        Optional error message string.

    Notes
    -----
    Subclass of `triton.errors.TritonError`. Raised specifically when the Triton interpreter encounters an execution error.

    Examples
    --------
```python
     from triton.runtime import InterpreterError

     try:
         raise InterpreterError("Interpreter execution failed")
     except InterpreterError as e:
         print(e)  # Outputs: Interpreter execution failed
```

---

### triton.runtime.JITFunction

```python
JITFunction(fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None, noinline=None, repr=None, launch_metadata=None)
```

## JITFunction

**`triton.runtime.JITFunction(fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None, noinline=None, repr=None, launch_metadata=None)`**

   Represent a JIT-compiled Triton kernel function.

   This class wraps a Python function decorated with `@triton.jit` and
   manages compilation, caching, and launch of GPU kernels. Instances are
   typically created via the `triton.jit()` decorator rather than
   directly instantiated.

   Parameters
   ----------
   fn : callable
      The Python function to be JIT-compiled. This function should contain
      Triton kernel code using `triton.language` operations.
   version : int, optional
      Version number for the kernel. Used for cache invalidation when
      kernel implementation changes.
   do_not_specialize : iterable of int or str, optional
      Parameter indices or names that should not be specialized during
      compilation. Specialization allows generating optimized code for
      specific argument values.
   do_not_specialize_on_alignment : iterable of int or str, optional
      Parameter indices or names that should not be specialized based on
      memory alignment.
   debug : bool, optional
      Enable debug mode for the kernel. When True, additional debugging
      information is preserved.
   noinline : bool, optional
      Prevent inlining of this function when called from other kernels.
   repr : callable, optional
      Custom function to generate string representation of the kernel.
   launch_metadata : callable, optional
      Function to generate metadata for kernel launches.

   Attributes
   ----------
   fn : callable
      The original Python function.
   module : str
      Module name where the function is defined.
   signature : inspect.Signature
      Function signature information.
   params : list of KernelParam
      Parameter metadata including specialization settings.
   device_caches : dict
      Per-device cache of compiled kernels.
   arg_names : list of str
      Names of all function arguments.
   constexprs : list of int
      Indices of constexpr parameters.

   Methods
   -------
   run(*args, grid, warmup, **kwargs)
      Execute the kernel with given arguments and grid configuration.
   warmup(*args, grid, **kwargs)
      Warmup the kernel by compiling without execution.
   __getitem__(grid)
      Return a callable that launches the kernel with specified grid.
   add_pre_run_hook(hook)
      Add a hook to be executed before kernel run.
   preload(specialization_data)
      Load pre-compiled kernel from serialized specialization data.

   Notes
   -----
   JITFunction objects are not meant to be called directly. Instead, use
   the grid syntax `kernel[grid](*args)` to launch the kernel on the GPU.
   The first call with a given argument signature triggers compilation;
   subsequent calls with the same signature use the cached compiled kernel.

   Arguments with `.data_ptr()` and `.dtype` attributes (e.g., PyTorch
   tensors) are implicitly converted to pointers. Global variables captured
   at compile time are checked at launch time to ensure consistency.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def add_kernel(x_ptr, y_ptr, out_ptr, n,
                  BLOCK_SIZE: tl.constexpr = 1024):
       pid = tl.program_id(axis=0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       out = x + y
       tl.store(out_ptr + offsets, out, mask=mask)

   # Launch kernel with grid configuration
   grid = (triton.cdiv(n, BLOCK_SIZE),)
   add_kernel[grid](x, y, out, n)

   # Access JITFunction properties
   print(add_kernel.arg_names)  # ['x_ptr', 'y_ptr', 'out_ptr', 'n', 'BLOCK_SIZE']
   print(add_kernel.constexprs)  # [4] (BLOCK_SIZE is constexpr)

See Also
--------
triton.jit : Decorator to create JITFunction instances
KernelInterface : Base class for kernel launch interface
JITCallable : Base class for JIT-compiled callables
```

---

### triton.runtime.KernelInterface

```python
KernelInterface()
```

## triton.runtime.KernelInterface

**`triton.runtime.KernelInterface()`**

   Abstract base class defining the interface for JIT-compiled Triton kernels.

   This class provides the core execution interface for kernels decorated with
   `@triton.jit`. It enables kernel launching with grid specification via
   the `__getitem__` syntax and supports warmup runs for performance
   benchmarking.

   Parameters
   ----------
   None

   Attributes
   ----------
   run : T
       Abstract method that executes the kernel. Must be implemented by
       subclasses.

   Methods
   -------
   warmup(*args, grid, **kwargs)
       Run the kernel in warmup mode to initialize internal state without
       executing the full kernel launch.

   run(*args, grid, warmup, **kwargs)
       Execute the kernel with the specified grid configuration.

   __getitem__(grid)
       Return a callable proxy that memorizes the grid configuration for
       kernel launch.

   Notes
   -----
   This is an abstract base class. Concrete implementations (e.g.,
   `triton.runtime.JITFunction`) must implement the `run` method.

   The typical usage pattern for launching a kernel is::

       kernel[grid](arg1, arg2, ...)

   where `grid` specifies the CUDA/synaptic grid dimensions. The
   `__getitem__` method returns a lambda that captures the grid and
   forwards arguments to `run`.

   Warmup runs skip actual kernel execution and are used for initialization
   and benchmarking purposes.

   Examples
   --------
   >>> import triton
   >>> import triton.language as tl
   >>> @triton.jit
   ... def my_kernel(x_ptr, y_ptr, n):
   ...     pid = tl.program_id(0)
   ...     if pid < n:
   ...         tl.store(y_ptr + pid, tl.load(x_ptr + pid) * 2.0)
   >>> grid = (1024,)
   >>> # Launch kernel with grid specification
   >>> my_kernel[grid](x, y, n)
   >>> # Warmup run for benchmarking
   >>> my_kernel.warmup(x, y, n, grid=grid)

---

### triton.runtime.MockTensor

```python
MockTensor(dtype, shape=None)
```

## class triton.runtime.MockTensor

Mock tensor object for kernel warmup without allocating real GPU memory.

This class provides a lightweight tensor-like interface that can be used in
place of actual tensors when calling `JITFunction.warmup()`. It mimics
the tensor API (dtype, shape, stride, data_ptr) without requiring actual
memory allocation.

### Parameters
dtype : dtype
    Data type of the mock tensor (e.g., `torch.float32`, `torch.int32`).
shape : list of int, optional
    Shape dimensions of the tensor. Default is `[1]`.

### Attributes
dtype : dtype
    The data type of the mock tensor.
shape : list of int
    The shape dimensions of the mock tensor.

### Methods
__init__(dtype, shape=None)
    Initialize the mock tensor.
stride()
    Return stride tuple for the tensor shape.
data_ptr()
    Return mock data pointer (always returns 0).
ptr_range()
    Return mock pointer range (always returns 0).

### Notes
- `data_ptr()` returns 0, optimistically assuming 16-byte alignment.
- `ptr_range()` returns 0, optimistically assuming 32-bit pointer range.
- This class is primarily used for kernel compilation warmup and testing.
- The static method `wrap_dtype()` automatically converts torch dtype
  objects to MockTensor instances.

### Examples
```python
 import torch
 import triton
 import triton.runtime as tr

 # Create a mock tensor for warmup
 mock_tensor = tr.MockTensor(torch.float32, shape=[1024, 512])

 # Use in kernel warmup
 @triton.jit
 def kernel(x_ptr, n):
     # kernel implementation
     pass

 kernel.warmup(tr.MockTensor(torch.float32), 1024, grid=(1,))

 # Automatically wrap torch dtype
 wrapped = tr.MockTensor.wrap_dtype(torch.float32)
 # Returns MockTensor(torch.float32)
```

---

### triton.runtime.OutOfResources

```python
OutOfResources(required, limit, name)
```

**`triton.runtime.OutOfResources(required, limit, name)`**

   Exception raised when a kernel requires more hardware resources than available.

   Parameters
   ----------
   required : int
       The amount of resource required by the kernel.
   limit : int
       The hardware limit available for the resource.
   name : str
       The name of the resource (e.g., `"shared memory"`, `"registers"`).

   Notes
   -----
   Inherits from `triton.errors.TritonError`. This exception indicates
   that the kernel configuration exceeds hardware constraints. Reducing block
   sizes or `num_stages` may resolve the issue.

   Examples
   --------
   Catching the exception during kernel compilation or launch:

```python
    try:
        kernel[grid](args)
    except triton.runtime.OutOfResources as e:
        print(e)
        # Adjust block sizes or num_stages
```

---

### triton.runtime.RedisRemoteCacheBackend

```python
RedisRemoteCacheBackend(key)
```

**`triton.runtime.RedisRemoteCacheBackend(key)`**

   Redis-backed remote cache implementation.

   Implements the `~triton.runtime.RemoteCacheBackend` interface using
   a Redis server. Configuration is loaded from `triton.knobs.cache.redis`.

   Parameters
   ----------
   key : str
       Unique identifier for the cache namespace. Used to prefix Redis keys.

   Notes
   -----
   Requires the `redis` Python package to be installed.

   Connection parameters are read from the following `triton.knobs` settings:

   - `knobs.cache.redis.host`
   - `knobs.cache.redis.port`
   - `knobs.cache.redis.key_format`

   Examples
   --------
```python
    from triton.runtime import RedisRemoteCacheBackend

    # Initialize the backend with a namespace key
    cache = RedisRemoteCacheBackend(key="my_cache_key")

    # Store data
    cache.put("kernel.so", b"binary_data")

    # Retrieve data
    data = cache.get(["kernel.so"])
```

---

### triton.runtime.RemoteCacheBackend

```python
RemoteCacheBackend(key: str)
```

**`triton.runtime.RemoteCacheBackend(key)`**

   A backend implementation for accessing a remote/distributed cache.

   This abstract base class defines the interface for remote cache backends.
   Concrete implementations (e.g., Redis) provide distributed caching
   capabilities for Triton compiled kernels across multiple nodes or sessions.

   Parameters
   ----------
   key : str
       Unique identifier for the cache namespace. Used to isolate cache
       entries between different compilation contexts or users.

   Notes
   -----
   This is an abstract base class. Concrete implementations must override
   the `get()` and `put()` methods. The class is typically used
   internally by `RemoteCacheManager` to handle remote cache
   operations.

   Remote cache backends enable distributed caching scenarios where:

   - Multiple machines share compiled kernel artifacts
   - Cache persistence survives local disk cleanup
   - LRU accounting works correctly across distributed systems

   Examples
   --------
   Creating a Redis-based remote cache backend:

```python
   from triton.runtime.cache import RedisRemoteCacheBackend

   # Initialize with a unique cache key
   backend = RedisRemoteCacheBackend(key="triton_cache_v1")

   # Store compiled kernel data
   backend.put("kernel_hash_abc", compiled_bytes)

   # Retrieve cached data
   results = backend.get(["kernel_hash_abc", "kernel_hash_xyz"])

Implementing a custom remote cache backend:

.. code-block:: python

   from triton.runtime.cache import RemoteCacheBackend
   from typing import Dict, List

   class MyCacheBackend(RemoteCacheBackend):
       def __init__(self, key: str):
           super().__init__(key)
           # Initialize custom backend (e.g., database connection)

       def get(self, filenames: List[str]) -> Dict[str, bytes]:
           # Fetch multiple files from remote storage
           pass

       def put(self, filename: str, data: bytes):
           # Store file to remote storage
           pass

See Also
--------
RemoteCacheManager : Manager class that uses RemoteCacheBackend
FileCacheManager : Local file-based cache implementation
RedisRemoteCacheBackend : Redis-based remote cache implementation
```

---

### triton.runtime.TensorWrapper

```python
TensorWrapper(base, dtype)
```

## class TensorWrapper

Wrapper class that reinterprets a tensor with a different dtype while preserving
the underlying memory and tensor properties.

### Parameters
base : tensor-like
    The underlying tensor object to wrap. Must have `data_ptr()`, `dtype`,
    `device`, `shape`, and `stride` methods or attributes.
dtype : dtype
    The new data type to interpret the tensor as.

### Attributes
dtype
    The reinterpret dtype.
base
    The underlying wrapped tensor.
data
    Reference to `base.data`.
device
    Reference to `base.device`.
shape
    Reference to `base.shape`.

### Methods
data_ptr()
    Returns the memory address of the tensor data.
stride(*args)
    Returns the stride for the specified dimension(s).
element_size()
    Returns the size in bytes of a single element.
cpu()
    Returns a TensorWrapper with the base tensor moved to CPU.
copy_(other)
    Copies data from another TensorWrapper into this one.
clone()
    Returns a new TensorWrapper with a cloned base tensor.
to(device)
    Returns a TensorWrapper with the base tensor moved to specified device.
new_empty(sizes)
    Returns a new TensorWrapper with an empty tensor of given sizes.

### Notes
TensorWrapper is primarily used internally by Triton for tensor reinterpretation
and mocking during kernel warmup. It allows viewing the same memory with a
different dtype without allocating new storage.

The wrapper delegates most operations to the underlying `base` tensor,
preserving the original tensor's memory layout while changing the dtype
interpretation.

### Examples
```python
 import torch
 import triton
 import triton.language as tl

 # Create a base tensor
 base = torch.randn(1024, dtype=torch.float32, device='cuda')

 # Wrap with different dtype
 wrapper = TensorWrapper(base, tl.int32)

 # Access tensor properties
 print(wrapper.shape)      # torch.Size([1024])
 print(wrapper.dtype)      # tl.int32
 print(wrapper.data_ptr()) # memory address

 # Move to different device
 cpu_wrapper = wrapper.cpu()

 # Create new empty tensor with same dtype
 empty = wrapper.new_empty((512,))
```

---

### triton.runtime.autotune

```python
autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False)
```

## triton.runtime.autotune


**`autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False)`**

   Decorator for auto-tuning a `triton.jit`'d function.

   Parameters
   ----------
   configs : list[triton.Config]
       A list of `triton.Config` objects representing different kernel configurations to benchmark.
   key : list[str]
       A list of argument names whose change in value will trigger the evaluation of all provided configs.
   prune_configs_by : dict, optional
       A dict of functions used to prune configs. Fields:
       
       * 'perf_model': performance model used to predicate running time with different configs, returns running time
       * 'top_k': number of configs to bench
       * 'early_config_prune': a function used to prune configs with signature
         `prune_configs_by(configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]`.
         Should return at least one config.
   reset_to_zero : list[str], optional
       A list of argument names whose value will be reset to zero before evaluating any configs.
   restore_value : list[str], optional
       A list of argument names whose value will be restored after evaluating any configs.
   pre_hook : callable, optional
       A function called before the kernel is called. Overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
       Signature: `lambda kwargs, reset_only`.
   post_hook : callable, optional
       A function called after the kernel is called. Overrides the default post_hook used for 'restore_value'.
       Signature: `lambda kwargs, exception`.
   warmup : int, optional
       Warmup time (in ms) to pass to benchmarking. Deprecated.
   rep : int, optional
       Repetition time (in ms) to pass to benchmarking. Deprecated.
   use_cuda_graph : bool, optional
       Whether to use CUDA graphs for benchmarking. Deprecated.
   do_bench : callable, optional
       A benchmark function to measure the time of each run. Signature: `lambda fn, quantiles`.
   cache_results : bool, optional
       Whether to cache autotune timings to disk. Defaults to False.

   Returns
   -------
   decorator : callable
       A decorator function that wraps the kernel function with auto-tuning logic.

   Notes
   -----
   When all configurations are evaluated, the kernel will run multiple times. This means that whatever value the kernel updates will be updated multiple times. To avoid this undesired behavior, use the `reset_to_zero` argument, which resets the value of the provided tensor to zero before running any configuration.

   If the environment variable `TRITON_PRINT_AUTOTUNING` is set to `"1"`, Triton will print a message to stdout after autotuning each kernel, including the time spent autotuning and the best configuration.

   The `warmup`, `rep`, and `use_cuda_graph` parameters are deprecated.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.autotune(
       configs=[
           triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
           triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
       ],
       key=['x_size']  # configs will be evaluated when x_size changes
   )
   @triton.jit
   def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
       # kernel implementation
       ...

.. code-block:: python

   # Using reset_to_zero to prevent side effects during autotuning
   @triton.autotune(
       configs=[
           triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=4),
           triton.Config(kwargs={'BLOCK_SIZE': 512}, num_warps=8),
       ],
       key=['n'],
       reset_to_zero=['output_ptr']  # reset output before each config test
   )
   @triton.jit
   def kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
       # kernel implementation
       ...
```

---

### triton.runtime.driver

**`triton.runtime.driver`**

    Configures and manages the active GPU driver backend.

    Provides abstraction for device management, memory operations, and kernel
    launching across supported hardware backends (e.g., CUDA, HIP).

    Notes
    -----
    The driver instance is initialized lazily upon first access. It handles
    context creation, stream management, and low-level kernel dispatch.
    Direct instantiation is discouraged; use the global runtime instance.

    Examples
    --------
    Query device information via the driver interface:

```python
     import triton.runtime

     driver = triton.runtime.driver
     device_count = driver.get_device_count()
     print(f"Detected devices: {device_count}")

 Access the default stream for kernel execution:

 .. code-block:: python

     stream = driver.get_stream(0)
```

---

### triton.runtime.heuristics

```python
heuristics(values)
```

Decorator for specifying how the values of certain meta-parameters may be computed.

This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

### Parameters
values : dict[str, Callable[[dict[str, Any]], Any]]
    A dictionary mapping meta-parameter names to functions that compute their values.
    Each function accepts a dictionary of kernel arguments and returns the computed
    value.

### Returns
decorator : Callable
    A decorator that wraps the JIT function, returning a `Heuristics` object.

### Notes
The heuristic functions are invoked at kernel launch time with a dictionary containing
all positional and keyword arguments passed to the kernel.

### Examples
```python
 # smallest power-of-two >= x_size
 @triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
 @triton.jit
 def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
     ...
```

---

### triton.runtime.reinterpret

```python
reinterpret(tensor, dtype)
```

**`triton.runtime.reinterpret(tensor, dtype)`**

    Reinterpret a tensor object as a specific data type.

    Creates a `TensorWrapper` around the input tensor with the specified
    `dtype`, or returns the base tensor if the dtype matches.

    Parameters
    ----------
    tensor : tensor-like or TensorWrapper
        Input tensor object. Must have a `data_ptr` method or be an
        existing `TensorWrapper`.
    dtype : dtype
        Target data type for the reinterpreted tensor.

    Returns
    -------
    TensorWrapper or tensor
        A `TensorWrapper` view of the tensor with the specified `dtype`.
        If `tensor` is a `TensorWrapper` and `dtype` matches the base
        tensor's dtype, the base tensor is returned directly.

    Raises
    ------
    TypeError
        If `tensor` does not have a `data_ptr` method and is not a
        `TensorWrapper`.

    Notes
    -----
    This function is primarily used internally for mocking tensor types
    during kernel warmup or testing scenarios. It does not change the
    underlying data pointer, only the perceived data type.

    Examples
    --------
```python
     import torch
     import triton.runtime

     tensor = torch.zeros((10,), dtype=torch.float32)
     wrapped = triton.runtime.reinterpret(tensor, torch.int32)
     print(wrapped.dtype)  # torch.int32
     print(wrapped.data_ptr() == tensor.data_ptr())  # True
```

---
