# triton

Core Triton API — JIT compilation, autotuning, configuration, and error types.

*21 APIs documented.*

---

## triton

### triton.AsyncCompileMode

```python
AsyncCompileMode(executor: 'Executor', *, ignore_errors=False)
```

**`triton.AsyncCompileMode(executor, *, ignore_errors=False)`**

   Context manager for managing asynchronous kernel compilation.

   Submits compilation tasks to a concurrent executor and ensures all
   outstanding futures are finalized upon exiting the context. Only one
   instance may be active within a given context scope.

   Parameters
   ----------
   executor : concurrent.futures.Executor
       The executor used to submit compilation tasks asynchronously.
   ignore_errors : bool, optional
       If True, exceptions raised during compilation are suppressed and
       `None` is returned instead of raising. Default is False.

   Methods
   -------
   submit(key, compile_fn, finalize_fn)
       Submit a compilation task to the executor.

   Raises
   ------
   RuntimeError
       If entering a context while another `AsyncCompileMode` is already
       active in the same scope.

   .. method:: submit(key, compile_fn, finalize_fn)

      Submit a kernel compilation task for asynchronous execution.

      Parameters
      ----------
      key : hashable
           Unique identifier for the compilation task. Used to deduplicate
           concurrent submissions for the same key.
      compile_fn : Callable
           Function that performs the compilation and returns the kernel.
      finalize_fn : Callable
           Function called with the compiled kernel result upon completion.

      Returns
      -------
      FutureKernel
           A wrapper object that resolves to the compiled kernel. Attribute
           access is deferred to the underlying kernel upon resolution.

   Examples
   --------
```python
   from concurrent.futures import ThreadPoolExecutor
   from triton.runtime import AsyncCompileMode

   with ThreadPoolExecutor() as executor:
       with AsyncCompileMode(executor) as mode:
           future_kernel = mode.submit(
               key="my_kernel",
               compile_fn=compile_my_kernel,
               finalize_fn=finalize_my_kernel
           )
           # Kernel is accessible after context exits
           kernel = future_kernel.result()
```

---

### triton.CompilationError

```python
CompilationError(src: Optional[str], node: ast.AST, error_message: Optional[str] = None)
```

**`triton.CompilationError(src, node, error_message=None)`**

   Base class for all errors raised during Triton compilation.

   Parameters
   ----------
   src : Optional[str]
       The source code string being compiled.
   node : ast.AST
       The Abstract Syntax Tree node where the error occurred.
   error_message : Optional[str], optional
       Specific error message detail (default None).

   Attributes
   ----------
   src : Optional[str]
       Stored source code.
   node : ast.AST
       Stored AST node.
   error_message : Optional[str]
       Stored specific error message.
   message : str
       The formatted error string including source excerpt.

   Notes
   -----
   The string representation includes source code snippets around the failing
   AST node (line and column info). The class is picklable via `__reduce__`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel_fn(pid):
       # Invalid operation triggering compilation error
       tl.static_assert(pid == 0)

   try:
       kernel_fn[grid](...)
   except triton.CompilationError as e:
       print(e.message)
```

---

### triton.Config

```python
Config(kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None)
```

## triton.Config

Represents a kernel configuration for the auto-tuner to try.

### Parameters
kwargs : dict
    Dictionary of meta-parameters to pass to the kernel as keyword arguments.
num_warps : int, optional
    Number of warps to use for the kernel when compiled for GPUs. Each warp
    contains 32 threads, so `num_warps=8` means 256 threads per kernel
    instance. Default is 4.
num_stages : int, optional
    Number of stages for software-pipelining loops. Mostly useful for matrix
    multiplication workloads on SM80+ GPUs. Default is 3.
num_ctas : int, optional
    Number of blocks in a block cluster. SM90+ only. Default is 1.
maxnreg : int, optional
    Maximum number of registers one thread can use. Corresponds to PTX
    `.maxnreg` directive. Not supported on all platforms. Default is None.
pre_hook : callable, optional
    Function called before the kernel is called. Receives kernel arguments
    as parameters. Default is None.
ir_override : str, optional
    Filename of a user-defined IR file (.{ttgir|llir|ptx|amdgcn}). Default
    is None.

### Attributes
kwargs : dict
    Dictionary of meta-parameters.
num_warps : int
    Number of warps.
num_stages : int
    Number of pipeline stages.
num_ctas : int
    Number of block clusters.
maxnreg : int or None
    Maximum register count per thread.
pre_hook : callable or None
    Pre-execution hook function.
ir_override : str or None
    IR override filename.

### Notes
This class is primarily used with `triton.autotune()` to specify
different configurations to benchmark during auto-tuning. Multiple
:py`Config` objects are passed to the `configs` parameter of
`autotune()`, and the auto-tuner will benchmark each configuration
to find the best one for the given input.

### Examples
```python
 import triton
 import triton.language as tl

 # Basic configuration with custom block size
 config = triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4)

 # Configuration optimized for matrix multiplication on SM80+
 config = triton.Config(
     kwargs={'BLOCK_SIZE': 256},
     num_warps=8,
     num_stages=2
 )

 # Use with autotune decorator
 @triton.autotune(
     configs=[
         triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
         triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=8),
     ],
     key=['x_size']
 )
 @triton.jit
 def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
     # kernel implementation
     pass
```

---

### triton.FutureKernel

```python
FutureKernel(finalize_compile: 'Callable', future: 'Future')
```

**`triton.FutureKernel(finalize_compile, future)`**

   Wrapper for asynchronous kernel compilation tasks.

   Parameters
   ----------
   finalize_compile : callable
       Callback function invoked after compilation completes.
   future : concurrent.futures.Future
       Future object representing the pending compilation task.

   .. method:: result(ignore_errors=False)

      Blocks until compilation finishes, finalizes the kernel, and returns
      the compiled object.

      Parameters
      ----------
      ignore_errors : bool, optional
          If True, suppress exceptions during compilation (default False).

      Returns
      -------
      kernel : object
          The compiled kernel object.

   Notes
   -----
   Implements `__getattr__` to transparently proxy attribute access to the
   underlying compiled kernel. The kernel result is cached after the first
   call to `result()`.

   Examples
   --------
   Typically used internally by `triton.runtime.AsyncCompileMode`:

```python
   with AsyncCompileMode(executor) as mode:
       future_kernel = mode.submit(key, compile_fn, finalize_fn)
       # Access kernel methods directly; compilation triggers lazily
       future_kernel.run(grid, args)
```

---

### triton.InterpreterError

```python
InterpreterError(error_message: Optional[str] = None)
```

**`triton.InterpreterError(error_message=None)`**

Exception raised when an error occurs during Triton interpreter execution.

### Parameters
error_message : str, optional
    The error message describing the failure.

### Notes
Inherits from `triton.TritonError`.

### Examples
```python
 try:
     raise triton.InterpreterError("Interpreter execution failed")
 except triton.InterpreterError as e:
     print(e)
```

---

### triton.JITFunction

```python
JITFunction(fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None, noinline=None, repr=None, launch_metadata=None)
```

## JITFunction


**`JITFunction(fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None, noinline=None, repr=None, launch_metadata=None)`**

   Just-in-time compiled Triton kernel function.

   This class represents a function decorated with `triton.jit()` that is
   compiled on-demand for GPU execution. It handles kernel specialization,
   caching, compilation, and launching. Instances are created automatically
   when using the `triton.jit()` decorator.

   Parameters
   ----------
   fn : callable
      The Python function to be JIT-compiled as a Triton kernel.
   version : int, optional
      Version number for the kernel (default: None).
   do_not_specialize : iterable of int or str, optional
      Parameter indices or names that should not be specialized during
      compilation (default: None).
   do_not_specialize_on_alignment : iterable of int or str, optional
      Parameter indices or names that should not be specialized based on
      alignment (default: None).
   debug : bool, optional
      Enable debug mode for kernel compilation and execution (default: None).
   noinline : bool, optional
      Prevent inlining of this function when called from other kernels
      (default: None).
   repr : callable, optional
      Custom function to control string representation of the kernel
      (default: None).
   launch_metadata : callable, optional
      Function to provide custom metadata during kernel launch (default: None).

   Attributes
   ----------
   fn : callable
      The original Python function.
   module : str
      Module name where the function is defined.
   version : int or None
      Kernel version number.
   debug : bool
      Debug mode flag.
   noinline : bool
      Inlining prevention flag.
   params : list
      List of `KernelParam` objects representing function parameters.
   arg_names : list of str
      Names of all kernel arguments.
   constexprs : list of int
      Indices of constexpr parameters.

   Methods
   -------
   run(*args, grid, warmup, **kwargs)
      Execute the kernel with the given grid and arguments.
   warmup(*args, grid, **kwargs)
      Perform kernel warmup compilation without execution.
   add_pre_run_hook(hook)
      Register a hook to be called before kernel execution.
   preload(specialization_data)
      Load a pre-compiled kernel specialization from serialized data.

   Notes
   -----
   JITFunction instances should not be instantiated directly. Use the
   `triton.jit()` decorator instead:

```python
   @triton.jit
   def kernel_fn(...):
       ...

The kernel is compiled lazily on first invocation with a specific set of
argument types. Subsequent calls with the same types reuse the cached
compilation.

Kernels are launched using the grid syntax:

.. code-block:: python

   kernel_fn[grid](arg1, arg2, ...)

where ``grid`` can be a tuple of integers or a callable that returns a tuple.

Global variables captured at compile time are checked at runtime to ensure
they haven't changed since compilation.

Examples
--------
Create a JIT-compiled kernel using the decorator:

.. code-block:: python

   import triton
   import triton.language as tl

   @triton.jit
   def add_kernel(x_ptr, y_ptr, output_ptr, n,
                  BLOCK_SIZE: tl.constexpr = 1024):
       pid = tl.program_id(axis=0)
       block_start = pid * BLOCK_SIZE
       offsets = block_start + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       output = x + y
       tl.store(output_ptr + offsets, output, mask=mask)

   # Launch the kernel
   add_kernel[(grid_size,)](x, y, output, n)

Access kernel metadata:

.. code-block:: python

   kernel = add_kernel
   print(kernel.arg_names)      # ['x_ptr', 'y_ptr', 'output_ptr', 'n', 'BLOCK_SIZE']
   print(kernel.constexprs)     # [4] (index of BLOCK_SIZE)

Register a pre-run hook:

.. code-block:: python

   def my_hook(*args, **kwargs):
       print("About to run kernel")

   add_kernel.add_pre_run_hook(my_hook)

See Also
--------
jit : Decorator to create JITFunction instances
KernelParam : Represents a kernel parameter with specialization metadata
```

---

### triton.KernelInterface

```python
KernelInterface()
```

## triton.KernelInterface

**`triton.KernelInterface`**

   Abstract base class for Triton kernel callable interface.

   This class provides the interface for JIT-compiled Triton kernels, enabling
   kernel launch with grid specification via the `fn[grid](*args)` syntax.
   It is typically not instantiated directly but inherited by `JITFunction`
   objects returned from `@triton.jit` decorated functions.

   Parameters
   ----------
   None
      This is an abstract base class and should not be instantiated directly.

   Attributes
   ----------
   run : T
      Abstract method that executes the kernel. Must be implemented by subclasses.

   Methods
   -------
   warmup(*args, grid, **kwargs)
      Warmup the kernel by running it once with the given grid and arguments.
      Useful for profiling or initialization.

   run(*args, grid, warmup, **kwargs)
      Execute the kernel with the specified grid configuration.
      Abstract method that must be implemented by subclasses.

   __getitem__(grid)
      Return a callable proxy that memorizes the grid configuration.
      Enables the `kernel[grid](*args)` launch syntax.

   Notes
   -----
   The `KernelInterface` is designed to work with Python's generic type system.
   On Python 3.12+, generic classes can declare type parameters directly in the
   class definition. On older versions, explicit inheritance from `Generic` is
   required.

   The `__getitem__` method returns a lambda that captures the grid and calls
   `run` with `warmup=False`, enabling the convenient kernel launch pattern::

       kernel[grid](arg1, arg2, ...)

   This class is abstract - the `run` method raises `NotImplementedError` and
   must be overridden by concrete implementations like `JITFunction`.

   Examples
   --------
   Typical usage through `@triton.jit` decorator:

```python
   import triton
   import triton.language as tl

   @triton.jit
   def add_kernel(x_ptr, y_ptr, output_ptr, n):
       pid = tl.program_id(0)
       if pid < n:
           tl.store(output_ptr + pid, tl.load(x_ptr + pid) + tl.load(y_ptr + pid))

   # Launch kernel with grid specification
   grid = (triton.cdiv(n, 256),)
   add_kernel[grid](x, y, output, n)

Direct interaction with the interface (advanced usage):

.. code-block:: python

   # Access the run method directly
   kernel = add_kernel
   kernel.run(*args, grid=grid, warmup=False)

   # Warmup the kernel for profiling
   kernel.warmup(*args, grid=grid)
```

---

### triton.MockTensor

```python
MockTensor(dtype, shape=None)
```

## class triton.MockTensor

Mock tensor object for kernel warmup testing without allocating real GPU memory.

### Parameters
dtype : dtype
    Data type of the mock tensor (e.g., `torch.float32`).
shape : list of int, optional
    Shape of the mock tensor. Default is `[1]`.

### Attributes
dtype : dtype
    The data type of the mock tensor.
shape : list of int
    The shape of the mock tensor.

### Methods
stride()
    Returns the strides for the mock tensor.
data_ptr()
    Returns a mock data pointer (always 0).
ptr_range()
    Returns a mock pointer range (always 0).

### Notes
`MockTensor` is designed to be used with `kernel.warmup()` to test kernel
compilation without allocating actual GPU memory. It mimics the tensor interface
by providing `dtype`, `shape`, `stride()`, and `data_ptr()` methods that
the Triton compiler expects during warmup.

The `data_ptr()` and `ptr_range()` methods return optimistic values (0)
assuming proper alignment and 32-bit pointer range. This is sufficient for
warmup purposes where no actual memory access occurs.

### Examples
```python
 import torch
 import triton
 import triton.language as tl

 @triton.jit
 def my_kernel(x_ptr, y_ptr, n: tl.constexpr):
     # Kernel implementation
     pass

 # Warmup the kernel without allocating real tensors
 my_kernel.warmup(
     MockTensor(torch.float32, [1024]),
     MockTensor(torch.float32, [1024]),
     n=1024,
     grid=(1,)
 )

 # Can also wrap torch dtypes directly
 my_kernel.warmup(
     MockTensor.wrap_dtype(torch.float32),
     grid=(1,)
 )
```

---

### triton.OutOfResources

```python
OutOfResources(required, limit, name)
```

Exception raised when kernel resource requirements exceed hardware limits.

### Parameters
required : int
    The quantity of the resource required by the kernel.
limit : int
    The hardware limit available for the resource.
name : str
    The identifier of the resource (e.g., `"registers"`, `"shared memory"`).

### Notes
This error indicates that the kernel configuration requests more resources
than the GPU supports. Mitigation strategies include reducing block sizes
or decreasing the `num_stages` parameter.

### Examples
```python
 import triton

 try:
     kernel.run(...)
 except triton.OutOfResources as e:
     print(e)
     # out of resource: shared memory, Required: 65536, Hardware limit: 49152. ...
```

---

### triton.TensorWrapper

```python
TensorWrapper(base, dtype)
```

**`triton.TensorWrapper(base, dtype)`**

   Wrapper that overrides the `dtype` of a base tensor while delegating
   memory and device properties.

   Parameters
   ----------
   base : tensor-like
       The underlying tensor object (e.g., `torch.Tensor`).
   dtype : dtype
       The data type to expose via this wrapper.

   Attributes
   ----------
   dtype : dtype
       The overridden data type.
   base : tensor-like
       The original underlying tensor.
   data : memory
       Points to `base.data`.
   device : device
       Points to `base.device`.
   shape : tuple
       Points to `base.shape`.

   Notes
   -----
   Methods such as `data_ptr()`, `stride()`, `cpu()`, `to()`, and
   `clone()` delegate to the base tensor but preserve the wrapper's
   `dtype` in returned objects where applicable.

   Primarily used internally by Triton to reinterpret tensor dtypes without
   altering the underlying memory allocation.

   Examples
   --------
```python
   >>> import torch
   >>> import triton
   >>> base = torch.zeros((10,), dtype=torch.float32)
   >>> wrapper = triton.TensorWrapper(base, dtype=torch.int32)
   >>> wrapper.dtype
   torch.int32
   >>> wrapper.shape
   torch.Size([10])
```

---

### triton.TritonError

**`TritonError`**

    Base class for all Triton-specific exceptions.

    Parameters
    ----------
    *args : tuple
        Arguments passed to the base `Exception` class.

    Notes
    -----
    Subclasses of this error indicate specific failure conditions within
    the Triton compiler or runtime. Catch this exception to distinguish
    Triton errors from standard Python exceptions.

    Examples
    --------
```python
     import triton

     try:
         raise triton.TritonError("Compilation failed")
     except triton.TritonError as e:
         print(f"Caught Triton error: {e}")
```

---

### triton.autotune

```python
autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False)
```

## triton.autotune

Decorator for auto-tuning a `triton.jit`'d function.

### Parameters

configs : list[triton.Config]
    A list of `triton.Config` objects defining the configurations to benchmark.

key : list[str]
    A list of argument names whose change in value will trigger the evaluation of all provided configs.

prune_configs_by : dict, optional
    A dict of functions used to prune configs before benchmarking. Fields:
    
    - 'perf_model': performance model used to predicate running time with different configs, returns running time
    - 'top_k': number of configs to bench
    - 'early_config_prune': a function used to prune configs. It should have the signature
      `prune_configs_by(configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]`
      and return pruned configs. It should return at least one config.

reset_to_zero : list[str], optional
    A list of argument names whose value will be reset to zero before evaluating any configs.

restore_value : list[str], optional
    A list of argument names whose value will be restored after evaluating any configs.

pre_hook : callable, optional
    A function that will be called before the kernel is called. This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'. Parameters:
    
    - 'kwargs': a dict of all arguments passed to the kernel
    - 'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook

post_hook : callable, optional
    A function that will be called after the kernel is called. This overrides the default post_hook used for 'restore_value'. Parameters:
    
    - 'kwargs': a dict of all arguments passed to the kernel
    - 'exception': the exception raised by the kernel in case of a compilation or runtime error

warmup : int, optional
    Warmup time (in ms) to pass to benchmarking (deprecated).

rep : int, optional
    Repetition time (in ms) to pass to benchmarking (deprecated).

do_bench : callable, optional
    A benchmark function to measure the time of each run. Signature: `lambda fn, quantiles`.

use_cuda_graph : bool, optional
    Whether to use CUDA graphs for benchmarking. Defaults to False.

cache_results : bool, optional
    Whether to cache autotune timings to disk. Defaults to False.

### Returns

decorator : callable
    A decorator that wraps the JIT function with auto-tuning capability.

### Notes

When all configurations are evaluated, the kernel will run multiple times. This means that whatever value the kernel updates will be updated multiple times. To avoid this undesired behavior, use the `reset_to_zero` argument, which resets the value of the provided tensor to zero before running any configuration.

If the environment variable `TRITON_PRINT_AUTOTUNING` is set to `"1"`, Triton will print a message to stdout after autotuning each kernel, including the time spent autotuning and the best configuration.

The `warmup`, `rep`, and `use_cuda_graph` parameters are deprecated.

### Examples

```python
 @triton.autotune(
     configs=[
         triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
         triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
     ],
     key=['x_size']  # configs will be evaluated when x_size changes
 )
 @triton.jit
 def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
     ...
```

---

### triton.cdiv

```python
cdiv(x: int, y: int)
```

**`triton.cdiv(x, y)`**

   Compute the ceiling of the division of `x` by `y`.

   Parameters
   ----------
   x : int
       The dividend.
   y : int
       The divisor.

   Returns
   -------
   int
       The smallest integer greater than or equal to `x / y`.

   Notes
   -----
   This is a constexpr function evaluated at compile time. It is
   equivalent to `(x + y - 1) // y`. Commonly used to calculate
   grid dimensions for kernel launches.

   Examples
   --------
```python
   import triton

   # Calculate grid size for a vector of size 1024 with block size 256
   grid = (triton.cdiv(1024, 256), 1, 1)  # (4, 1, 1)
```

---

### triton.compile

```python
compile(src, target=None, options=None, _env_vars=None)
```

## compile

Compile a Triton kernel from source code or IR representation.

**`triton.compile(src, target=None, options=None, _env_vars=None)`**

   Compiles Triton kernel source into executable GPU code with caching support.

   Parameters
   ----------
   src : ASTSource or str
       Source to compile. Either an `ASTSource` object (from a `@triton.jit`
       decorated function) or a file path string to an IR file (e.g., `.ttir`,
       `.ttgir`, `.ptx`).

   target : GPUTarget, optional
       GPU target architecture to compile for. If `None`, uses the currently
       active target from the driver. Must be a `GPUTarget` instance.

   options : dict, optional
       Compilation options as a dictionary. Common options include `num_warps`,
       `num_ctas`, and backend-specific flags. If `None`, uses defaults.

   _env_vars : dict, optional
       Environment variables that invalidate the compilation cache. If `None`,
       automatically detected from the environment. Primarily for internal use.

   Returns
   -------
   CompiledKernel
       A compiled kernel object containing the binary code, metadata, and launch
       interface. The kernel can be executed via `kernel[grid](*args)`.

   Notes
   -----
   This function implements a multi-stage compilation pipeline:

   1. Parses source into MLIR intermediate representation
   2. Applies optimization passes per backend
   3. Generates target-specific binary (e.g., `cubin` for NVIDIA, `hsaco` for AMD)
   4. Caches results using a hash of source, options, and environment

   Compilation results are cached to avoid redundant compilation. Cache hits return
   immediately without recompiling. Use `knobs.compilation.always_compile` to force
   recompilation.

   The returned `CompiledKernel` provides:

   - `metadata`: compilation metadata (hash, target, options, etc.)
   - `asm`: dictionary of IR representations at each compilation stage
   - `run`: low-level launch function
   - `__getitem__`: grid-based kernel launcher

   Examples
   --------
   Compile a JIT-decorated function:

```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offset < n
       tl.store(x_ptr + offset, tl.load(x_ptr + offset), mask=mask)

   compiled = triton.compile(kernel, options={"num_warps": 4})
   compiled[grid](x_ptr, n, BLOCK_SIZE=1024)

Compile from an IR file:

.. code-block:: python

   compiled = triton.compile("kernel.ttir", target=my_target)
   print(compiled.metadata.hash)  # Access compilation metadata
   print(compiled.asm.keys())     # View available IR representations

Check cache behavior:

.. code-block:: python

   # First call compiles
   kernel1 = triton.compile(src)

   # Second call returns cached result
   kernel2 = triton.compile(src)
   assert kernel1.hash == kernel2.hash
```

---

### triton.constexpr_function

```python
constexpr_function(fn)
```

## constexpr_function


**`constexpr_function(fn)`**

    Wrap a Python function for compile-time evaluation with constexpr arguments.

    Parameters
    ----------
    fn : callable
        The Python function to wrap. Can be any callable that operates on
        constexpr values and returns a Python object.

    Returns
    -------
    ConstexprFunction
        A wrapped function object that can be called at compile-time within
        Triton kernels with constexpr arguments and returns a constexpr result.

    Notes
    -----
    This decorator enables Python functions to be evaluated during kernel
    compilation rather than at runtime. The wrapped function:

    * Can only operate on constexpr arguments (values known at compile-time)
    * Returns a constexpr result that can be used in other compile-time
      computations
    * Cannot access runtime tensor data or GPU memory
    * Is useful for computing compile-time constants, shapes, or other
      metadata

    The function is executed by the Triton compiler during kernel compilation,
    not during kernel execution.

    Examples
    --------
    >>> import triton
    >>> import triton.language as tl
    >>>
    >>> @triton.constexpr_function
    ... def compute_block_size(n: tl.constexpr):
    ...     return tl.constexpr(128 if n > 1024 else 64)
    >>>
    >>> @triton.jit
    ... def kernel(x_ptr, n: tl.constexpr):
    ...     block_size = compute_block_size(n)
    ...     # block_size is a constexpr value known at compile-time
    ...     pass

    See Also
    --------
    triton.jit : Decorator for JIT-compiling GPU kernels
    triton.language.constexpr : Mark a value as compile-time constant

---

### triton.heuristics

```python
heuristics(values)
```

## triton.heuristics


**`heuristics(values)`**

   Decorator for specifying how the values of certain meta-parameters may be computed.

   This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

   Parameters
   ----------
   values : dict[str, Callable[[dict[str, Any]], Any]]
       A dictionary of meta-parameter names and functions that compute the value of the meta-parameter. Each function takes a dictionary of kernel arguments as input and returns the computed value.

   Returns
   -------
   decorator : Callable
       A decorator function that wraps the kernel function with heuristic value computation.

   Notes
   -----
   The heuristic functions are called at kernel launch time with a dictionary containing all kernel arguments (both positional and keyword). The returned values are passed as meta-parameters to the kernel.

   Examples
   --------
   >>> import triton
   >>> import triton.language as tl

   Compute the smallest power-of-two greater than or equal to x_size:

```python
   @triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
   @triton.jit
   def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
       # BLOCK_SIZE is automatically computed based on x_size
       ...

Multiple heuristics can be specified:

.. code-block:: python

   @triton.heuristics(values={
       'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size']),
       'NUM_WARPS': lambda args: 4 if args['x_size'] < 1024 else 8
   })
   @triton.jit
   def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr, NUM_WARPS: tl.constexpr):
       ...
```

---

### triton.jit

```python
jit(fn: 'Optional[T]' = None, *, version=None, repr: 'Optional[Callable]' = None, launch_metadata: 'Optional[Callable]' = None, do_not_specialize: 'Optional[Iterable[int | str]]' = None, do_not_specialize_on_alignment: 'Optional[Iterable[int | str]]' = None, debug: 'Optional[bool]' = None, noinline: 'Optional[bool]' = None) -> 'KernelInterface[T]'
```

## triton.jit


**`jit(fn=None, *, version=None, repr=None, launch_metadata=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None, noinline=None)`**

   Decorator for JIT-compiling a Python function into a GPU kernel using the Triton compiler.

   When applied to a function, `@triton.jit` transforms it into a `JITFunction` that can be launched on the GPU with a grid specification. The decorated function is compiled just-in-time based on the types and values of its arguments.

   Parameters
   ----------
   fn : callable, optional
      The function to be JIT-compiled. When used as `@triton.jit` without parentheses, this is passed directly. When used as `@triton.jit(...)` with keyword arguments, this is passed as the first positional argument.
   version : int, optional
      Version number for the kernel. Used for cache invalidation when kernel logic changes.
   repr : callable, optional
      Custom function to generate string representation of the kernel for debugging and logging.
   launch_metadata : callable, optional
      Custom function to generate metadata for kernel launches.
   do_not_specialize : iterable of int or str, optional
      Parameter indices or names that should not be used for kernel specialization. Arguments at these positions will not trigger recompilation when their values change.
   do_not_specialize_on_alignment : iterable of int or str, optional
      Parameter indices or names that should not be specialized based on memory alignment.
   debug : bool, optional
      Enable debug mode for the kernel. When True, additional runtime checks and debugging information are enabled.
   noinline : bool, optional
      Prevent the compiler from inlining this function when called from other JIT functions.

   Returns
   -------
   JITFunction or callable
      If `fn` is provided, returns a `JITFunction` wrapping the decorated function. If `fn` is not provided, returns a decorator function that can be applied to a function.

   Notes
   -----
   When a JIT-decorated function is called, tensor arguments are implicitly converted to pointers if they have a `.data_ptr()` method and a `.dtype` attribute. This allows seamless integration with PyTorch tensors and other array libraries.

   The decorated function will be compiled and executed on the GPU. It has access only to:

   * Python primitives (int, float, bool, etc.)
   * Builtins within the `triton.language` package
   * Arguments passed to the function
   * Other `@triton.jit` decorated functions

   Standard Python library functions and modules outside of Triton are not available within the kernel body.

   Kernels are launched using the grid syntax: `kernel[grid](*args, **kwargs)` where `grid` is a tuple of 3 integers (or a function returning such a tuple) specifying the number of thread blocks in each dimension.

   Examples
   --------
   Basic kernel definition and launch:

```python
   import triton
   import triton.language as tl

   @triton.jit
   def add_kernel(x_ptr, y_ptr, output_ptr, n):
       pid = tl.program_id(axis=0)
       mask = pid < n
       x = tl.load(x_ptr + pid, mask=mask)
       y = tl.load(y_ptr + pid, mask=mask)
       tl.store(output_ptr + pid, x + y, mask=mask)

   # Launch the kernel with a 1D grid
   n = 1024
   grid = (n,)
   add_kernel[grid](x, y, output, n)

Using keyword arguments for customization:

.. code-block:: python

   @triton.jit(
       do_not_specialize=['n'],
       debug=True,
       noinline=False
   )
   def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
       # Kernel implementation
       pass

Custom representation function:

.. code-block:: python

   def custom_repr(kernel):
       return f"CustomKernel[{kernel.__name__}]"

   @triton.jit(repr=custom_repr)
   def my_kernel(x_ptr, n):
       pass
```

---

### triton.must_use_result

```python
must_use_result(x, s=True)
```

## must_use_result


**`must_use_result(x, s=True)`**

   Mark a function or value as requiring its result to be used.

   If the result of a function or operation marked with this decorator is unused, an error will be raised.

   Parameters
   ----------
   x : function or str
       The function to decorate, or a string message describing why the result must be used.
   s : bool, optional
       Whether the result must be used (default True).

   Returns
   -------
   function or object
       The decorated function or the original object with the `_must_use_result` attribute set.

   Notes
   -----
   This is typically used as a decorator on Triton JIT functions or operations that have no side effects. When called with a string as the first argument, it returns a decorator factory that can be applied to functions.

   This helps prevent bugs where the result of a pure operation is accidentally discarded.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    @tl.must_use_result("Result must be assigned")
    def my_kernel(...):
        ...

    # Using as a decorator on operations
    @tl.must_use_result(
        "Note that tl.advance does not have any side effects. To move the block pointer, you need to assign the result of tl.advance to a variable."
    )
    @triton.jit
    def advance(base, offsets):
        ...

    # The result must be assigned to a variable
    ptr = tl.advance(ptr, offsets)
```

---

### triton.next_power_of_2

```python
next_power_of_2(n: int)
```

**`triton.next_power_of_2(n)`**

   Return the smallest power of 2 greater than or equal to `n`.

   Parameters
   ----------
   n : int
       Input integer value.

   Returns
   -------
   int
       The smallest power of 2 greater than or equal to `n`.

   Notes
   -----
   This is a `constexpr_function <triton.constexpr_function>()`, evaluated at
   compile time. Commonly used for determining block sizes or memory allocations.

   Examples
   --------
```python
   import triton

   # Returns 16
   result = triton.next_power_of_2(10)

   # Returns 1024
   result = triton.next_power_of_2(1024)
```

---

### triton.reinterpret

```python
reinterpret(tensor, dtype)
```

**`reinterpret(tensor, dtype)`**

    Create a tensor view interpreting the underlying memory as a different data type.

    Parameters
    ----------
    tensor : object
        Input tensor or `TensorWrapper`. Must implement `data_ptr()`.
    dtype : dtype
        Target data type for the reinterpretation.

    Returns
    -------
    TensorWrapper or object
        A `TensorWrapper` viewing the memory as `dtype`, or the base tensor
        if reverting to the original type.

    Raises
    ------
    TypeError
        If `tensor` does not support reinterpretation.

    Notes
    -----
    No data copying is performed. This operation is equivalent to a bitcast.
    If `tensor` is already a `TensorWrapper` and `dtype` matches the
    base tensor's original dtype, the base tensor is returned directly.

    Examples
    --------
```python
     import triton
     import torch

     # Create a tensor
     tensor = torch.zeros((10,), dtype=torch.float32)

     # Reinterpret memory as int32
     view = triton.reinterpret(tensor, torch.int32)

     # View shares memory with original tensor
     assert view.data_ptr() == tensor.data_ptr()
     print(view.dtype)  # torch.int32
```

---

### triton.set_allocator

```python
set_allocator(allocator: triton.runtime._allocation.Allocator) -> None
```

**`triton.set_allocator(allocator: triton.runtime._allocation.Allocator) -> None`**

   Set the global memory allocator for Triton kernels.

   The allocator function is called during kernel launch for kernels that
   require additional global memory workspace.

   Parameters
   ----------
   allocator : triton.runtime._allocation.Allocator
       A callable object implementing the Allocator protocol. The callable
       must accept `size`, `alignment`, and `stream` arguments and
       return a `Buffer` object.

   Returns
   -------
   None

   Notes
   -----
   If no allocator is set, Triton uses a default `NullAllocator` which
   raises a `RuntimeError` when memory allocation is requested.

   The allocator is stored in a context variable, making it specific to
   the current execution context.

   Examples
   --------
```python
   import triton

   class MyAllocator:
       def __call__(self, size, alignment, stream):
           # Implementation specific logic
           return buffer_object

   triton.set_allocator(MyAllocator())
```

---
