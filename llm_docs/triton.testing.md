# triton.testing

Testing and benchmarking utilities.

*12 APIs documented.*

---

## triton.testing

### triton.testing.Benchmark

```python
Benchmark(x_names: List[str], x_vals: List[Any], line_arg: str, line_vals: List[Any], line_names: List[str], plot_name: str, args: Dict[str, Any], xlabel: str = '', ylabel: str = '', x_log: bool = False, y_log: bool = False, styles=None)
```

## class triton.testing.Benchmark

Configuration class for benchmarking and performance plotting.

This class is used by the `perf_report` function to generate line plots with a concise API. It defines the parameters to sweep during benchmarking and configures plot appearance.

### Parameters
x_names : list of str
    Name of the arguments that should appear on the x axis of the plot.
x_vals : list of any
    List of values to use for the arguments in `x_names`. Can be a list
    of scalars or a list of tuples/lists. If `x_vals` is a list of scalars
    and there are multiple `x_names`, all arguments will have the same value.
    If `x_vals` is a list of tuples/lists, each element should have the same
    length as `x_names`.
line_arg : str
    Argument name for which different values correspond to different lines in
    the plot.
line_vals : list of any
    List of values to use for the argument in `line_arg`.
line_names : list of str
    Label names for the different lines in the plot.
plot_name : str
    Name of the plot (used for saving figures and data files).
args : dict
    Dictionary of keyword arguments to remain fixed throughout the benchmark.
xlabel : str, optional
    Label for the x axis of the plot. Default is empty string.
ylabel : str, optional
    Label for the y axis of the plot. Default is empty string.
x_log : bool, optional
    Whether the x axis should use logarithmic scale. Default is False.
y_log : bool, optional
    Whether the y axis should use logarithmic scale. Default is False.
styles : list of tuple, optional
    A list of tuples, where each tuple contains two elements: a color and a
    linestyle. Used to customize line appearance in plots.

### Notes
The `Benchmark` class is typically used with the `perf_report` decorator
to mark a function for benchmarking. After decoration, call the `run` method
on the returned object to execute benchmarks and generate plots.

### Examples
```python
 import triton
 import triton.testing as tt

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     # kernel implementation
     pass

 def run_kernel(x_ptr, y_ptr, BLOCK_SIZE):
     # wrapper to execute kernel
     pass

 # Define benchmark configuration
 benchmark = tt.Benchmark(
     x_names=['BLOCK_SIZE'],
     x_vals=[2**i for i in range(6, 10)],
     line_arg='dtype',
     line_vals=['float16', 'float32'],
     line_names=['FP16', 'FP32'],
     plot_name='kernel_benchmark',
     args={},
     xlabel='Block Size',
     ylabel='Time (ms)',
     x_log=True,
 )

 # Mark function for benchmarking
 @tt.perf_report(benchmark)
 def bench_kernel(BLOCK_SIZE, dtype):
     # setup and run kernel
     return run_kernel(...)

 # Execute benchmarks and generate plots
 bench_kernel.run(show_plots=True, save_path='./results')
```

---

### triton.testing.Mark

```python
Mark(fn, benchmarks)
```

## class triton.testing.Mark

Wrapper class for benchmarking functions with specified configurations.

### Parameters
fn : callable
    The function to benchmark. Should accept arguments matching the benchmark
    configuration and return timing metrics (mean, min, max) or just mean.
benchmarks : Benchmark or list of Benchmark
    Benchmark configuration(s) defining x-axis values, line values, plot
    settings, and other benchmark parameters.

### Attributes
fn : callable
    The wrapped benchmark function.
benchmarks : Benchmark or list of Benchmark
    The benchmark configurations.

### Methods
run(show_plots=False, print_data=False, save_path='', return_df=False, **kwargs)
    Execute the benchmarks and optionally save results.

    Parameters
    ----------
    show_plots : bool, optional
        Whether to display matplotlib plots (default: False).
    print_data : bool, optional
        Whether to print benchmark data to stdout (default: False).
    save_path : str, optional
        Directory path to save plots and CSV results (default: '').
    return_df : bool, optional
        Whether to return pandas DataFrame(s) with results (default: False).
    **kwargs
        Additional keyword arguments passed to the benchmarked function.

    Returns
    -------
    df : pandas.DataFrame or list of pandas.DataFrame or None
        If `return_df=True`, returns DataFrame(s) containing benchmark
        results. Otherwise returns None.

    Notes
    -----
    The benchmarked function should return either a single value (mean) or
    a tuple of (mean, min, max) timing values. Results are saved as PNG
    plots and CSV files if `save_path` is provided.

### Examples
```python
 import triton
 import triton.testing as tt

 @triton.jit
 def kernel_fn(...):
     # kernel implementation
     pass

 def run_benchmark(...):
     # benchmark harness
     return triton.testing.do_bench(lambda: kernel_fn[...])

 # Create benchmark configuration
 bench = tt.Benchmark(
     x_names=['size'],
     x_vals=[2**i for i in range(10, 20)],
     line_arg='provider',
     line_vals=['triton', 'torch'],
     line_names=['Triton', 'PyTorch'],
     plot_name='vector-add-benchmark',
     args={},
     ylabel='GB/s'
 )

 # Mark function for benchmarking
 benchmark = tt.perf_report(bench)(run_benchmark)

 # Execute benchmarks
 benchmark.run(save_path='results', print_data=True)
```

---

### triton.testing.assert_close

```python
assert_close(x, y, atol=None, rtol=None, err_msg='')
```

## triton.testing.assert_close

**`triton.testing.assert_close(x, y, atol=None, rtol=None, err_msg='')`**

   Asserts that two inputs are element-wise close within specified tolerances.

   Parameters
   ----------
   x : scalar, list, numpy.ndarray, or torch.Tensor
       The first input to compare.
   y : scalar, list, numpy.ndarray, or torch.Tensor
       The second input to compare.
   atol : float, optional
       Absolute tolerance. Default is `1e-2`.
   rtol : float, optional
       Relative tolerance. Default is `0`.
   err_msg : str, optional
       Error message to display if the assertion fails. Default is empty string.

   Returns
   -------
   None
       This function does not return a value. It raises an `AssertionError` if
       the inputs are not close within the specified tolerances.

   Raises
   ------
   AssertionError
       If `x` and `y` are not close within the given tolerances.

   Notes
   -----
   - Inputs are converted to PyTorch tensors, then to NumPy arrays for comparison.
   - `bfloat16` tensors are converted to `float32` before comparison.
   - Uses `numpy.testing.assert_allclose` for arrays with size > 1.
   - Uses `numpy.allclose` for scalar or size-1 arrays to provide better error messages.
   - NaN values are considered equal (`equal_nan=True`).

   Examples
   --------
```python
   import torch
   import triton.testing as tt

   # Compare two tensors
   x = torch.tensor([1.0, 2.0, 3.0])
   y = torch.tensor([1.01, 2.02, 3.03])
   tt.assert_close(x, y, atol=1e-2)

   # Compare scalars
   tt.assert_close(1.0, 1.001, atol=1e-3)

   # Custom error message
   tt.assert_close(x, y, atol=1e-3, err_msg="Values mismatch: ")
```

---

### triton.testing.cuda_memcheck

```python
cuda_memcheck(**target_kwargs)
```

**`triton.testing.cuda_memcheck(**target_kwargs)`**

    Decorator to run specific test configurations under `cuda-memcheck`.

    Wraps a test function to re-execute it via `cuda-memcheck` if the call
    arguments match the specified `target_kwargs`. Validates that no memory
    access errors occur during execution.

    Parameters
    ----------
    **target_kwargs : dict
        Keyword arguments used to filter test configurations. If the test is
        invoked with kwargs that include these items, the test will be
        re-executed under `cuda-memcheck`.

    Returns
    -------
    decorator : callable
        A decorator that wraps the test function.

    Notes
    -----
    Requires the `cuda-memcheck` binary to be available in the system PATH.
    The wrapped test function must be a `pytest` test and accept a
    `request` fixture. Sets `PYTORCH_NO_CUDA_MEMORY_CACHING=1` during
    memcheck execution. Asserts that `cuda-memcheck` exits with code 0 and
    reports 0 errors.

    Examples
    --------
    Apply the decorator to a pytest test function. The test must accept the
    `request` fixture.

```python
     import triton.testing as tt

     @tt.cuda_memcheck(block_size=128)
     def test_my_kernel(block_size, request):
         # Define kernel logic here
         # This test will run under cuda-memcheck when block_size=128
         pass
```

---

### triton.testing.do_bench

```python
do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode='mean')
```

Benchmark the runtime of a GPU kernel or function.

Performs multiple repetitions of the function with warmup iterations,
clearing GPU cache between runs to ensure accurate timing measurements.

### Parameters
fn : callable
    Function to benchmark. Should be a GPU kernel or function that can be
    timed using CUDA events.
warmup : int, optional
    Warmup time in milliseconds (default is 25). The function will be
    executed multiple times during warmup to ensure GPU is ready.
rep : int, optional
    Repetition time in milliseconds (default is 100). Determines how many
    benchmark iterations to run based on estimated function runtime.
grad_to_none : list of torch.Tensor, optional
    List of tensors whose gradients should be reset to None before each
    benchmark iteration. Useful when benchmarking functions that include
    backward passes.
quantiles : list of float, optional
    Performance percentiles to return (e.g., [0.2, 0.8] for 20th and 80th
    percentiles). If provided, overrides return_mode.
return_mode : str, optional
    Statistical measure to return. Options are "min", "max", "mean",
    "median", or "all" (default is "mean").

### Returns
float or list of float
    Benchmark timing results. Returns a single float based on return_mode,
    or a list of floats if quantiles is specified or return_mode is "all".

### Notes
The function automatically estimates the runtime of `fn` to determine
appropriate warmup and repetition counts. GPU L2 cache is cleared before
each benchmark iteration to ensure consistent measurements.

### Examples
```python
 >>> import triton
 >>> import triton.testing
 >>> import torch
 >>> @triton.jit
 ... def kernel_fn(...):
 ...     pass
 >>> # Benchmark with default settings
 >>> result = triton.testing.do_bench(lambda: kernel_fn[grid](...))
 >>> # Benchmark with custom quantiles
 >>> result = triton.testing.do_bench(lambda: kernel_fn[grid](...),
 ...                                  quantiles=[0.2, 0.5, 0.8])
 >>> # Get all timing measurements
 >>> all_times = triton.testing.do_bench(lambda: kernel_fn[grid](...),
 ...                                     return_mode="all")
```

---

### triton.testing.do_bench_cudagraph

```python
do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode='mean')
```

## triton.testing.do_bench_cudagraph

**`do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode='mean')`**

   Benchmark the runtime of the provided function using CUDA graphs.

   This function measures kernel execution time by constructing a CUDA graph
   with unrolled function calls to minimize host overhead. It is more accurate
   than `do_bench()` for measuring kernel performance without CPU launch
   overhead.

   Parameters
   ----------
   fn : callable
      Function to benchmark. Should contain CUDA kernel launches.
   rep : int, optional
      Total repetition time in milliseconds. Default is 20.
   grad_to_none : list of torch.Tensor, optional
      List of tensors whose gradients should be reset to None before each run.
      Useful when benchmarking functions with backward passes.
   quantiles : list of float, optional
      Quantiles to compute (e.g., [0.2, 0.8] for 20th and 80th percentiles).
      If provided, overrides return_mode.
   return_mode : str, optional
      Statistical measure to return. Options are "min", "max", "mean",
      "median", or "all". Default is "mean".

   Returns
   -------
   float or list of float
      Runtime measurement in milliseconds. Type depends on return_mode:
      - "min", "max", "mean", "median": single float value
      - "all": list of all measured times
      - With quantiles: list of quantile values or single value if one quantile

   Notes
   -----
   This function differs from `do_bench()` in that it uses CUDA graphs to
   eliminate host launch overhead, providing more accurate kernel-only timing.
   The benchmarking process:

   1. Warmup: Runs the function once to initialize CUDA context.
   2. Estimation: Runs 5 iterations to estimate kernel runtime.
   3. Graph construction: Builds a CUDA graph with unrolled calls based on
      estimated runtime and target repetition time.
   4. Measurement: Replays the graph 10 times and reports statistics.

   CUDA graph creation has ~300ms overhead on A100, so this function is best
   suited for benchmarks where accuracy matters more than setup time.

   Examples
   --------
```python
   import torch
   import triton
   import triton.testing as tt

   @triton.jit
   def kernel_fn(x_ptr, y_ptr, n):
       # Kernel implementation
       pass

   def run_kernel():
       x = torch.randn(1024, device='cuda')
       y = torch.empty_like(x)
       kernel_fn[(1,)](x, y, 1024)
       return y

   # Basic benchmark
   ms = tt.do_bench_cudagraph(run_kernel, rep=100)
   print(f"Kernel runtime: {ms} ms")

   # Get min and max times
   min_ms = tt.do_bench_cudagraph(run_kernel, return_mode='min')
   max_ms = tt.do_bench_cudagraph(run_kernel, return_mode='max')

   # Get all measurements
   all_times = tt.do_bench_cudagraph(run_kernel, return_mode='all')

   # With quantiles (20th, 50th, 80th percentiles)
   percentiles = tt.do_bench_cudagraph(run_kernel, quantiles=[0.2, 0.5, 0.8])
```

---

### triton.testing.get_dram_gbps

```python
get_dram_gbps(device=None)
```

## triton.testing.get_dram_gbps

**`get_dram_gbps(device=None)`**

   Return the DRAM bandwidth in GB/s for a GPU device.

   Parameters
   ----------
   device : int, optional
       Device index to query. If `None`, uses the currently active device.

   Returns
   -------
   bw_gbps : float
       DRAM bandwidth in gigabytes per second (GB/s).

   Notes
   -----
   The bandwidth is calculated from device properties using the formula:

   .. math::

      \text{BW} = \frac{\text{mem\_clock\_khz} \times \text{bus\_width} \times 2}{10^6 \times 8}

   where `mem_clock_rate` is in kHz and `mem_bus_width` is in bits.

   Examples
   --------
   >>> import triton.testing as tt
   >>> bw = tt.get_dram_gbps()
   >>> print(f"DRAM bandwidth: {bw:.2f} GB/s")
   DRAM bandwidth: 1935.00 GB/s

   Specify a device index:

   >>> bw = tt.get_dram_gbps(device=0)

---

### triton.testing.get_max_simd_tflops

```python
get_max_simd_tflops(dtype, clock_rate, device=None)
```

## get_max_simd_tflops


.. autofunction:: get_max_simd_tflops

**`get_max_simd_tflops(dtype, clock_rate, device=None)`**

    Calculate the maximum theoretical SIMD TFLOPS for a GPU device.

    Computes peak floating-point performance based on the number of streaming
    multiprocessors, clock rate, and operations per sub-core for the given
    data type and GPU compute capability.

    Parameters
    ----------
    dtype : torch.dtype
        Data type for computation. Supported types depend on compute capability:
        `torch.float32`, `torch.float16`, or `torch.bfloat16`.
    clock_rate : float
        GPU clock rate in MHz.
    device : int, optional
        CUDA device ID. Defaults to current device if not specified.

    Returns
    -------
    tflops : float
        Maximum theoretical TFLOPS (tera floating-point operations per second).

    Raises
    ------
    RuntimeError
        If the dtype is not supported for the device's compute capability.

    Notes
    -----
    The calculation uses the formula:

    `tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9`

    where `num_subcores = multiprocessor_count * 4`.

    Operations per sub-core vary by compute capability and dtype:

    - Compute capability < 8.0: float32 = 32 ops, float16 = 64 ops
    - Compute capability >= 8.0: float32 = 32 ops, float16/bfloat16 = 64 ops

    Examples
    --------
    >>> import torch
    >>> import triton.testing as tt
    >>> # Calculate max SIMD TFLOPS for float32 at 1400 MHz
    >>> tflops = tt.get_max_simd_tflops(torch.float32, 1400)
    >>> print(f"Peak performance: {tflops:.2f} TFLOPS")

    >>> # Specify device explicitly
    >>> tflops = tt.get_max_simd_tflops(torch.float16, 1200, device=0)

---

### triton.testing.get_max_tensorcore_tflops

```python
get_max_tensorcore_tflops(dtype, clock_rate, device=None)
```

## get_max_tensorcore_tflops


**`get_max_tensorcore_tflops(dtype, clock_rate, device=None)`**

   Calculate the maximum theoretical TFLOPS for Tensor Core operations on a GPU.

   Parameters
   ----------
   dtype : torch.dtype
       Data type for the computation. Supported types depend on GPU compute capability:
       - Compute capability < 8.0: only `torch.float16`
       - Compute capability >= 8.0: `torch.float32`, `torch.int32`, `torch.float16`,
         `torch.bfloat16`, `torch.int16`, `torch.int8`, `tl.float8e4nv`,
         `tl.float8e4b15`, `tl.float8e5`
   clock_rate : float
       GPU clock rate in MHz.
   device : int, optional
       CUDA device ID. Defaults to current device if not specified.

   Returns
   -------
   tflops : float
       Maximum theoretical TFLOPS (tera floating-point operations per second) for
       Tensor Core operations with the given dtype and clock rate.

   Notes
   -----
   The calculation is based on the number of Tensor Core sub-cores per multiprocessor
   and the operations per sub-core, which varies by dtype and compute capability:

   - Compute capability < 8.0: 256 ops per sub-core (2 4x4x4 Tensor Cores)
   - Compute capability >= 8.0:
     - `float32`, `int32`: 256 ops per sub-core
     - `float16`, `bfloat16`, `int16`: 512 ops per sub-core
     - `int8`, `float8` variants: 1024 ops per sub-core

   Formula: `tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9`

   where `num_subcores = multiprocessor_count * 4`.

   Examples
   --------
   >>> import torch
   >>> import triton.testing as tt
   >>> # Calculate max TFLOPS for FP16 on current device at 1410 MHz
   >>> tflops = tt.get_max_tensorcore_tflops(torch.float16, 1410)
   >>> print(f"{tflops:.2f} TFLOPS")
   312.48 TFLOPS

---

### triton.testing.nvsmi

```python
nvsmi(attrs)
```

Query GPU attributes via the `nvidia-smi` command-line utility.

### Parameters
attrs : list of str
    List of attribute names to query (e.g., `['clocks.current.sm']`).

### Returns
list of int
    List of integer values corresponding to the requested attributes.

### Notes
Executes `nvidia-smi` as a subprocess targeting device index 0.
Output is parsed as CSV and converted to integers.

### Examples
```python
 import triton.testing
 triton.testing.nvsmi(['clocks.current.sm'])
```

---

### triton.testing.perf_report

```python
perf_report(benchmarks)
```

**`perf_report(benchmarks)`**

   Mark a function for benchmarking.

   Decorator that wraps a function with benchmarking configurations. The
   returned object provides a `.run` method to execute the benchmarks
   and generate performance reports with plots and data.

   Parameters
   ----------
   benchmarks : Benchmark or list of Benchmark
       Benchmarking configurations. Can be a single `Benchmark`
       instance or a list of `Benchmark` instances defining the
       parameters to sweep during benchmarking.

   Returns
   -------
   wrapper : callable
       A decorator function that wraps the target benchmark function. The
       wrapped function returns a `Mark` object with a `run`
       method to execute the benchmarks.

   Notes
   -----
   The benchmark function should accept the parameters defined in
   `benchmarks` (x_names, line_arg, and args). The function should
   return timing measurements, typically from `do_bench()` or
   `do_bench_cudagraph()`.

   The `run` method supports the following options:

   * `show_plots` : Display matplotlib plots
   * `print_data` : Print benchmark results to stdout
   * `save_path` : Directory to save plots and CSV data
   * `return_df` : Return pandas DataFrame with results

   Examples
   --------
   >>> import triton
   >>> import triton.testing as tt
   >>>
   >>> # Define benchmark configuration
   >>> benchmark = tt.Benchmark(
   ...     x_names=['size'],
   ...     x_vals=[2**i for i in range(10, 20)],
   ...     line_arg='provider',
   ...     line_vals=['triton', 'torch'],
   ...     line_names=['Triton', 'PyTorch'],
   ...     plot_name='vector-add-performance',
   ...     args={},
   ...     ylabel='GB/s',
   ... )
   >>>
   >>> # Mark function for benchmarking
   >>> @triton.testing.perf_report(benchmark)
   ... def benchmark_kernel(size, provider):
   ...     # Benchmark implementation
   ...     return tt.do_bench(lambda: kernel_fn(size))
   >>>
   >>> # Run benchmarks and save results
   >>> benchmark_kernel.run(save_path='results', print_data=True)

---

### triton.testing.set_gpu_clock

```python
set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215)
```

## triton.testing.set_gpu_clock

**`triton.testing.set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215)`**

   Context manager that locks GPU SM and memory clocks to specified frequencies
   for benchmarking.

   Sets the GPU clocks using `nvidia-smi`, verifies the settings, and yields
   theoretical compute throughput (TFLOPS) and memory bandwidth (GBPS). Restores
   default clock settings on exit.

   Parameters
   ----------
   ref_sm_clock : int, optional
       Reference SM clock frequency in MHz. Default is 1350.
   ref_mem_clock : int, optional
       Reference memory clock frequency in MHz. Default is 1215.

   Yields
   ------
   tflops : float
       Theoretical tensor core throughput in TFLOPS.
   gbps : float
       Theoretical memory bandwidth in GBPS.

   Raises
   ------
   AssertionError
       If GPU clocks cannot be set to the specified frequencies within
       10 MHz tolerance.

   Notes
   -----
   This context manager requires `nvidia-smi` to be available in PATH and
   sufficient permissions to modify GPU clock settings. Only GPU 0 is configured.

   TFLOPS calculation assumes: 108 SMs, 4 tensor cores per SM, 256 FMA ops per
   cycle, 2 operations per FMA.

   GBPS calculation assumes: 640-bit memory interface, DDR (2x) transfer rate.

   Clock settings are automatically reset when exiting the context.

   Examples
   --------
```python
   import triton.testing as tt

   with tt.set_gpu_clock(ref_sm_clock=1350, ref_mem_clock=1215) as (tflops, gbps):
       print(f"Theoretical TFLOPS: {tflops}")
       print(f"Theoretical GBPS: {gbps}")
       # Run benchmarks here with fixed clock speeds

   # GPU clocks are automatically restored after context exits
```

---
