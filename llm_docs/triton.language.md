# triton.language

Triton language primitives — tensor ops, math, memory, atomics, reductions, scans, and random number generation.

*140 APIs documented.*

---

## triton.language

### triton.language.PropagateNan

Enumeration controlling NaN propagation behavior in floating-point operations.

### Attributes
NONE
    Do not propagate NaNs. NaN inputs may produce non-NaN outputs.
ALL
    Propagate NaNs according to IEEE 754. Any NaN input produces a NaN output.

### Notes
Pass this enumeration to operations such as `triton.language.reduce()` to
control floating-point semantics.

### Examples
```python
 import triton.language as tl

 # Configure reduction to propagate NaNs
 tl.reduce(x, axis=0, combine_fn=fn, propagate_nan=tl.PropagateNan.ALL)
```

---

### triton.language.TRITON_MAX_TENSOR_NUMEL

.. py:data:: triton.language.TRITON_MAX_TENSOR_NUMEL

   Maximum number of elements allowed in a Triton tensor.

   Defines the compile-time upper bound on the total number of elements
   (numel) supported in a tensor operation. This limit ensures compatibility
   with hardware address spaces and internal compiler IR constraints.

   Notes
   -----
   Exceeding this limit may result in compilation errors or undefined
   behavior. The value corresponds to the maximum signed 32-bit integer
   (`2147483647`), restricting tensor sizes to fit within standard
   indexing ranges.

   Examples
   --------
   Access the constant value:

```python
   import triton.language as tl

   print(tl.TRITON_MAX_TENSOR_NUMEL)
   # 2147483647
```

---

### triton.language.abs

```python
abs(x, _semantic=None)
```

## triton.language.abs


**`abs(x, _semantic=None)`**

   Computes the element-wise absolute value of `x`.

   Parameters
   ----------
   x : tl.tensor
       The input tensor. Can be floating-point, signed integer, or unsigned
       integer dtype.

   Returns
   -------
   tl.tensor
       A tensor containing the absolute value of each element in `x`.
       The output has the same shape and dtype as the input.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.abs()` instead of `abs(x)`.

   Behavior varies by dtype:

   - **Floating-point**: Uses GPU fabs instruction
   - **Signed integer**: Uses GPU iabs instruction
   - **Unsigned integer**: No-op (returns input unchanged)
   - **fp8e4b15**: Masks sign bit with 0x7F

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def abs_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)
       y = tl.abs(x)  # or x.abs()
       tl.store(y_ptr + offsets, y)
```

---

### triton.language.add

```python
add(x, y, sanitize_overflow: 'constexpr' = True, _semantic=None)
```

## Add


**`add(x, y, sanitize_overflow=True)`**

   Element-wise addition of two tensors.

   Parameters
   ----------
   x : tensor
       The first input tensor.
   y : tensor
       The second input tensor.
   sanitize_overflow : constexpr, optional
       If True (default), enables overflow sanitization. Set to False for
       potentially faster but unchecked arithmetic.

   Returns
   -------
   tensor
       A tensor containing the element-wise sum of `x` and `y`.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.add(y)` instead of `add(x, y)`.

   The `+` operator is overloaded to call this function, so
   `x + y` is equivalent to `tl.add(x, y)`.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n):
        pid = tl.program_id(0)
        offset = pid * tl.arange(0, 64)
        mask = offset < n
        x = tl.load(x_ptr + offset, mask=mask)
        y = tl.load(y_ptr + offset, mask=mask)
        z = tl.add(x, y)
        tl.store(output_ptr + offset, z, mask=mask)

    # Equivalent using the + operator:
    z = x + y

    # Equivalent using member function:
    z = x.add(y)
```

---

### triton.language.advance

```python
advance(base, offsets, _semantic=None)
```

## triton.language.advance


**`advance(base, offsets, _semantic=None)`**

    Advance a block pointer by the given offsets.

    Parameters
    ----------
    base : tensor
        The block pointer to advance.
    offsets : tuple
        The offsets to advance, specified as a tuple with one value per dimension.

    Returns
    -------
    tensor
        The advanced block pointer.

    Notes
    -----
    This function does not have any side effects. To move the block pointer, you
    must assign the result to a variable.

    This function can also be called as a member function on :py`tensor`,
    as `x.advance(...)` instead of `advance(x, ...)`.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(ptr, BLOCK_SIZE: tl.constexpr):
         # Create a block pointer
         block_ptr = tl.make_block_ptr(
             base=ptr,
             shape=[1024],
             strides=[1],
             offsets=[0],
             block_shape=[BLOCK_SIZE],
             order=[0]
         )

         # Load from current position
         data = tl.load(block_ptr)

         # Advance the pointer by BLOCK_SIZE
         block_ptr = tl.advance(block_ptr, [BLOCK_SIZE])

         # Load from new position
         more_data = tl.load(block_ptr)
```

---

### triton.language.arange

```python
arange(start, end, _semantic=None)
```

## triton.language.arange

**`triton.language.arange(start, end, _semantic=None)`**

   Returns contiguous values within the half-open interval `[start, end)`.

   Parameters
   ----------
   start : int32
       Start of the interval. Must be a power of two.
   end : int32
       End of the interval. Must be a power of two greater than `start`.

   Returns
   -------
   tensor
       1-D tensor of contiguous values from `start` to `end - 1`.

   Notes
   -----
   `end - start` must be less than or equal to `TRITON_MAX_TENSOR_NUMEL = 1048576`.

   Both `start` and `end` must be powers of two, which is required for Triton's block memory model.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(
       output_ptr,
       BLOCK_SIZE: tl.constexpr,
   ):
       # Create indices from 0 to BLOCK_SIZE - 1
       offsets = tl.arange(0, BLOCK_SIZE)
       # Use offsets to compute memory addresses
       tl.store(output_ptr + offsets, offsets)

   # Launch kernel with BLOCK_SIZE = 128 (must be power of two)
   kernel[(1,)](output, BLOCK_SIZE=128)
```

---

### triton.language.argmax

```python
argmax(input, axis, tie_break_left=True, keep_dims=False)
```

## argmax


**`argmax(input, axis, tie_break_left=True, keep_dims=False)`**

   Returns the indices of the maximum values along an axis.

   Parameters
   ----------
   input : tensor
       Input tensor.
   axis : int
       Axis along which to find the maximum index. If None, reduce all dimensions.
   tie_break_left : bool, optional
       If True (default), in case of a tie (multiple elements have the same maximum
       value), return the left-most index for values that aren't NaN.
   keep_dims : bool, optional
       If True, keep the reduced dimensions with length 1 (default is False).

   Returns
   -------
   indices : tensor
       Tensor of indices corresponding to the maximum values along the specified axis.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.argmax(...)` instead of `argmax(x, ...)`.

   The reduction operation is associative and commutative. When `tie_break_left=True`,
   ties are broken by selecting the smallest index (left-most). When `tie_break_left=False`,
   tie-breaking behavior is implementation-defined.

   Examples
   --------
```python
   import triton.language as tl

   # Find index of maximum value along axis 0
   x = tl.arange(0, 8).reshape((2, 4))
   indices = tl.argmax(x, axis=0)  # Returns indices of max values in each column

   # Find index of maximum value along axis 1
   indices = tl.argmax(x, axis=1)  # Returns indices of max values in each row

   # Keep reduced dimensions
   indices = tl.argmax(x, axis=0, keep_dims=True)  # Output shape matches input except reduced axis
```

---

### triton.language.argmin

```python
argmin(input, axis, tie_break_left=True, keep_dims=False)
```

## triton.language.argmin


**`argmin(input, axis, tie_break_left=True, keep_dims=False)`**

   Returns the indices of the minimum values along a given axis.

   Parameters
   ----------
   input : Tensor
       The input tensor.
   axis : int
       The dimension along which the reduction should be done.
   tie_break_left : bool, optional
       If True (default), in case of a tie (i.e., multiple elements have the
       same minimum value), return the left-most index. If False, return any
       index with the minimum value.
   keep_dims : bool, optional
       If True, the reduced dimensions are retained with length 1. Default is
       False.

   Returns
   -------
   Tensor
       A tensor containing the indices of the minimum values. The shape
       depends on the `axis` and `keep_dims` parameters.

   Notes
   -----
   This function can also be called as a member function on
   :py`tensor`, as `x.argmin(...)` instead of
   `argmin(x, ...)`.

   The reduction operation is associative and commutative.

   Examples
   --------
```python
   import triton.language as tl

   # Find the index of minimum value along axis 0
   x = tl.arange(0, 16).reshape((4, 4))
   min_idx = tl.argmin(x, axis=0)

   # Keep reduced dimension
   min_idx_keep = tl.argmin(x, axis=0, keep_dims=True)

   # Use as member function
   min_idx_member = x.argmin(axis=1)
```

---

### triton.language.associative_scan

```python
associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None)
```

## triton.language.associative_scan

**`triton.language.associative_scan(input, axis, combine_fn, reverse=False)`**

   Applies an associative scan operation along the specified axis using a custom
   combine function.

   Parameters
   ----------
   input : tensor or tuple of tensors
       The input tensor(s) to perform the scan over.
   axis : int
       The dimension along which to perform the scan.
   combine_fn : callable
       A JIT-compiled function that combines two scalar tensors. Must be
       decorated with `@triton.jit`. The function should take two arguments
       (representing pairs of elements) and return their combined result.
   reverse : bool, optional
       If True, perform the scan in reverse order along the axis (default: False).

   Returns
   -------
   tensor or tuple of tensors
       The scanned output tensor(s) with the same shape as the input. Each
       position contains the cumulative combination of all elements up to that
       position along the specified axis.

   Notes
   -----
   The `combine_fn` must implement an associative operation (i.e.,
   `combine_fn(a, combine_fn(b, c)) == combine_fn(combine_fn(a, b), c)`).
   Common examples include addition, multiplication, or min/max operations.

   This function can also be called as a member function on :py`tensor`,
   as `x.associative_scan(...)` instead of
   `associative_scan(x, ...)`.

   For cumulative sum operations, consider using :py`tl.cumsum()` which is
   optimized for addition scans.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def add_fn(a, b):
        return a + b

    @triton.jit
    def kernel(input_ptr, output_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        offsets = tl.arange(0, N)
        input = tl.load(input_ptr + offsets)
        # Perform cumulative sum scan
        output = tl.associative_scan(input, 0, add_fn)
        tl.store(output_ptr + offsets, output)

    # Example with multiplication scan (cumulative product)
    @triton.jit
    def mul_fn(a, b):
        return a * b

    @triton.jit
    def cumprod_kernel(input_ptr, output_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        offsets = tl.arange(0, N)
        input = tl.load(input_ptr + offsets)
        output = tl.associative_scan(input, 0, mul_fn)
        tl.store(output_ptr + offsets, output)
```

---

### triton.language.assume

```python
assume(cond, _semantic=None)
```

## triton.language.assume


**`assume(cond, _semantic=None)`**

    Allow the compiler to assume a condition is true.

    This function inserts a compiler hint that the given condition `cond`
    always evaluates to true at runtime. This can enable additional optimizations
    in the generated GPU code by eliminating unreachable control flow paths.

    Parameters
    ----------
    cond : tensor or bool
        The condition to assume is true. Must evaluate to a boolean value.
    _semantic : optional
        Internal parameter for semantic analysis. Do not set manually.

    Returns
    -------
    None
        This function has no runtime effect and returns nothing. It only
        provides a hint to the compiler.

    Notes
    -----
    This is equivalent to LLVM's `assume` intrinsic. The condition is
    not checked at runtime; if the assumption is violated, behavior is
    undefined. Use only when you can guarantee the condition holds.

    This function must be called within a `@triton.jit` decorated kernel.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(ptr, n, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK
         # Assume offset is always within bounds
         tlassume(offset < n)
         # Compiler can optimize knowing offset < n is true
         val = tl.load(ptr + offset)
         tl.store(ptr + offset, val * 2)
```

---

### triton.language.atomic_add

```python
atomic_add(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_add


**`atomic_add(pointer, val, mask=None, sem=None, scope=None)`**

   Performs an atomic add at the memory location specified by `pointer`.

   Return the data stored at `pointer` before the atomic operation.

   Parameters
   ----------
   pointer : tensor of `triton.PointerDType`
       The memory locations to operate on.
   val : tensor
       The values with which to perform the atomic operation. Must have
       `dtype=pointer.dtype.element_ty`.
   mask : tensor of `triton.int1`, optional
       If `mask[idx]` is false, do not perform the atomic operation at
       `pointer[idx]`.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are
       `"acquire"`, `"release"`, `"acq_rel"` (stands for
       "ACQUIRE_RELEASE"), and `"relaxed"`. If not provided, the function
       defaults to using `"acq_rel"` semantics.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of
       the atomic operation. Acceptable values are `"gpu"` (default),
       `"cta"` (cooperative thread array, thread block), or `"sys"`
       (stands for "SYSTEM").

   Returns
   -------
   tensor
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on `tensor`,
   as `x.atomic_add(...)` instead of `atomic_add(x, ...)`.

   Examples
   --------
```python
   @triton.jit
   def atomic_add_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offset)
       # Atomically add x to y_ptr, returns old value
       old_val = tl.atomic_add(y_ptr + offset, x)
       tl.store(x_ptr + offset, old_val)
```

---

### triton.language.atomic_and

```python
atomic_and(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_and


**`atomic_and(pointer, val, mask=None, sem=None, scope=None)`**

   Performs an atomic logical AND at the memory location specified by `pointer`.

   Return the data stored at `pointer` before the atomic operation.

   Parameters
   ----------
   pointer : triton.PointerType, or block of `triton.PointerType`
       The memory locations to operate on.
   val : Block of `pointer.dtype.element_ty`
       The values with which to perform the atomic operation.
   mask : Block of `triton.int1`, optional
       If `mask[idx]` is false, do not perform the atomic operation at `pointer[idx]`.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are `"acquire"`,
       `"release"`, `"acq_rel"` (stands for "ACQUIRE_RELEASE"), and `"relaxed"`.
       If not provided, the function defaults to using `"acq_rel"` semantics.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the atomic operation.
       Acceptable values are `"gpu"` (default), `"cta"` (cooperative thread array, thread block),
       or `"sys"` (stands for "SYSTEM"). The default value is `"gpu"`.

   Returns
   -------
   result : Block
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_and(...)` instead of `atomic_and(x, ...)`.

   Examples
   --------
```python
   @triton.jit
   def kernel(ptr, val, BLOCK: tl.constexpr):
       offs = tl.arange(0, BLOCK)
       old_val = tl.atomic_and(ptr + offs, val)
```

---

### triton.language.atomic_cas

```python
atomic_cas(pointer, cmp, val, sem=None, scope=None, _semantic=None)
```

## atomic_cas


**`atomic_cas(pointer, cmp, val, sem=None, scope=None)`**

   Performs an atomic compare-and-swap at the memory location specified by `pointer`.

   Atomically compares the value at `pointer` with `cmp`. If they are equal, stores `val` at `pointer`. Returns the value stored at `pointer` before the operation.

   Parameters
   ----------
   pointer : tensor
       The memory location to operate on. Must be a pointer or block of pointers.
   cmp : tensor
       The value expected to be found at the atomic location. Must have the same dtype as `pointer.dtype.element_ty`.
   val : tensor
       The value to store if the comparison succeeds. Must have the same dtype as `pointer.dtype.element_ty`.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are `"acquire"`, `"release"`, `"acq_rel"` (ACQUIRE_RELEASE), and `"relaxed"`. Defaults to `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are `"gpu"` (default), `"cta"` (cooperative thread array/thread block), or `"sys"` (SYSTEM). Defaults to `"gpu"`.

   Returns
   -------
   tensor
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`, as `x.atomic_cas(...)` instead of `atomic_cas(x, ...)`.

   The compare-and-swap operation is useful for implementing locks, counters, and other synchronization primitives in GPU kernels.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def atomic_counter_kernel(counter_ptr, increment, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       cmp = tl.load(counter_ptr + offset)
       val = cmp + increment
       old_val = tl.atomic_cas(counter_ptr + offset, cmp, val)
       # old_val contains the value before the atomic operation

   @triton.jit
   def atomic_lock_kernel(lock_ptr, thread_id, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       cmp = 0  # Expected value: unlocked
       val = thread_id  # New value: locked by this thread
       old_val = tl.atomic_cas(lock_ptr + offset, cmp, val)
       # old_val == 0 means this thread acquired the lock
```

---

### triton.language.atomic_max

```python
atomic_max(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_max


**`atomic_max(pointer, val, mask=None, sem=None, scope=None)`**

   Performs an atomic maximum operation at the memory location specified by `pointer`.

   Returns the data stored at `pointer` before the atomic operation.

   Parameters
   ----------
   pointer : tensor
       The memory locations to operate on. Must be a block of pointer type.
   val : tensor
       The values with which to perform the atomic operation. Must have the same
       dtype as `pointer.dtype.element_ty`.
   mask : tensor, optional
       If `mask[idx]` is false, do not perform the atomic operation at
       `pointer[idx]`. Must be a block of `triton.int1`.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are
       `"acquire"`, `"release"`, `"acq_rel"` (ACQUIRE_RELEASE), and
       `"relaxed"`. Defaults to `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the
       atomic operation. Acceptable values are `"gpu"` (default), `"cta"`
       (cooperative thread array, thread block), or `"sys"` (SYSTEM).

   Returns
   -------
   tensor
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_max(...)` instead of `atomic_max(x, ...)`.

   The atomic maximum operation computes `max(*pointer, val)` and stores the
   result back to `*pointer` atomically.

   Examples
   --------
```python
   @triton.jit
   def kernel(ptr, vals, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       old_vals = tl.atomic_max(ptr + offset, vals)
```

---

### triton.language.atomic_min

```python
atomic_min(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_min


**`atomic_min(pointer, val, mask=None, sem=None, scope=None)`**

   Performs an atomic minimum operation at the memory location specified by `pointer`.

   Atomically computes `min(*pointer, val)` and stores the result at `pointer`. Returns the value stored at `pointer` before the atomic operation.

   Parameters
   ----------
   pointer : triton.PointerType or block of triton.PointerType
       The memory locations to operate on. Must be a pointer or block of pointers.
   val : block of pointer.dtype.element_ty
       The values with which to perform the atomic minimum operation. Must have the same dtype as `pointer.dtype.element_ty`.
   mask : block of triton.int1, optional
       If `mask[idx]` is false, the atomic operation is not performed at `pointer[idx]`.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are `"acquire"`, `"release"`, `"acq_rel"` (ACQUIRE_RELEASE), and `"relaxed"`. Defaults to `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the atomic operation. Acceptable values are `"gpu"` (default), `"cta"` (cooperative thread array/thread block), or `"sys"` (SYSTEM).

   Returns
   -------
   block of pointer.dtype.element_ty
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`, as `x.atomic_min(...)` instead of `atomic_min(x, ...)`.

   The atomic minimum operation is useful for parallel reduction patterns where multiple threads may attempt to update the same memory location concurrently.

   Examples
   --------
```python
   @triton.jit
   def atomic_min_kernel(pointer, val, BLOCK_SIZE: tl.constexpr):
       # Perform atomic min across all threads
       old_val = tl.atomic_min(pointer, val)
       # old_val contains the value before the atomic operation

   @triton.jit
   def atomic_min_masked_kernel(pointer, val, mask, BLOCK_SIZE: tl.constexpr):
       # Perform atomic min only where mask is true
       old_val = tl.atomic_min(pointer, val, mask=mask)

   @triton.jit
   def atomic_min_member_fn(x, val):
       # Can also be called as a member function
       old_val = x.atomic_min(val)
```

---

### triton.language.atomic_or

```python
atomic_or(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_or


**`atomic_or(pointer, val, mask=None, sem=None, scope=None)`**

    Performs an atomic logical OR at the memory location specified by `pointer`.

    Returns the data stored at `pointer` before the atomic operation.

    Parameters
    ----------
    pointer : triton.PointerType or block of triton.PointerType
        The memory locations to operate on.
    val : block of dtype=pointer.dtype.element_ty
        The values with which to perform the atomic operation.
    mask : block of triton.int1, optional
        If `mask[idx]` is false, do not perform the atomic operation at
        `pointer[idx]`.
    sem : str, optional
        Specifies the memory semantics for the operation. Acceptable values are
        `"acquire"`, `"release"`, `"acq_rel"` (ACQUIRE_RELEASE), and
        `"relaxed"`. If not provided, defaults to `"acq_rel"`.
    scope : str, optional
        Defines the scope of threads that observe the synchronizing effect of
        the atomic operation. Acceptable values are `"gpu"` (default),
        `"cta"` (cooperative thread array/thread block), or `"sys"`
        (SYSTEM).

    Returns
    -------
    block
        The data stored at `pointer` before the atomic operation.

    Notes
    -----
    This function can also be called as a member function on :py`tensor`,
    as `x.atomic_or(...)` instead of `atomic_or(x, ...)`.

    The atomic OR operation computes `*pointer = *pointer | val` atomically.

    Examples
    --------
```python
     @triton.jit
     def kernel(ptr, val, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         old_val = tl.atomic_or(ptr + offset, val)
```

---

### triton.language.atomic_xchg

```python
atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_xchg


**`atomic_xchg(pointer, val, mask=None, sem=None, scope=None)`**

   Performs an atomic exchange at the memory location specified by `pointer`.

   Atomically stores `val` at `pointer` and returns the value that was previously stored at that location.

   Parameters
   ----------
   pointer : triton.PointerType or block of triton.PointerType
       The memory locations to operate on.
   val : block of pointer.dtype.element_ty
       The values with which to perform the atomic operation.
   mask : block of triton.int1, optional
       If `mask[idx]` is false, do not perform the atomic operation at `pointer[idx]`.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are `"acquire"`,
       `"release"`, `"acq_rel"` (ACQUIRE_RELEASE), and `"relaxed"`. Defaults to `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the atomic operation.
       Acceptable values are `"gpu"` (default), `"cta"` (cooperative thread array/thread block),
       or `"sys"` (SYSTEM).

   Returns
   -------
   block of pointer.dtype.element_ty
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_xchg(...)` instead of `atomic_xchg(x, ...)`.

   Examples
   --------
```python
   @triton.jit
   def kernel(ptr, val, BLOCK: tl.constexpr):
       offs = tl.arange(0, BLOCK)
       old_val = tl.atomic_xchg(ptr + offs, val)
       # old_val contains the value before the exchange
```

---

### triton.language.atomic_xor

```python
atomic_xor(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_xor


**`atomic_xor(pointer, val, mask=None, sem=None, scope=None)`**

   Performs an atomic logical XOR at the memory location specified by `pointer`.

   Returns the data stored at `pointer` before the atomic operation.

   Parameters
   ----------
   pointer : tensor
       The memory locations to operate on. Must be a block of pointer type.
   val : tensor
       The values with which to perform the atomic operation. Must have the same
       dtype as `pointer.dtype.element_ty`.
   mask : tensor, optional
       If `mask[idx]` is false, do not perform the atomic operation at
       `pointer[idx]`.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are
       `"acquire"`, `"release"`, `"acq_rel"` (stands for "ACQUIRE_RELEASE"),
       and `"relaxed"`. If not provided, defaults to `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the
       atomic operation. Acceptable values are `"gpu"` (default), `"cta"`
       (cooperative thread array, thread block), or `"sys"` (stands for
       "SYSTEM").

   Returns
   -------
   tensor
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_xor(...)` instead of `atomic_xor(x, ...)`.

   The atomic XOR operation computes `*pointer = *pointer ^ val` atomically.
   This is useful for implementing concurrent bit manipulation operations.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def atomic_xor_kernel(ptr, val, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       old_val = tl.atomic_xor(ptr + offsets, val)
       # old_val contains the value before XOR

   @triton.jit
   def atomic_xor_masked_kernel(ptr, val, mask, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       old_val = tl.atomic_xor(ptr + offsets, val, mask=mask)
       # Only performs XOR where mask is true
```

---

### triton.language.bfloat16

.. py:data:: triton.language.bfloat16

   bfloat16 floating-point data type.

   Notes
   -----
   The bfloat16 format uses 1 sign bit, 8 exponent bits, and 7 mantissa bits.
   It is designed to accelerate machine learning workloads on GPU hardware,
   particularly NVIDIA Ampere architecture and later. This type provides
   a dynamic range similar to `float32` while maintaining the memory
   bandwidth benefits of `float16`.

   Examples
   --------
   >>> import triton.language as tl
   >>> @triton.jit
   ... def kernel(ptr):
   ...     val = tl.full((1,), 3.14, dtype=tl.bfloat16)
   ...     tl.store(ptr, val)

---

### triton.language.bitonic_merge

```python
bitonic_merge(x, dim: 'core.constexpr' = None, descending: 'core.constexpr' = constexpr[0])
```

## bitonic_merge


**`bitonic_merge(x, dim=None, descending=False)`**

    Performs a bitonic merge operation on a tensor along a specified dimension.

    This function applies a bitonic sorting network to merge elements along the
    specified dimension. The dimension size must be a power of 2. Currently only
    the last (minor) dimension is supported.

    Parameters
    ----------
    x : tensor
        Input tensor to merge. The size of the merge dimension must be a power of 2.
    dim : int, optional
        Dimension along which to perform the merge. If None, uses the last dimension.
        Currently only the last dimension is supported.
    descending : bool, optional
        If True, merge in descending order. If False (default), merge in ascending order.

    Returns
    -------
    tensor
        Tensor with elements merged along the specified dimension.

    Notes
    -----
    This is a low-level primitive used by sorting operations like `sort()` and
    `topk()`. The bitonic merge operation requires the dimension size to be a
    power of 2. Only the last dimension is currently supported.

    Examples
    --------
```python
     import triton.language as tl

     # Merge a 1D tensor of 8 elements (must be power of 2)
     x = tl.arange(0, 8)
     merged = tl.bitonic_merge(x)  # Ascending order

     # Merge in descending order
     merged_desc = tl.bitonic_merge(x, descending=True)

     # 2D tensor - merges along last dimension
     y = tl.reshape(tl.arange(0, 16), (2, 8))
     merged_2d = tl.bitonic_merge(y)  # Merges each row
```

---

### triton.language.block_type

```python
block_type(element_ty: 'dtype', shape: 'List')
```

**`triton.language.block_type`**

   Represents a block (tensor) type with a specific element type and shape.

   `block_type` is a subtype of :py`dtype` that describes multi-dimensional
   tensor types in Triton. It is used internally to represent the type of tensors
   and can be queried to obtain shape and element type information.

   Parameters
   ----------
   element_ty : dtype
       The scalar element data type (e.g., :py:data:`tl.float32`, :py:data:`tl.int16`).
   shape : list of int
       The shape of the block as a list of integers. Empty lists are forbidden
       (0D block types are not allowed).

   Attributes
   ----------
   element_ty : dtype
       The scalar element type of the block.
   shape : tuple of int
       The shape of the block as a tuple of integers.
   numel : int
       The total number of elements in the block (product of shape dimensions).
   nbytes : int
       The total size in bytes of the block (numel × element size).
   scalar : dtype
       Alias for :py`element_ty`.

   Methods
   -------
   to_ir(builder)
       Converts the block type to an MLIR block type using the given builder.
   is_block()
       Returns `True` (distinguishes from scalar dtypes).
   get_block_shapes()
       Returns the shape tuple of the block.
   with_element_ty(scalar_ty)
       Returns a new block_type with the same shape but different element type.
   mangle()
       Returns a mangled string representation for type identification.

   Notes
   -----
   - `block_type` is typically used internally by Triton's type system.
   - Users typically interact with :py`tensor` objects rather than
     directly constructing `block_type` instances.
   - The shape must be non-empty; 0D block types raise a `TypeError`.
   - `block_type` inherits from :py`dtype` and implements the dtype
     interface for type checking and code generation.

   Examples
   --------
   Querying block type information from a tensor:

```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
       x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
       # Access type information
       dtype = x.dtype          # scalar element type (e.g., float32)
       shape = x.shape          # block shape (e.g., (BLOCK_SIZE,))
       is_block = x.type.is_block()  # True for tensor types

Creating a block type directly (advanced usage):

.. code-block:: python

   import triton.language as tl

   # Create a 1D block type with 32 float32 elements
   block_ty = tl.block_type(tl.float32, [32])
   print(block_ty)      # Output: <(32,), float32>
   print(block_ty.nbytes)  # Output: 128 (32 × 4 bytes)
   print(block_ty.shape)   # Output: (32,)

Using ``with_element_ty`` to create a variant with different element type:

.. code-block:: python

   # Create int16 variant with same shape
   int16_block = block_ty.with_element_ty(tl.int16)
   print(int16_block)    # Output: <(32,), int16>
   print(int16_block.nbytes)  # Output: 64 (32 × 2 bytes)
```

---

### triton.language.broadcast

```python
broadcast(input, other, _semantic=None)
```

Broadcast two tensors to a common compatible shape.

Applies NumPy-style broadcasting rules to make two input tensors compatible for element-wise operations.

### Parameters
input : tensor
    The first input tensor.
other : tensor
    The second input tensor.

### Returns
tensor
    A tuple of two tensors broadcasted to a common shape.

### Notes
This function follows standard broadcasting semantics where dimensions of
size 1 can be expanded to match corresponding dimensions in the other
tensor. Both tensors must be broadcastable to a common shape.

### Examples
```python
 @triton.jit
 def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     y = tl.full((BLOCK,), 1.0, dtype=tl.float32)
     # Broadcast x and y to compatible shapes
     x_b, y_b = tl.broadcast(x, y)
     out = x_b + y_b
     tl.store(out_ptr + tl.arange(0, BLOCK), out)
```

---

### triton.language.broadcast_to

```python
broadcast_to(input, *shape, _semantic=None)
```

## triton.language.broadcast_to


**`broadcast_to(input, *shape)`**

   Broadcast a tensor to a new shape.

   Parameters
   ----------
   input : tl.tensor
       The input tensor to broadcast.
   *shape : int or tuple of ints
       The desired shape. Can be passed as individual integers or as a tuple.

   Returns
   -------
   tl.tensor
       A tensor broadcast to the specified shape.

   Notes
   -----
   The shape can be passed as a tuple or as individual parameters. Both of the
   following calls are equivalent::

       broadcast_to(x, (32, 32))
       broadcast_to(x, 32, 32)

   This function can also be called as a member function on :py`tensor`,
   as `x.broadcast_to(...)` instead of `broadcast_to(x, ...)`.

   The input tensor must be broadcastable to the target shape according to
   standard broadcasting rules (dimensions of size 1 can be expanded).

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
        # Broadcast scalar to vector
        scalar = tl.full((1,), 5.0, dtype=tl.float32)
        broadcasted = tl.broadcast_to(scalar, (BLOCK_SIZE,))
        y = x + broadcasted
        tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.cast

```python
cast(input, dtype: 'dtype', fp_downcast_rounding: 'Optional[str]' = None, bitcast: 'bool' = False, _semantic=None)
```

## triton.language.cast


**`cast(input, dtype, fp_downcast_rounding=None, bitcast=False)`**

    Casts a tensor to the given `dtype`.

    Parameters
    ----------
    input : tensor
        The input tensor to cast.
    dtype : tl.dtype
        The target data type.
    fp_downcast_rounding : str, optional
        The rounding mode for downcasting floating-point values. This parameter
        is only used when input is a floating-point tensor and `dtype` is
        a floating-point type with a smaller bitwidth. Supported values are
        `"rtne"` (round to nearest, ties to even) and `"rtz"`
        (round towards zero).
    bitcast : bool, optional
        If true, the tensor is bitcasted to the given `dtype`, instead of
        being numerically casted. Default is `False`.

    Returns
    -------
    tensor
        A tensor with the same shape as `input` but with the specified
        `dtype`.

    Notes
    -----
    This function can also be called as a member function on :py`tensor`,
    as `x.cast(...)` instead of `cast(x, ...)`.

    When `bitcast=True`, the bit representation of the input is
    reinterpreted as the target dtype without numerical conversion. This is
    useful for type-punning operations.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         # Cast float32 to float16
         y = tl.cast(x, tl.float16)
         tl.store(y_ptr + tl.arange(0, BLOCK), y)

     # Using member function syntax
     @triton.jit
     def kernel2(x_ptr, y_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         y = x.cast(tl.int32)
         tl.store(y_ptr + tl.arange(0, BLOCK), y)

     # Bitcast example: reinterpret float32 bits as int32
     @triton.jit
     def kernel3(x_ptr, y_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         y = tl.cast(x, tl.int32, bitcast=True)
         tl.store(y_ptr + tl.arange(0, BLOCK), y)
```

---

### triton.language.cat

```python
cat(input, other, can_reorder=False, dim=0, _semantic=None)
```

## tl.cat

Concatenate two tensors along a specified dimension.

```python
 tl.cat(input, other, can_reorder=False, dim=0, _semantic=None)

```
### Parameters

input : tensor
    The first input tensor.

other : tensor
    The second input tensor.

can_reorder : bool, optional
    Compiler hint. If true, the compiler is allowed to reorder elements while
    concatenating inputs. Only use if the order does not matter (e.g., result
    is only used in reduction ops). Default is `False`.

dim : int, optional
    The dimension to concatenate along. Used when `can_reorder` is `False`.
    Default is `0`.

### Returns

tensor
    The concatenated tensor.

### Notes

When `can_reorder=False`, the following constraints apply:

- Both tensors must have the same rank.
- All dimensions except the concatenation dimension must match.

When `can_reorder=True`, the compiler may optimize the concatenation by
reordering elements, which can improve performance but does not preserve
element order. Use this only when the result is consumed by order-independent
operations (e.g., reductions).

### Examples

Concatenate two tensors along dimension 0:

```python
 @triton.jit
 def kernel(x_ptr, y_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr):
     pid = tl.program_id(0)
     offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = tl.arange(0, N)
     mask = offs_m[:, None] < M
     x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
     y = tl.load(y_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
     out = tl.cat(x, y, dim=0)
     tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)

```
Concatenate with reordering allowed (for reduction operations):

```python
 @triton.jit
 def kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
     offs_n = tl.arange(0, BLOCK_N)
     x = tl.load(x_ptr + offs_n)
     y = tl.load(y_ptr + offs_n)
     combined = tl.cat(x, y, can_reorder=True)
     result = tl.sum(combined, axis=0)
     tl.store(out_ptr, result)
```

---

### triton.language.cdiv

```python
cdiv(x, div)
```

## triton.language.cdiv


**`cdiv(x, div)`**

   Computes the ceiling division of `x` by `div`.

   Parameters
   ----------
   x : Block
       The input number or tensor.
   div : Block
       The divisor.

   Returns
   -------
   Block
       The ceiling division result `ceil(x / div)`.

   Notes
   -----
   This function computes `(x + (div - 1)) // div`, which is equivalent to
   ceiling division for positive integers. It can also be called as a member
   function on :py`tensor`, as `x.cdiv(...)` instead of
   `cdiv(x, ...)`.

   Examples
   --------
```python
   import triton.language as tl

   # Ceiling division of scalar values
   result = tl.cdiv(10, 3)  # Returns 4

   # Ceiling division on tensors
   x = tl.arange(0, 16)
   result = tl.cdiv(x, 4)  # Returns [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]

   # Using member function syntax
   result = x.cdiv(4)  # Equivalent to tl.cdiv(x, 4)
```

---

### triton.language.ceil

```python
ceil(x, _semantic=None)
```

## ceil


**`ceil(x, _semantic=None)`**

   Computes the element-wise ceiling of the input tensor.

   Parameters
   ----------
   x : Block
       The input tensor. Must have floating-point dtype (`fp32` or `fp64`).

   Returns
   -------
   Block
       A tensor containing the ceiling of each element in `x`. The output has
       the same shape and dtype as the input.

   Notes
   -----
   This function only supports `fp32` and `fp64` dtypes. Passing other
   dtypes will result in a compilation error.

   This function can also be called as a member function on :py`tensor`,
   as `x.ceil()` instead of `tl.ceil(x)`.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = tl.ceil(x)
        tl.store(y_ptr + offsets, y)

    # Using member function syntax
    @triton.jit
    def kernel_member(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = x.ceil()
        tl.store(y_ptr + offsets, y)
```

---

### triton.language.clamp

```python
clamp(x, min, max, propagate_nan: 'constexpr' = <PROPAGATE_NAN.NONE: 0>, _semantic=None)
```

## triton.language.clamp


**`clamp(x, min, max, propagate_nan=PropagateNan.NONE)`**

    Clamps the input tensor `x` within the range `[min, max]`.

    Parameters
    ----------
    x : Block
        The input tensor to clamp.
    min : Block
        The lower bound for clamping.
    max : Block
        The upper bound for clamping.
    propagate_nan : tl.PropagateNan, optional
        Whether to propagate NaN values. Applies only to the `x` tensor.
        If either `min` or `max` is NaN, the result is undefined.
        Default is `PropagateNan.NONE`.

    Returns
    -------
    Block
        A tensor with values clamped to the range `[min, max]`.

    Notes
    -----
    Behavior when `min > max` is undefined.

    bfloat16 inputs are automatically promoted to float32 before the operation.

    See Also
    --------
    tl.PropagateNan : Enum controlling NaN propagation behavior.
    tl.minimum : Element-wise minimum operation.
    tl.maximum : Element-wise maximum operation.

    Examples
    --------
```python
     @triton.jit
     def clamp_kernel(x_ptr, min_val, max_val, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         clamped = tl.clamp(x, min_val, max_val)
         tl.store(x_ptr + offsets, clamped)
```

---

### triton.language.condition

```python
condition(arg1, disable_licm=False)
```

## class triton.language.condition

While loop condition wrapper for `triton.jit` functions.

### Parameters
arg1 : tensor
    The loop condition value.
disable_licm : bool, optional
    If True, prevents the compiler from hoisting loop invariant code outside
    the loop. Default is False. This can help avoid creating long live ranges
    within a loop.

### Notes
This is a special wrapper used to annotate while loops in the context of
`triton.jit` functions. It allows users to pass extra attributes to
the compiler.

### Examples
```python
 @triton.jit
 def kernel(...):
     while tl.condition(c, disable_licm=True):
         ...
```

---

### triton.language.const

```python
const()
```

## const


**`const`**

   Type annotation for pointers to constant data.

   This class is used as a type annotation to mark pointers to constant data.
   The :py`store()` function cannot be called with a pointer to const.
   Constness is part of the pointer type and the usual Triton type consistency
   rules apply.

   .. rubric:: Notes

   For example, you cannot have a function that returns a constant pointer in
   one return statement and a non-constant pointer in another. The pointer type
   must be consistent throughout the function.

   .. rubric:: Examples

```python
   import triton
   import triton.language as tl

   @triton.jit
   def load_const_kernel(ptr: tl.const_pointer, out_ptr, BLOCK: tl.constexpr):
       # Load from constant pointer (read-only)
       value = tl.load(ptr + tl.arange(0, BLOCK))
       # Cannot store to const pointer - this would be a type error
       # tl.store(ptr, value)  # ERROR
       tl.store(out_ptr, value)
```

---

### triton.language.constexpr

```python
constexpr(value)
```

**`triton.language.constexpr(value)`**

   Store a value that is known at compile-time.

   This class wraps Python values to mark them as compile-time constants in Triton
   JIT functions. constexpr values are resolved during compilation rather than at
   runtime, enabling compile-time specialization and optimization.

   Parameters
   ----------
   value : int, float, bool, or str
       The compile-time constant value to wrap. Nested constexpr objects are
       automatically unwrapped.

   Attributes
   ----------
   value
       The underlying Python value.
   type
       The constexpr_type associated with this value.

   Methods
   -------
   __repr__()
       Returns string representation as `constexpr[value]`.
   __hash__()
       Returns hash based on value and type.
   __index__()
       Returns the underlying value for indexing operations.
   __bool__()
       Returns boolean value of the underlying value.

   Notes
   -----
   constexpr values are used throughout Triton to mark parameters that should be
   treated as compile-time constants. Common use cases include:

   - Block sizes and shapes (e.g., `BLOCK_SIZE: tl.constexpr`)
   - Loop bounds known at compile-time
   - Configuration flags for kernel specialization

   Arithmetic and comparison operations on constexpr objects return new constexpr
   objects, preserving compile-time knowledge. In interpreter mode, constant
   values may not be wrapped in constexpr, so internal code uses helper functions
   to handle both cases.

   constexpr is typically used as a type annotation in JIT function signatures
   rather than constructed directly in kernel code.

   Examples
   --------
   >>> import triton.language as tl
   >>> @triton.jit
   ... def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
   ...     # BLOCK_SIZE is a compile-time constant
   ...     pid = tl.program_id(0)
   ...     offset = pid * BLOCK_SIZE
   ...     # ... rest of kernel

   >>> # constexpr values support arithmetic
   >>> a = tl.constexpr(8)
   >>> b = tl.constexpr(4)
   >>> c = a + b
   >>> c
   constexpr[12]

   >>> # Comparison operations return constexpr
   >>> a > b
   constexpr[True]

   >>> # Use in type annotations for JIT functions
   >>> @triton.jit
   ... def matmul(a, b, c, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
   ...     # BLOCK_M and BLOCK_N are compile-time constants
   ...     pass

---

### triton.language.constexpr_type

```python
constexpr_type(value)
```

## constexpr_type


**`constexpr_type(value)`**

   Type representation for compile-time constant values in Triton IR.

   Parameters
   ----------
   value : int, float, or dtype
       The compile-time constant value to wrap. This value is known at
       compile-time and does not require runtime computation.

   Attributes
   ----------
   value
       The wrapped constant value.

   Notes
   -----
   `constexpr_type` is used internally by Triton to represent types of
   compile-time constants. It inherits from `base_type` and implements
   type mangling for kernel compilation.

   Users typically interact with `constexpr` values rather than directly
   instantiating `constexpr_type`. The type is automatically created when
   a `constexpr` value is constructed.

   Examples
   --------
   >>> import triton.language as tl
   >>> from triton.language import constexpr_type
   >>> t = constexpr_type(42)
   >>> t
   constexpr_type[42]
   >>> t.value
   42

   >>> # constexpr_type is typically used indirectly through constexpr
   >>> c = tl.constexpr(10)
   >>> c.type
   constexpr_type[10]

---

### triton.language.cos

```python
cos(x, _semantic=None)
```

## triton.language.cos


**`cos(x, _semantic=None)`**

   Computes the element-wise cosine of `x`.

   Parameters
   ----------
   x : tl.tensor
       The input tensor. Must be a floating-point type (`fp32` or `fp64`).

   Returns
   -------
   tl.tensor
       A tensor containing the cosine of each element in `x`. The output has the same shape and dtype as the input.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`, as `x.cos()` instead of `tl.cos(x)`.

   Supported dtypes are `fp32` and `fp64`. Passing other dtypes will result in a compilation error.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def cosine_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)
       y = tl.cos(x)
       tl.store(y_ptr + offsets, y)

   # Or using the member function syntax:
   @triton.jit
   def cosine_kernel_member(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)
       y = x.cos()
       tl.store(y_ptr + offsets, y)
```

---

### triton.language.cumprod

```python
cumprod(input, axis=0, reverse=False)
```

## cumprod


**`cumprod(input, axis=0, reverse=False)`**

   Returns the cumulative product of all elements in the `input` tensor along the provided `axis`.

   Parameters
   ----------
   input : Tensor
       The input tensor.
   axis : int, optional
       The dimension along which the cumulative product is computed. Default is 0.
   reverse : bool, optional
       If True, the scan is performed in the reverse direction. Default is False.

   Returns
   -------
   Tensor
       A tensor containing the cumulative product of the input elements along the specified axis.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`, as `x.cumprod(...)` instead of `cumprod(x, ...)`.

   bfloat16 inputs are automatically promoted to float32 before computation.

   The cumulative product is computed using an associative scan with multiplication as the combine operation.

   Examples
   --------
```python
   import triton.language as tl

   # Compute cumulative product along axis 0
   x = tl.arange(0, 8) + 1  # [1, 2, 3, 4, 5, 6, 7, 8]
   y = tl.cumprod(x, axis=0)  # [1, 2, 6, 24, 120, 720, 5040, 40320]

   # Use as member function
   z = x.cumprod(axis=0)  # Equivalent to tl.cumprod(x, axis=0)

   # Reverse cumulative product
   w = tl.cumprod(x, axis=0, reverse=True)
```

---

### triton.language.cumsum

```python
cumsum(input, axis=0, reverse=False, dtype: 'core.constexpr' = None)
```

### cumsum

Compute the cumulative sum of the elements along a given axis.

### Parameters
input : tensor
    The input tensor.
axis : int, optional
    The axis along which the cumulative sum is computed. Default is 0.
reverse : bool, optional
    If `True`, the scan is performed in the reverse direction. Default is `False`.
dtype : dtype, optional
    The desired data type of the returned tensor. If specified, the input tensor is cast to `dtype` before the operation. If not specified, small integer types (< 32 bits) are upcasted to prevent overflow. `tl.bfloat16` inputs are automatically promoted to `tl.float32`.

### Returns
tensor
    A tensor of the same shape as `input` containing the cumulative sums.

### Notes
This function can also be called as a member function on `tensor`, as `x.cumsum(...)` instead of `cumsum(x, ...)`.

### Examples
```python
 import triton.language as tl

 # Basic usage
 x = tl.arange(0, 8)
 y = tl.cumsum(x)  # [0, 1, 3, 6, 10, 15, 21, 28]

 # Member function usage
 z = x.cumsum()

 # Reverse cumulative sum
 r = tl.cumsum(x, reverse=True)
```

---

### triton.language.debug_barrier

```python
debug_barrier(_semantic=None)
```

## debug_barrier


**`debug_barrier()`**

   Insert a barrier to synchronize all threads in a block.

   This is a debugging primitive that inserts a synchronization barrier at the
   PTX level. All threads in a thread block will wait at this barrier until all
   other threads in the block reach it.

   Parameters
   ----------
   None

   Returns
   -------
   None

   Notes
   -----
   This function is intended for debugging purposes only. It should not be used
   in production code as it may significantly impact performance. The barrier
   ensures that all memory operations issued before the barrier are visible to
   all threads in the block after the barrier.

   Examples
   --------
```python
   @triton.jit
   def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)
       
       # Debug barrier to synchronize threads
       tl.debug_barrier()
       
       y = x * 2
       tl.store(x_ptr + offsets, y)
```

---

### triton.language.device_assert

```python
device_assert(cond, msg='', mask=None, _semantic=None)
```

## triton.language.device_assert


**`device_assert(cond, msg='', mask=None, _semantic=None)`**

    Assert the condition at runtime from the device.

    Requires that the environment variable `TRITON_DEBUG` is set to a value
    besides `0` in order for this to have any effect.

    Parameters
    ----------
    cond : tensor
        The condition to assert. Must be a boolean tensor.
    msg : str, optional
        The message to print if the assertion fails. Must be a string literal.
        Default is empty string.
    mask : tensor, optional
        If provided, only assert where mask is true.

    Returns
    -------
    None

    Notes
    -----
    Using the Python `assert` statement is equivalent to calling this
    function, except that the message argument must be provided and must be
    a string, e.g., `assert pid == 0, "pid != 0"`. The `TRITON_DEBUG`
    environment variable must be set for the `assert` statement to have
    any effect.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(pid):
         # Assert that pid equals 0
         tl.device_assert(pid == 0)
         
         # Equivalent Python assert statement
         assert pid == 0, "pid != 0"
         
         # With custom message
         tl.device_assert(pid < 1024, "pid out of range")
         
         # With mask (only assert for certain threads)
         mask = pid % 2 == 0
         tl.device_assert(pid < 512, "even pid too large", mask=mask)
```

---

### triton.language.device_print

```python
device_print(prefix, *args, hex=False, _semantic=None)
```

## device_print

Print values at runtime from the GPU device.

### Parameters
prefix : str
    A prefix to print before the values. Must be a string literal (compile-time
    constant). Must contain only ASCII printable characters.
*args : tensor or scalar
    The values to print. Can be any tensor or scalar. All non-string arguments
    are converted to tensors.
hex : bool, optional
    If True, print all values in hexadecimal format instead of decimal.
    Default is False.

### Returns
None
    This function prints values as a side effect and does not return a
    meaningful value.

### Notes
String formatting does not work for runtime values. Provide the values you want
to print as separate arguments after the prefix string.

Calling the Python builtin `print` inside a `@triton.jit` function is
equivalent to calling `device_print`, and the argument requirements match
this function (not the normal requirements for Python's `print`).

On CUDA, printf output is streamed through a buffer of limited size (default
approximately 6912 KiB, but may vary across GPUs and CUDA versions). If printf
output is being dropped, increase the buffer size by calling:

```python
 triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)

```
CUDA may raise an error if you attempt to change this value after running a
kernel that uses printfs. The value set may only affect the current device.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(pid):
     # Print program ID with prefix
     tl.device_print("pid", pid)

     # Print multiple values
     tl.device_print("values", pid, pid * 2)

     # Print in hexadecimal format
     tl.device_print("hex", pid, hex=True)

     # Using Python print is equivalent
     print("pid", pid)
```

---

### triton.language.div_rn

```python
div_rn(x, y, _semantic=None)
```

## triton.language.div_rn


**`div_rn(x, y, _semantic=None)`**

   Computes the element-wise precise division of `x` and `y`, rounding to nearest per IEEE 754 standard.

   Parameters
   ----------
   x : tensor
       The dividend input values. Must be floating-point type.
   y : tensor
       The divisor input values. Must be floating-point type.

   Returns
   -------
   tensor
       Element-wise quotient of `x` and `y`.

   Notes
   -----
   This operation uses IEEE 754 round-to-nearest rounding mode for precise floating-point division.

   Only supports `fp32` dtype. Inputs are automatically legalized to a common type.

   Examples
   --------
```python
   @triton.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK + tl.arange(0, BLOCK)
       x = tl.load(x_ptr + offsets)
       y = tl.load(y_ptr + offsets)
       out = tl.div_rn(x, y)
       tl.store(out_ptr + offsets, out)
```

---

### triton.language.dot

```python
dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=triton.language.float32, _semantic=None)
```

## triton.language.dot

**`triton.language.dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=triton.language.float32)`**

   Returns the matrix product of two blocks.

   The two blocks must both be two-dimensional or three-dimensional and have
   compatible inner dimensions. For three-dimensional blocks, performs batched
   matrix multiplication where the first dimension represents the batch.

   Parameters
   ----------
   input : tensor
       The first tensor to be multiplied. Must be a 2D or 3D tensor with scalar
       type in {`int8`, `float8_e5m2`, `float16`, `bfloat16`, `float32`}.
   other : tensor
       The second tensor to be multiplied. Must be a 2D or 3D tensor with scalar
       type in {`int8`, `float8_e5m2`, `float16`, `bfloat16`, `float32`}.
   acc : tensor, optional
       The accumulator tensor. If not None, the result is added to this tensor.
       Must be a 2D or 3D tensor with scalar type in {`float16`, `float32`,
       `int32`}.
   input_precision : str, optional
       How to exercise Tensor Cores for f32 x f32. If the device does not have
       Tensor Cores or inputs are not dtype f32, this option is ignored.
       Default is `"tf32"` for devices with Tensor Cores.

       - NVIDIA options: `"tf32"`, `"tf32x3"`, `"ieee"`
       - AMD options: `"ieee"`, (CDNA3 only) `"tf32"`
   allow_tf32 : bool, optional
       *Deprecated.* If true, `input_precision` is set to `"tf32"`. Only one
       of `input_precision` and `allow_tf32` can be specified (at least one
       must be `None`).
   max_num_imprecise_acc : int, optional
       Maximum number of imprecise accumulations.
   out_dtype : dtype, optional
       Output data type. Default is `tl.float32`.

   Returns
   -------
   result : tensor
       The matrix product of `input` and `other`. If `acc` is provided,
       returns `input @ other + acc`. The output shape is determined by the
       input shapes: for 2D inputs `(M, K)` and `(K, N)`, output is `(M, N)`.
       For 3D inputs with batch dimension `B`, output is `(B, M, N)`.

   Notes
   -----
   When using TF32 precision, float32 inputs may be truncated to TF32 format
   (19-bit floating point) without rounding, which may bias the result. For best
   results, round to TF32 explicitly, or load data using `TensorDescriptor`
   with `round_f32_to_tf32=True`.

   The function validates that input tensors have compatible shapes for matrix
   multiplication. For 2D tensors, the inner dimensions must match. For 3D
   tensors, batch dimensions must match and inner dimensions must be compatible.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def matmul_kernel(
       a_ptr, b_ptr, c_ptr,
       M, N, K,
       BLOCK_SIZE_M: tl.constexpr,
       BLOCK_SIZE_N: tl.constexpr,
       BLOCK_SIZE_K: tl.constexpr,
   ):
       pid = tl.program_id(axis=0)
       num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
       pid_m = pid % num_pid_m
       pid_n = pid // num_pid_m

       offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
       offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
       offs_k = tl.arange(0, BLOCK_SIZE_K)

       a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
       b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]

       accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
       for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
           a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
           b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
           accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
           a_ptrs += BLOCK_SIZE_K
           b_ptrs += BLOCK_SIZE_K * N

       c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
       tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

.. code-block:: python

   # Batched matrix multiplication (3D tensors)
   @triton.jit
   def batched_matmul_kernel(
       a_ptr, b_ptr, c_ptr,
       B, M, N, K,
       BLOCK_SIZE_M: tl.constexpr,
       BLOCK_SIZE_N: tl.constexpr,
       BLOCK_SIZE_K: tl.constexpr,
   ):
       pid_b = tl.program_id(axis=0)
       pid_m = tl.program_id(axis=1)
       pid_n = tl.program_id(axis=2)

       offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
       offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
       offs_k = tl.arange(0, BLOCK_SIZE_K)

       a_ptrs = a_ptr + pid_b * M * K + offs_m[:, None] * K + offs_k[None, :]
       b_ptrs = b_ptr + pid_b * K * N + offs_k[:, None] * N + offs_n[None, :]

       accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
       for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
           a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
           b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
           accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
           a_ptrs += BLOCK_SIZE_K
           b_ptrs += BLOCK_SIZE_K * N

       c_ptrs = c_ptr + pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
       tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

---

### triton.language.dot_scaled

```python
dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=triton.language.float32, _semantic=None)
```

## triton.language.dot_scaled


**`dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, lhs_k_pack=True, rhs_k_pack=True, out_dtype=triton.language.float32)`**

    Returns the matrix product of two blocks in microscaling format.

    Parameters
    ----------
    lhs : tensor
        The first tensor to be multiplied. 2D tensor representing fp4, fp8 or bf16
        elements. Fp4 elements are packed into uint8 inputs with the first element
        in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.
    lhs_scale : tensor or None
        Scale factor for lhs tensor. Shape should be `[M, K//group_size]` when
        lhs is `[M, K]`, where group_size is 32 if scales type are `e8m0`.
        e8m0 type represented as an uint8 tensor, or None.
    lhs_format : str
        Format of the lhs tensor. Available formats: `e2m1`, `e4m3`,
        `e5m2`, `bf16`, `fp16`.
    rhs : tensor
        The second tensor to be multiplied. 2D tensor representing fp4, fp8 or
        bf16 elements. Fp4 elements are packed into uint8 inputs with the first
        element in lower bits. Fp8 are stored as uint8 or the corresponding fp8
        type.
    rhs_scale : tensor or None
        Scale factor for rhs tensor. Shape should be `[N, K//group_size]` where
        rhs is `[K, N]`. Important: Do NOT transpose rhs_scale. e8m0 type
        represented as an uint8 tensor, or None.
    rhs_format : str
        Format of the rhs tensor. Available formats: `e2m1`, `e4m3`,
        `e5m2`, `bf16`, `fp16`.
    acc : tensor, optional
        The accumulator tensor. If not None, the result is added to this tensor.
    fast_math : bool, optional
        Whether to use fast math approximations (default False).
    lhs_k_pack : bool, optional
        If false, the lhs tensor is packed into uint8 along M dimension
        (default True).
    rhs_k_pack : bool, optional
        If false, the rhs tensor is packed into uint8 along N dimension
        (default True).
    out_dtype : dtype, optional
        The output data type. Currently only `float32` is supported
        (default `triton.language.float32`).

    Returns
    -------
    out : tensor
        The matrix product result tensor with dtype specified by `out_dtype`.

    Notes
    -----
    lhs and rhs use microscaling formats described in the OCP Microscaling
    Formats MX v1.0 specification:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

    Software emulation enables targeting hardware architectures without native
    microscaling operation support. For such cases, microscaled lhs/rhs are
    upcasted to `bf16` element type beforehand for dot computation, with one
    exception: for AMD CDNA3 specifically, if one of the inputs is of `fp16`
    element type, the other input is also upcasted to `fp16` element type
    instead. This behavior is experimental and may be subject to change in the
    future.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def mx_dot_kernel(lhs_ptr, lhs_scale_ptr, rhs_ptr, rhs_scale_ptr,
                       out_ptr, M, N, K, BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
         pid_m = tl.program_id(0)
         pid_n = tl.program_id(1)

         offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
         offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
         offs_k = tl.arange(0, BLOCK_K)

         lhs = tl.load(lhs_ptr + offs_m[:, None] * K + offs_k[None, :])
         lhs_scale = tl.load(lhs_scale_ptr + offs_m[:, None] * (K // 32))
         rhs = tl.load(rhs_ptr + offs_k[:, None] * N + offs_n[None, :])
         rhs_scale = tl.load(rhs_scale_ptr + offs_n[:, None] * (K // 32))

         out = tl.dot_scaled(lhs, lhs_scale, "e4m3",
                            rhs, rhs_scale, "e4m3")
         tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out)
```

---

### triton.language.dtype

```python
dtype(name)
```

## class dtype

Data type descriptor for Triton tensors.

The `dtype` class represents scalar data types used in Triton GPU kernels. It encapsulates type information including bitwidth, signedness, and floating-point characteristics.

### Parameters
name : str
    The name of the data type. Must be one of the supported type strings:
    `'int1'`, `'int8'`, `'int16'`, `'int32'`, `'int64'`,
    `'uint8'`, `'uint16'`, `'uint32'`, `'uint64'`,
    `'fp8e4b15'`, `'fp8e4nv'`, `'fp8e4b8'`, `'fp8e5'`, `'fp8e5b16'`,
    `'fp16'`, `'bf16'`, `'fp32'`, `'fp64'`, or `'void'`.

### Attributes
name : str
    The type name string.
primitive_bitwidth : int
    The bitwidth of the primitive type.
itemsize : int
    The size in bytes (`primitive_bitwidth // 8`).
int_signedness : SIGNEDNESS
    Signedness for integer types (`SIGNED` or `UNSIGNED`).
int_bitwidth : int
    Bitwidth for integer types.
fp_mantissa_width : int
    Mantissa width for floating-point types.
exponent_bias : int
    Exponent bias for floating-point types.

### Methods
is_fp8()
    Returns `True` if this is an fp8 type.
is_fp16(), is_bf16(), is_fp32(), is_fp64()
    Returns `True` if this matches the specific floating-point type.
is_int8(), is_int16(), is_int32(), is_int64()
    Returns `True` if this matches the specific signed integer type.
is_uint8(), is_uint16(), is_uint32(), is_uint64()
    Returns `True` if this matches the specific unsigned integer type.
is_int()
    Returns `True` if this is any integer type (signed or unsigned).
is_floating()
    Returns `True` if this is any floating-point type.
is_standard_floating()
    Returns `True` if this is a standard floating-point type (fp16, bf16, fp32, fp64).
is_bool()
    Returns `True` if this is `int1` (boolean type).
kind()
    Returns the type kind (`KIND.BOOLEAN`, `KIND.INTEGRAL`, or `KIND.FLOATING`).
get_int_max_value()
    Returns the maximum representable integer value for this type.
get_int_min_value()
    Returns the minimum representable integer value for this type.
to_ir(builder)
    Converts this dtype to an MLIR type using the given builder.

### Notes
Predefined dtype instances are available as module-level constants:
`tl.int1`, `tl.int8`, `tl.int16`, `tl.int32`, `tl.int64`,
`tl.uint8`, `tl.uint16`, `tl.uint32`, `tl.uint64`,
`tl.float8e5`, `tl.float8e5b16`, `tl.float8e4nv`, `tl.float8e4b8`,
`tl.float8e4b15`, `tl.float16`, `tl.bfloat16`, `tl.float32`, `tl.float64`.

The `dtype` class is typically not instantiated directly by user code. Instead, use the predefined constants or pass dtype strings to functions that accept dtype parameters.

### Examples
```python
 import triton.language as tl

 # Using predefined dtype constants
 dtype = tl.int32
 print(dtype.name)              # 'int32'
 print(dtype.primitive_bitwidth)  # 32
 print(dtype.itemsize)          # 4
 print(dtype.is_int())          # True
 print(dtype.is_floating())     # False

 # Creating dtype from string
 dtype = tl.dtype('fp16')
 print(dtype.is_fp16())         # True
 print(dtype.fp_mantissa_width) # 10
 print(dtype.exponent_bias)     # 15

 # Type inspection in kernel code
 @triton.jit
 def kernel(x_ptr, dtype: tl.constexpr):
     if dtype.is_floating():
         # Handle floating-point types
         pass
     elif dtype.is_int():
         # Handle integer types
         pass
```

---

### triton.language.erf

```python
erf(x, _semantic=None)
```

Computes the element-wise error function of the input tensor.

The error function is defined as:

.. math::

    \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt

### Parameters
x : tensor
    The input tensor. Must have floating-point dtype (`fp32` or `fp64`).

### Returns
tensor
    A tensor of the same shape and dtype as `x` containing the error function
    values.

### Notes
Only `fp32` and `fp64` dtypes are supported. This function can also be called
as a member function on :py`tensor`, as `x.erf()` instead of
`tl.erf(x)`.

### Examples
```python
 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.erf(x)
     tl.store(y_ptr + offsets, y)

 # Can also be called as a member function
 y = x.erf()
```

---

### triton.language.exp

```python
exp(x, _semantic=None)
```

## triton.language.exp

**`exp(x, _semantic=None)`**

   Computes the element-wise natural exponential (e^x) of the input tensor.

   Parameters
   ----------
   x : tl.tensor
       The input tensor. Must be of floating-point type (`fp32` or `fp64`).

   Returns
   -------
   tl.tensor
       A tensor of the same shape as `x` containing the exponential values.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.exp()` instead of `tl.exp(x)`.

   The exponential is computed as e^x where e is Euler's number (approximately
   2.71828). Only floating-point types `fp32` and `fp64` are supported.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def exp_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offs)
       y = tl.exp(x)
       tl.store(y_ptr + offs, y)
```

---

### triton.language.exp2

```python
exp2(x, _semantic=None)
```

## exp2


**`exp2(x, _semantic=None)`**

   Computes the element-wise exponential (base 2) of `x`.

   Parameters
   ----------
   x : Block
       The input tensor. Must have floating-point dtype (`fp32` or `fp64`).

   Returns
   -------
   Block
       A tensor of the same shape as `x` containing `2^x` for each element.

   Notes
   -----
   This function is only supported for floating-point types (`fp32`, `fp64`).

   This function can also be called as a member function on :py`tensor`,
   as `x.exp2()` instead of `exp2(x)`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(X, Y, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(X + offsets)
       y = tl.exp2(x)  # Compute 2^x element-wise
       tl.store(Y + offsets, y)
```

---

### triton.language.expand_dims

```python
expand_dims(input, axis, _semantic=None)
```

## expand_dims

Expand the shape of a tensor by inserting new length-1 dimensions.

### Parameters

input : tl.tensor
    The input tensor to expand.

axis : int or Sequence[int]
    The axis or axes along which to insert new length-1 dimensions. Axis
    indices are with respect to the resulting tensor, so
    `result.shape[axis]` will be 1 for each specified axis.

### Returns

tl.tensor
    A tensor with the same data as `input` but with additional dimensions
    of size 1 inserted at the specified axes.

### Notes

This function can also be called as a member function on :py`tensor`,
as `x.expand_dims(...)` instead of `expand_dims(x, ...)`.

Duplicate axes will raise a `ValueError`. Negative axis indices are
supported and are normalized with respect to the resulting tensor's rank.

### Examples

```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     # Load a 1D tensor of shape [BLOCK_SIZE]
     x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))

     # Expand to 2D with shape [1, BLOCK_SIZE]
     y = tl.expand_dims(x, axis=0)

     # Expand to 3D with shape [1, 1, BLOCK_SIZE]
     z = tl.expand_dims(x, axis=(0, 1))

     # Store the expanded tensor
     tl.store(y_ptr, y)
```

---

### triton.language.fdiv

```python
fdiv(x, y, ieee_rounding=False, _semantic=None)
```

## triton.language.fdiv


**`fdiv(x, y, ieee_rounding=False)`**

   Computes the element-wise fast division of `x` and `y`.

   Parameters
   ----------
   x : tl.tensor
       The numerator input values.
   y : tl.tensor
       The denominator input values.
   ieee_rounding : bool, optional
       If true, use IEEE rounding mode. Default is false (fast division).

   Returns
   -------
   tl.tensor
       The element-wise division result of `x / y`.

   Notes
   -----
   Fast division (`ieee_rounding=False`) may have different rounding
   behavior than IEEE-754 division but is typically faster on GPU hardware.
   Set `ieee_rounding=True` for IEEE-754 compliant division.

   Examples
   --------
```python
    @triton.jit
    def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, BLOCK))
        y = tl.load(y_ptr + tl.arange(0, BLOCK))
        out = tl.fdiv(x, y)
        tl.store(out_ptr + tl.arange(0, BLOCK), out)
```

---

### triton.language.flip

```python
flip(x, dim=None)
```

**`triton.language.flip(x, dim=None)`**

    Flips a tensor along a specified dimension.

    Parameters
    ----------
    x : tl.block
        The input tensor to flip.
    dim : int, optional
        The dimension along which to flip. Defaults to the last dimension.
        Negative indices are supported.

    Returns
    -------
    tl.block
        The flipped tensor.

    Notes
    -----
    The size of the dimension being flipped must be a power of 2.
    This function can also be called as a member function on :py`tl.block`,
    as `x.flip(dim)` instead of `tl.flip(x, dim)`.

    Examples
    --------
```python
     import triton.language as tl

     # Flip along the last dimension (size must be power of 2)
     x = tl.arange(0, 8)
     y = tl.flip(x)  # Equivalent to x.flip()

     # Flip along a specific dimension
     z = tl.reshape(x, (2, 4))
     w = tl.flip(z, dim=1)
```

---

### triton.language.float16

## triton.language.float16

IEEE 754 binary16 floating-point data type.

### Notes
`float16` represents the half-precision floating-point format. It uses 16 bits
(1 sign bit, 5 exponent bits, 10 mantissa bits). This dtype is optimized for
memory bandwidth and compute throughput on modern GPUs, particularly when using
Tensor Cores. It has reduced numerical range and precision compared to
`float32` and should be used where performance outweighs strict numerical
accuracy requirements.

### Examples
Use `float16` to reduce memory usage and increase throughput in mixed-precision
kernels:

```python
 import triton
 import triton.language as tl

 @triton.jit
 def matmul_fp16_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
     # Define pointers with float16 dtype
     a_ptrs = a_ptr + tl.arange(0, 128)[:, None] * K
     b_ptrs = b_ptr + tl.arange(0, 128)[None, :] * K
     # Load and compute in float16
     a = tl.load(a_ptrs).to(tl.float16)
     b = tl.load(b_ptrs).to(tl.float16)
     c = tl.dot(a, b)
     tl.store(c_ptr, c)
```

---

### triton.language.float32

.. py:data:: triton.language.float32

   32-bit floating point data type.

   Represents IEEE 754 single-precision floating-point numbers. This type is
   used to specify the element type of tensors within Triton kernels.

   Notes
   -----
   Corresponds to the CUDA `float` type. Use this constant when defining
   tensor types in operations such as `triton.language.full()`,
   `triton.language.zeros()`, or `triton.language.cast()`.

   Examples
   --------
   Create a tensor of zeros with 32-bit float precision:

```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(ptr, n):
       offsets = tl.arange(0, n)
       zeros = tl.zeros((n,), dtype=tl.float32)
       tl.store(ptr + offsets, zeros)
```

---

### triton.language.float64

.. py:data:: triton.language.float64

   64-bit floating point data type.

   Represents IEEE 754 double-precision floating point numbers within Triton.
   This symbol is used to specify data types for pointers, tensors, and explicit
   type conversions inside JIT-compiled kernels.

   Notes
   -----
   Hardware support for `float64` varies by GPU architecture. While supported
   on NVIDIA Volta (V100) and later architectures, double-precision throughput
   may be significantly limited on consumer-grade GPUs compared to `float32`.

   Examples
   --------
   Use `float64` to cast loaded values for high-precision computation:

```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, out_ptr, n):
       pid = tl.program_id(0)
       # Load default type and cast to float64
       val = tl.load(x_ptr + pid).to(tl.float64)
       res = val * 2.0
       tl.store(out_ptr + pid, res)
```

---

### triton.language.float8e4b15

## triton.language.float8e4b15

8-bit floating point type with 4 exponent bits and 3 mantissa bits (E4M3).

This data type instance represents the NVIDIA FP8 E4M3 format. It is designed
for deep learning workloads on supported GPU architectures, such as NVIDIA
Hopper (H100) and Ada Lovelace (L40). Compared to :obj:`triton.language.float8e5b16`,
`float8e4b15` provides higher precision near zero at the cost of reduced
dynamic range.

### Notes
Hardware support for this dtype requires compute capability 8.9 or 9.0+.
When used in matrix multiplication operations via :obj:`triton.language.dot`,
accumulation typically occurs in FP16 or FP32 depending on the target hardware
and compiler flags.

### Examples
Use `float8e4b15` to define tensor types in a kernel:

```python
 import triton
 import triton.language as tl

 @triton.jit
 def fp8_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     # Load data as float8e4b15
     x = tl.load(x_ptr, dtype=tl.float8e4b15)
     y = tl.load(y_ptr, dtype=tl.float8e4b15)

     # Perform computation (accumulation promoted to float32)
     out = tl.dot(x, y)

     # Store result
     tl.store(out_ptr, out)
```

---

### triton.language.float8e4b8

## triton.language.float8e4b8

8-bit floating point data type with 4 exponent bits and 3 mantissa bits.

### Notes
This data type represents the FP8 E4M3 format (1 sign bit, 4 exponent bits, 3 mantissa bits).
It is designed for deep learning workloads on NVIDIA Hopper and Ampere architectures.
Using this type reduces memory bandwidth usage and increases tensor core throughput
compared to 16-bit floating point types, at the cost of reduced dynamic range and precision.

### Examples
Use the dtype for explicit type casting or pointer type annotation within a Triton kernel:

```python
 import triton
 import triton.language as tl

 @triton.jit
 def quantize_kernel(x_ptr, out_ptr, n_elements):
     pid = tl.program_id(axis=0)
     x = tl.load(x_ptr + pid)
     # Cast 32-bit float to float8e4b8
     x_fp8 = x.to(tl.float8e4b8)
     tl.store(out_ptr + pid, x_fp8)
```

---

### triton.language.float8e4nv

.. py:data:: triton.language.float8e4nv

    NVIDIA FP8 (E4M3) data type instance.

    Represents an 8-bit floating point format with 4 exponent bits and 3
    mantissa bits, compliant with the NVIDIA Hopper FP8 specification.

    Notes
    -----
    This dtype corresponds to the `E4M3` format (1 sign bit, 4 exponent bits,
    3 mantissa bits). It supports finite values and NaN, but does not support
    infinities. Hardware execution requires NVIDIA GPUs with Compute
    Capability >= 9.0 (Hopper architecture). This type is commonly used as
    input operand for mixed-precision matrix multiplications via `tl.dot`.

    Examples
    --------
    Use this dtype to cast tensors or define pointer types in Triton kernels:

```python
     import triton
     import triton.language as tl

     @triton.jit
     def fp8_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         mask = offs < BLOCK_SIZE
         x = tl.load(x_ptr + offs, mask=mask)
         # Cast to FP8 E4M3
         x_fp8 = x.to(tl.float8e4nv)
         tl.store(out_ptr + offs, x_fp8, mask=mask)
```

---

### triton.language.float8e5

.. py:data:: triton.language.float8e5

   8-bit floating point format (E5M2).

   Represents the FP8 data type with 5 exponent bits and 2 mantissa bits.
   This format offers a wider dynamic range than `float8e4` (E4M3) with
   reduced precision. Primarily used for mixed-precision matrix
   multiplications on modern NVIDIA GPUs.

   Notes
   -----
   Hardware support requires NVIDIA compute capability >= 9.0 (Hopper) or
   8.9 (Ada Lovelace) for native operations. Using this type on unsupported
   hardware may result in emulation or compilation errors.

   Examples
   --------
   Specify the dtype in pointer operations or type casting:

```python
   import triton
   import triton.language as tl

   @triton.jit
   def fp8_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
       offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offs < n
       x = tl.load(x_ptr + offs, mask=mask).to(tl.float8e5)
       tl.store(y_ptr + offs, x, mask=mask)
```

---

### triton.language.float8e5b16

.. py:data:: triton.language.float8e5b16

   8-bit floating point type with 5 exponent bits and exponent bias of 16.

   Notes
   -----
   This data type represents a specific FP8 format consisting of 1 sign bit,
   5 exponent bits, and 2 mantissa bits (E5M2). The exponent bias is 16,
   distinguishing it from the standard E5M2 format (bias 15). It is primarily
   supported on NVIDIA Hopper (sm_90) architectures for mixed-precision
   workloads.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def fp8_kernel(ptr, SIZE: tl.constexpr):
        # Allocate a tensor using float8e5b16
        buffer = tl.zeros([SIZE], dtype=tl.float8e5b16)

        # Perform operations...
        # tl.store(ptr, buffer)
```

---

### triton.language.floor

```python
floor(x, _semantic=None)
```

Compute the element-wise floor of the input tensor.

For each element `x[i]`, returns the largest integer less than or equal to `x[i]`.

### Parameters
x : tl.tensor
    Input tensor. Must have floating point dtype (`fp32` or `fp64`).

### Returns
tl.tensor
    Tensor of the same shape as `x` containing the floor values.

### Notes
This function can also be called as a member function on :py`tensor`,
as `x.floor()` instead of `tl.floor(x)`.

Only floating point types are supported (`fp32`, `fp64`).

### Examples
```python
 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
     y = tl.floor(x)
     tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.fma

```python
fma(x, y, z, _semantic=None)
```

## triton.language.fma


**`fma(x, y, z, _semantic=None)`**

   Computes the element-wise fused multiply-add of three tensors: `x * y + z`.

   Parameters
   ----------
   x : Block
       The first input tensor (multiplicand).
   y : Block
       The second input tensor (multiplier).
   z : Block
       The third input tensor (addend).

   Returns
   -------
   Block
       A tensor containing the element-wise result of `x * y + z`.

   Notes
   -----
   Fused multiply-add (FMA) performs the multiplication and addition in a single
   operation with one rounding, which can provide better numerical accuracy and
   performance compared to separate multiply and add operations.

   All inputs are broadcast to a common compatible shape if necessary. The inputs
   must have compatible types for the operation.

   This function is only available inside `@triton.jit` decorated kernel
   functions.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)
       y = tl.load(y_ptr + offsets)
       z = tl.load(z_ptr + offsets)
       result = tl.fma(x, y, z)
       tl.store(out_ptr + offsets, result)
```

---

### triton.language.full

```python
full(shape, value, dtype, _semantic=None)
```

## triton.language.full


.. autofunction:: full

Create a tensor filled with a scalar value.

Returns a tensor of the specified shape and dtype, with all elements set to the given scalar value.

### Parameters
shape : tuple of ints
    Shape of the new tensor, e.g., `(8, 16)` or `(8,)`.
value : scalar
    A scalar value to fill the tensor with.
dtype : tl.dtype
    Data type of the new tensor, e.g., `tl.float16`, `tl.int32`.

### Returns
tensor
    A tensor of the specified shape and dtype filled with the scalar value.

### Notes
This function is analogous to `numpy.full()`. The shape must satisfy Triton's
block shape constraints (power-of-two sizes, maximum numel limited by
`TRITON_MAX_TENSOR_NUMEL`).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel():
     # Create a 4x4 tensor filled with 1.0 (float32)
     ones = tl.full((4, 4), 1.0, tl.float32)

     # Create a 1D tensor of 8 zeros (int32)
     zeros = tl.full((8,), 0, tl.int32)

     # Use in computation
     result = ones + zeros  # element-wise addition
```

---

### triton.language.gather

```python
gather(src, index, axis, _semantic=None)
```

Gather elements from a tensor along a given dimension using indices.

### Parameters
src : tensor
    The source tensor from which elements are gathered.
index : tensor
    The index tensor specifying which elements to gather. Indices are
    applied along the specified axis.
axis : int
    The dimension along which to gather elements. Must be in the range
    `[-src.ndim, src.ndim)`.

### Returns
tensor
    A tensor containing the gathered elements. The output shape matches
    the index shape along the gathered axis, with other dimensions
    preserved from the source tensor.

### Notes
This function can also be called as a member function on :py`tensor`,
as `x.gather(index, axis)` instead of `gather(x, index, axis)`.

The index tensor determines which slices along the specified axis are
selected from the source tensor. All other dimensions are broadcast
between the source and index tensors.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def gather_kernel(src_ptr, index_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
     pid = tl.program_id(0)
     offsets = tl.arange(0, BLOCK)
     mask = offsets < N

     # Load source data
     src = tl.load(src_ptr + pid * N + offsets, mask=mask)

     # Load indices (e.g., select specific columns)
     index = tl.load(index_ptr + pid * N + offsets, mask=mask)

     # Gather along axis 0 (rows)
     gathered = tl.gather(src, index, axis=0)

     tl.store(out_ptr + pid * N + offsets, gathered, mask=mask)
```

---

### triton.language.histogram

```python
histogram(input, num_bins, mask=None, _semantic=None, _generator=None)
```

Compute a histogram of the input tensor values.

The histogram bins have a width of 1 and start at 0, covering the range
`[0, num_bins)`.

### Parameters
input : tensor
    The input tensor containing values to histogram.
num_bins : int
    Number of histogram bins.
mask : tensor of `triton.int1`, optional
    If `mask[idx]` is false, exclude `input[idx]` from the histogram.

### Returns
tensor
    A 1D tensor of length `num_bins` containing the histogram counts.

### Notes
This function can also be called as a member function on :py`tensor`,
as `x.histogram(...)` instead of `histogram(x, ...)`.

Input values outside the range `[0, num_bins)` are ignored.

### Examples
```python
 @triton.jit
 def histogram_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
     input = tl.load(input_ptr + offsets, mask=mask)
     hist = tl.histogram(input, num_bins=16)
     tl.store(output_ptr + tl.arange(0, 16), hist)
```

---

### triton.language.inline_asm_elementwise

```python
inline_asm_elementwise(asm: 'str', constraints: 'str', args: 'Sequence', dtype: 'Union[dtype, Sequence[dtype]]', is_pure: 'bool', pack: 'int', _semantic=None)
```

## inline_asm_elementwise


Execute inline assembly over a tensor.

Essentially, this is a `map` operation where the function is inline assembly. The input tensors are implicitly broadcasted to the same shape.

### Parameters
asm : str
    Assembly code to run. Must match the target's assembly format (e.g., PTX for NVIDIA GPUs).
constraints : str
    Inline assembly constraints in `LLVM format <https://llvm.org/docs/LangRef.html#inline-asm-constraint-string>`_.
args : Sequence[tensor]
    Input tensors whose values are passed to the assembly block. These are implicitly broadcasted to the same shape.
dtype : dtype or Sequence[dtype]
    The element type(s) of the returned tensor(s). Can be a single dtype or a tuple of dtypes for multiple outputs.
is_pure : bool
    If True, the compiler assumes the assembly block has no side-effects, enabling additional optimizations.
pack : int
    The number of elements processed by one instance of inline assembly. Each invocation processes `pack` elements at a time.

### Returns
tensor or tuple[tensor, ...]
    One tensor or a tuple of tensors with the given dtypes. If `dtype` is a single type, returns a single tensor. If `dtype` is a sequence, returns a tuple of tensors.

### Notes
Each invocation of the inline assembly processes `pack` elements at a time. Exactly which set of inputs a block receives is unspecified. Input elements of size less than 4 bytes are packed into 4-byte registers.

This operation does not support empty `dtype` -- the inline assembly must return at least one tensor, even if you don't need it. You can work around this by returning a dummy tensor of arbitrary type; it shouldn't cost you anything if you don't use it.

### Examples
Example using `PTX assembly <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`_:

```python
 @triton.jit
 def kernel(A, B, C, D, BLOCK: tl.constexpr):
     a = tl.load(A + tl.arange(0, BLOCK))  # uint8 tensor
     b = tl.load(B + tl.arange(0, BLOCK))  # float32 tensor

     # For each (a,b) in zip(a,b), perform the following:
     # - Let ai be `a` converted to int32.
     # - Let af be `a` converted to float.
     # - Let m be the max of ai and b.
     # - Return ai and mi.
     # Do the above 4 elements at a time.
     (c, d) = tl.inline_asm_elementwise(
         asm="""
         {
             // Unpack `a` into `ai`.
             .reg .b8 tmp<4>;
             mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
             cvt.u32.u8 $0, tmp0;
             cvt.u32.u8 $1, tmp1;
             cvt.u32.u8 $2, tmp2;
             cvt.u32.u8 $3, tmp3;
         }
         // Convert `ai` to float.
         cvt.rn.f32.s32 $4, $0;
         cvt.rn.f32.s32 $5, $1;
         cvt.rn.f32.s32 $6, $2;
         cvt.rn.f32.s32 $7, $3;
         // Take max of `ai` and `b`.
         max.f32 $4, $4, $9;
         max.f32 $5, $5, $10;
         max.f32 $6, $6, $11;
         max.f32 $7, $7, $12;
         """,
         constraints=(
             # 8 output registers: $0=ai0, $1=ai1, $2=ai2, $3=ai3,
             #                     $4=m0,  $5=m1,  $6=m2,  $7=m3.
             "=r,=r,=r,=r,=r,=r,=r,=r,"
             # 5 input registers: $8=ai (packed), $9=b0, $10=b1, $11=b2, $12=b3.
             "r,r,r,r,r"),
         args=[a, b],
         dtype=(tl.int32, tl.float32),
         is_pure=True,
         pack=4,
     )
     tl.store(C + tl.arange(0, BLOCK), c)
     tl.store(D + tl.arange(0, BLOCK), d)
```

---

### triton.language.int1

.. py:data:: triton.language.int1

   1-bit signed integer data type.

   Notes
   -----
   This instance represents the 1-bit signed integer element type in Triton.
   It is used to specify the `dtype` argument in tensor creation and memory
   operations, such as `triton.language.zeros()` or `triton.language.store()`.
   Corresponds to `i1` in LLVM IR.

   Examples
   --------
   >>> import triton.language as tl
   >>> @triton.jit
   ... def kernel(ptr):
   ...     x = tl.zeros([1], dtype=tl.int1)
   ...     tl.store(ptr, x)

---

### triton.language.int16

16-bit signed integer data type.

This dtype represents a signed integer with a width of 16 bits. It is
suitable for operations where memory bandwidth is critical and the
dynamic range of values fits within `[-32768, 32767]`.

### Examples
Use `tl.int16` to specify data types in kernel operations:

```python
 import triton.language as tl

 @triton.jit
 def kernel(ptr, n):
     pid = tl.program_id(0)
     # Create a 16-bit integer constant
     value = tl.full((1,), 100, dtype=tl.int16)
     # Store as int16
     tl.store(ptr + pid, value)
```

---

### triton.language.int32

.. py:data:: triton.language.int32

    32-bit signed integer data type.

    Specifies the signed 32-bit integer element type for tensors and pointers
    in Triton kernels. This dtype is compatible with standard integer arithmetic
    operations provided by `triton.language`.

    Notes
    -----
    Maps to `i32` in MLIR. For unsigned 32-bit integers, use
    :data:`triton.language.uint32`.

    Examples
    --------
    Initialize a tensor with 32-bit integer precision:

```python
     import triton.language as tl

     @triton.jit
     def kernel(ptr, SIZE: tl.constexpr):
         offsets = tl.arange(0, SIZE)
         mask = offsets < SIZE
         vals = tl.full((SIZE,), 1, dtype=tl.int32)
         tl.store(ptr + offsets, vals, mask=mask)
```

---

### triton.language.int64

64-bit signed integer data type.

This type represents a signed 64-bit integer within Triton kernels. It corresponds to LLVM IR `i64` and is used to specify tensor element types or perform explicit casts.

### Notes
Native hardware support for 64-bit integers varies by GPU architecture. On architectures lacking native `int64` support (e.g., NVIDIA Volta), operations may be emulated via 32-bit instructions, which can impact performance. For optimal performance on supported hardware (e.g., NVIDIA Ampere+), use `tl.int64` directly.

### Examples
Use `tl.int64` to specify tensor element types or perform casts within a kernel:

```python
 import triton
 import triton.language as tl

 @triton.jit
 def int64_kernel(x_ptr, y_ptr, SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offset = pid + tl.arange(0, SIZE)
     mask = offset < SIZE

     # Load data and cast to int64
     x = tl.load(x_ptr + offset, mask=mask).to(tl.int64)

     # Perform int64 arithmetic
     y = x + 1

     # Store result
     tl.store(y_ptr + offset, y, mask=mask)

 # Launch kernel
 # triton.run(int64_kernel, ...)
```

---

### triton.language.int8

## int8

8-bit signed integer data type.

This object represents the 8-bit signed integer type in Triton. It is used to
specify data types for tensor operations, pointer casting, and kernel
arguments. It corresponds to CUDA's `int8` and NumPy's `numpy.int8`.

### Notes
The representable range for `tl.int8` is `[-128, 127]`. Arithmetic
operations follow standard two's complement wrapping semantics on the GPU.

### Examples
Specify `tl.int8` as the `dtype` argument in tensor creation or type
conversion operations:

```python
 import triton
 import triton.language as tl

 @triton.jit
 def quantize_kernel(x_ptr, out_ptr, n):
     offsets = tl.arange(0, n)
     x = tl.load(x_ptr + offsets)
     # Convert to int8
     q = (x * 10.0).to(tl.int8)
     tl.store(out_ptr + offsets, q)

 # Create a constant tensor with int8 dtype
 const = tl.full((128,), -1, dtype=tl.int8)
```

---

### triton.language.interleave

```python
interleave(a, b)
```

Interleaves the values of two tensors along their last dimension.

The two input tensors must have the same shape. The resulting tensor contains
elements from `a` and `b` alternating along the last axis.

### Parameters
a : tl.tensor
    The first input tensor.
b : tl.tensor
    The second input tensor. Must have the same shape as `a`.

### Returns
tl.tensor
    A tensor containing interleaved values. The last dimension size is twice
    the size of the input tensors' last dimension.

### Notes
This operation is equivalent to `tl.join(a, b).reshape(a.shape[:-1] + [2 * a.shape[-1]])`.

### Examples
```python
 import triton.language as tl

 # Interleave two 1D tensors
 a = tl.arange(0, 4)  # [0, 1, 2, 3]
 b = tl.arange(4, 8)  # [4, 5, 6, 7]
 c = tl.interleave(a, b)  # [0, 4, 1, 5, 2, 6, 3, 7]
```

---

### triton.language.join

```python
join(a, b, _semantic=None)
```

## triton.language.join


**`join(a, b, _semantic=None)`**

    Join the given tensors in a new, minor dimension.

    Parameters
    ----------
    a : tensor
        The first input tensor.
    b : tensor
        The second input tensor.

    Returns
    -------
    tensor
        A tensor with the same shape as the inputs plus a new minor dimension
        of size 2.

    Notes
    -----
    The two inputs are broadcasted to be the same shape before joining.

    If you want to join more than two elements, you can use multiple calls to
    this function. This reflects the constraint in Triton that tensors must
    have power-of-two sizes.

    `join()` is the inverse of `split()`.

    Examples
    --------
```python
     import triton.language as tl

     @triton.jit
     def kernel(Ap, Bp, Cp, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
         offs_m = tl.arange(0, BLOCK_M)
         offs_n = tl.arange(0, BLOCK_N)
         a = tl.load(Ap + offs_m[:, None] * N + offs_n[None, :])
         b = tl.load(Bp + offs_m[:, None] * N + offs_n[None, :])
         # Join two (BLOCK_M, BLOCK_N) tensors into (BLOCK_M, BLOCK_N, 2)
         c = tl.join(a, b)
         tl.store(Cp + offs_m[:, None, None] * N * 2 + offs_n[None, :, None] * 2 + tl.arange(0, 2)[None, None, :], c)
```

---

### triton.language.load

```python
load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False, _semantic=None)
```

## load

Load data from memory at location defined by pointer.

### Parameters
pointer : triton.PointerType or block of triton.PointerType
    Pointer to the data to be loaded.
mask : block of triton.int1, optional
    If `mask[idx]` is false, do not load the data at address `pointer[idx]`.
    Must be `None` with block pointers.
other : block, optional
    If `mask[idx]` is false, return `other[idx]`.
boundary_check : tuple of ints, optional
    Tuple of integers indicating the dimensions which should do the boundary check.
padding_option : str, optional
    Should be one of {"", "zero", "nan"}. The padding value to use while out of bounds.
    "" means an undefined value.
cache_modifier : str, optional
    Changes cache option in NVIDIA PTX. Should be one of {"", ".ca", ".cg", ".cv"},
    where ".ca" stands for cache at all levels, ".cg" stands for cache at global
    level (cache in L2 and below, not L1), and ".cv" means don't cache and fetch
    again. See `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_
    for more details.
eviction_policy : str, optional
    Changes eviction policy in NVIDIA PTX.
volatile : bool, optional
    Changes volatile option in NVIDIA PTX.

### Returns
tensor
    A tensor of data whose values are loaded from memory.

### Notes
The behavior depends on the type of `pointer`:

1. Single element pointer: loads a scalar. `mask` and `other` must be scalars,
   `other` is implicitly typecast to `pointer.dtype.element_ty`, and
   `boundary_check` and `padding_option` must be empty.

2. N-dimensional tensor of pointers: loads an N-dimensional tensor. `mask` and
   `other` are implicitly broadcast to `pointer.shape`, `other` is implicitly
   typecast to `pointer.dtype.element_ty`, and `boundary_check` and
   `padding_option` must be empty.

3. Block pointer (from `make_block_ptr`): loads a tensor. `mask` and `other`
   must be `None`, and `boundary_check` and `padding_option` can be specified
   to control out-of-bound access behavior.

### Examples
Load a single element:

```python
 @triton.jit
 def kernel(ptr):
     value = tl.load(ptr)

```
Load with masking:

```python
 @triton.jit
 def kernel(ptr, mask, other):
     value = tl.load(ptr, mask=mask, other=other)

```
Load using block pointer:

```python
 @triton.jit
 def kernel(base_ptr, shape, strides):
     block_ptr = tl.make_block_ptr(
         base_ptr,
         shape=shape,
         strides=strides,
         offsets=(0, 0),
         block_shape=(16, 16),
         order=(0, 1)
     )
     value = tl.load(block_ptr)
```

---

### triton.language.load_tensor_descriptor

```python
load_tensor_descriptor(desc: 'tensor_descriptor_base', offsets: 'Sequence[constexpr | tensor]', _semantic=None) -> 'tensor'
```

## load_tensor_descriptor


**`load_tensor_descriptor(desc, offsets)`**

   Load a block of data from a tensor descriptor.

   Parameters
   ----------
   desc : tensor_descriptor_base
      The tensor descriptor to load from. Created by :py`make_tensor_descriptor()`.
   offsets : Sequence[constexpr | tensor]
      The offsets specifying the starting position for each dimension of the block load.
      Must have the same length as the tensor rank.

   Returns
   -------
   tensor
      A tensor containing the loaded data with shape matching the descriptor's block shape.

   Notes
   -----
   This function uses the Tensor Memory Accelerator (TMA) on NVIDIA GPUs with TMA support.
   Loads are performed in units of the descriptor's block shape. Values outside of the tensor
   bounds will be filled with zeros.

   Offsets must be multiples of 16 bytes for proper alignment.

   See Also
   --------
   make_tensor_descriptor : Create a tensor descriptor
   store_tensor_descriptor : Store data to a tensor descriptor

   Examples
   --------
```python
   @triton.jit
   def load_kernel(desc, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
       moffset = tl.program_id(0) * M_BLOCK
       noffset = tl.program_id(1) * N_BLOCK
       data = tl.load_tensor_descriptor(desc, [moffset, noffset])
       # data has shape [M_BLOCK, N_BLOCK]
```

---

### triton.language.log

```python
log(x, _semantic=None)
```

## triton.language.log

**`log(x, _semantic=None)`**

   Computes the element-wise natural logarithm of `x`.

   Parameters
   ----------
   x : tl.tensor
       The input tensor. Must have floating-point dtype (`fp32` or `fp64`).

   Returns
   -------
   tl.tensor
       A tensor containing the natural logarithm of each element in `x`.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.log()` instead of `tl.log(x)`.

   The natural logarithm is the logarithm to the base `e` (Euler's number).
   Only floating-point dtypes (`fp32`, `fp64`) are supported.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def log_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.log(x)
        tl.store(y_ptr + offsets, y, mask=mask)

    # Call as a free function
    y = tl.log(x)

    # Call as a member function
    y = x.log()
```

---

### triton.language.log2

```python
log2(x, _semantic=None)
```

log2(x, _semantic=None)

Computes the element-wise base-2 logarithm of the input tensor.

### Parameters
x : tl.tensor
    Input tensor. Must have floating-point dtype (`fp32` or `fp64`).

### Returns
tl.tensor
    Tensor containing the base-2 logarithm of each element in `x`.

### Notes
This function is only supported for floating-point types (`fp32`, `fp64`).
The result is undefined for negative or zero input values.

This function can also be called as a member function on :py`tensor`,
as `x.log2()` instead of `tl.log2(x)`.

### Examples
```python
 @triton.jit
 def log2_kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
     pid = tl.program_id(axis=0)
     block_start = pid * BLOCK
     offsets = block_start + tl.arange(0, BLOCK)
     mask = offsets < n_elements
     x = tl.load(in_ptr + offsets, mask=mask)
     y = tl.log2(x)
     tl.store(out_ptr + offsets, y, mask=mask)
```

---

### triton.language.make_block_ptr

```python
make_block_ptr(base: 'tensor', shape, strides, offsets, block_shape, order, _semantic=None)
```

## make_block_ptr

**`triton.language.make_block_ptr(base, shape, strides, offsets, block_shape, order)`**

   Returns a pointer to a block in a parent tensor.

   Parameters
   ----------
   base : tensor
       The base pointer to the parent tensor.
   shape : tuple of ints
       The shape of the parent tensor.
   strides : tuple of ints
       The strides of the parent tensor.
   offsets : tuple of ints
       The offsets to the block.
   block_shape : tuple of ints
       The shape of the block.
   order : tuple of ints
       The order of the original data format (e.g., row-major or column-major).

   Returns
   -------
   tensor_descriptor
       A pointer to the specified block in the parent tensor.

   Notes
   -----
   Block pointers enable efficient memory access patterns for tiled operations.
   They are commonly used with `triton.language.load()` and `triton.language.store()`
   when the `pointer` argument is a block pointer.

   Use `triton.language.advance()` to move the block pointer to different positions.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
       block_ptr = tl.make_block_ptr(
           base=ptr,
           shape=(M, N),
           strides=(N, 1),
           offsets=(0, 0),
           block_shape=(BLOCK_M, BLOCK_N),
           order=(1, 0)
       )
       block = tl.load(block_ptr)
       # ... process block ...
```

---

### triton.language.make_tensor_descriptor

```python
make_tensor_descriptor(base: 'tensor', shape: 'List[tensor]', strides: 'List[tensor]', block_shape: 'List[constexpr]', padding_option='zero', _semantic=None) -> 'tensor_descriptor'
```

## make_tensor_descriptor


.. autofunction:: make_tensor_descriptor

Create a tensor descriptor object for efficient global memory access.

### Parameters
base : tensor
    The base pointer of the tensor. Must be 16-byte aligned.
shape : list of tensor
    A list of non-negative integers representing the tensor shape.
strides : list of tensor
    A list of tensor strides. Leading dimensions must be multiples of 16-byte
    strides and the last dimension must be contiguous.
block_shape : list of constexpr
    The shape of block to be loaded/stored from global memory.
padding_option : str, optional
    Padding option for out-of-bounds access. Default is `"zero"`.

### Returns
tensor_descriptor
    A descriptor object representing the tensor in global memory.

### Notes
On NVIDIA GPUs with TMA (Tensor Memory Accelerator) support, this will result
in a TMA descriptor object and loads and stores from the descriptor will be
backed by the TMA hardware.

Currently only 2-5 dimensional tensors are supported.

TMA descriptors require a global memory allocation. Use
`triton.set_allocator()` to configure the allocator appropriately.

### Examples
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

### triton.language.map_elementwise

```python
map_elementwise(scalar_fn: 'Callable[..., Tuple[tensor, ...]]', *args: 'tensor', pack=1, _semantic=None, _generator=None)
```

## map_elementwise


**`map_elementwise(scalar_fn, *args, pack=1)`**

   Map a scalar function over input tensors elementwise.

   Applies `scalar_fn` to each element of the input tensors after broadcasting
   them to a common shape. Unlike `where()`, this allows control flow that
   only evaluates one branch per element, which can be more efficient for
   multi-branch functions where branches have different costs.

   Parameters
   ----------
   scalar_fn : Callable[..., Tuple[tensor, ...]]
       A JIT-compiled function (decorated with `@triton.jit`) that operates
       on scalar values and returns one or more tensors. This function is
       called once per element (or per `pack` elements) of the input tensors.
   *args : tensor
       Input tensors to map over. All tensors are implicitly broadcasted to
       the same shape before the function is applied.
   pack : int, optional
       The number of elements to be processed by one function call. Default is 1.
       Must be >= 1.

   Returns
   -------
   tensor or tuple of tensor
       One tensor if `scalar_fn` returns a single tensor, or a tuple of
       tensors if `scalar_fn` returns multiple values. The output shape
       matches the broadcasted input shape.

   Notes
   -----
   This function enables per-element control flow that avoids evaluating
   inactive branches. With `where()`, both branches are always computed
   regardless of the condition. With `map_elementwise` and Python `if`
   statements inside `scalar_fn`, only the taken branch is executed.

   The `pack` parameter allows processing multiple elements per function
   call, which can improve performance by reducing call overhead. All
   elements in a pack must have compatible types.

   Examples
   --------
   Define a scalar function with conditional logic:

```python
    import triton
    import triton.language as tl

    @triton.jit
    def selu_scalar(x, alpha):
        if x > 0:
            return x
        else:
            return alpha * (tl.exp(x) - 1)

    @triton.jit
    def selu(x, alpha):
        return tl.map_elementwise(selu_scalar, x, alpha)

Use with multiple return values:

.. code-block:: python

    @triton.jit
    def abs_and_sign(x):
        return tl.abs(x), tl.where(x > 0, 1, -1)

    @triton.jit
    def decompose(x):
        return tl.map_elementwise(abs_and_sign, x)

    # Returns a tuple of two tensors
```

---

### triton.language.max

```python
max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
```

## triton.language.max

**`max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)`**

   Returns the maximum of all elements in the input tensor along the provided axis.

   Parameters
   ----------
   input : tensor
       The input values.
   axis : int, optional
       The dimension along which the reduction should be done. If None, reduce all dimensions.
   return_indices : bool, optional
       If True, return index corresponding to the maximum value. Default is False.
   return_indices_tie_break_left : bool, optional
       If True, in case of a tie (i.e., multiple elements have the same maximum value), return the left-most index for values that aren't NaN. Default is True.
   keep_dims : bool, optional
       If True, keep the reduced dimensions with length 1. Default is False.

   Returns
   -------
   tensor or tuple of tensors
       If `return_indices` is False, returns a tensor containing the maximum values. If `return_indices` is True, returns a tuple of (max_values, indices).

   Notes
   -----
   The reduction operation is associative and commutative.

   Input tensors with bfloat16 dtype are promoted to float32. For integer or floating-point dtypes with bitwidth less than 32, the input is promoted to int32 or float32 respectively before reduction.

   This function can also be called as a member function on :py`tensor`, as `x.max(...)` instead of `max(x, ...)`.

   Examples
   --------
```python
   import triton.language as tl

   # Find maximum along axis 0
   x = tl.arange(0, 16).reshape(4, 4)
   m = tl.max(x, axis=0)

   # Keep reduced dimensions
   m_keep = tl.max(x, axis=0, keep_dims=True)

   # Get maximum value and index
   m_val, m_idx = tl.max(x, axis=1, return_indices=True)

   # Control tie-breaking behavior
   m_val, m_idx = tl.max(x, axis=1, return_indices=True, 
                         return_indices_tie_break_left=False)
```

---

### triton.language.max_constancy

```python
max_constancy(input, values, _semantic=None)
```

## triton.language.max_constancy


.. autofunction:: max_constancy

Let the compiler know that groups of values in `input` are constant.

### Parameters
input : tensor
    The input tensor.
values : list of constexpr[int]
    Compile-time constant integers specifying the group sizes. Each group of
    `d` values in `input` should all be equal, for each `d`
    in `values`.
_semantic : optional
    Internal parameter used by the Triton compiler. Do not set manually.

### Returns
tensor
    The input tensor, annotated with constancy information for the compiler.

### Notes
This is a compiler hint that helps the Triton compiler optimize code by
exploiting known constant patterns in the input data. The hint asserts that
consecutive groups of values in the input are constant (all equal within each
group).

For example, if `values` is `[4]`, then each group of 4 values in
`input` should all be equal, such as `[0, 0, 0, 0, 1, 1, 1, 1]`.

All elements in `values` must be `constexpr[int]` types.

### Examples
```python
 @triton.jit
 def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
     offset = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offset)
     
     # Tell compiler that groups of 4 values are constant
     x = tl.max_constancy(x, [4])
     
     # ... rest of kernel
```

---

### triton.language.max_contiguous

```python
max_contiguous(input, values, _semantic=None)
```

## max_contiguous


.. autofunction:: max_contiguous

Let the compiler know that the first values in `input` are contiguous.

### Parameters
input : tensor
    The input tensor.
values : constexpr or list of constexpr
    Compile-time constant values indicating the number of contiguous elements.
    Each element must be a `constexpr[int]`.
_semantic : optional
    Internal parameter used by Triton compiler. Do not set manually.

### Returns
tensor
    The input tensor (unchanged). This function is a compiler hint and does not
    modify the tensor values.

### Notes
This is a compiler hint that helps the Triton compiler optimize memory access
patterns. It informs the compiler that the first N values in the input tensor
are stored contiguously in memory, which can enable more efficient load/store
operations and memory coalescing.

The `values` parameter must contain only `constexpr[int]` elements.
Passing non-constexpr or non-integer values will raise a `TypeError`.

This function is similar to :py`multiple_of()` and :py`max_constancy()`,
which provide other types of compiler hints about tensor properties.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, BLOCK: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     # Inform compiler that first 16 elements are contiguous
     x = tl.max_contiguous(x, tl.constexpr([16]))
     # ... rest of kernel

 @triton.jit
 def kernel_multi(x_ptr, BLOCK: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     # Multiple hints can be provided
     x = tl.max_contiguous(x, tl.constexpr([16, 32]))
     # ... rest of kernel

```
### See Also
multiple_of, max_constancy

---

### triton.language.maximum

```python
maximum(x, y, propagate_nan: 'constexpr' = <PROPAGATE_NAN.NONE: 0>, _semantic=None)
```

## triton.language.maximum


**`maximum(x, y, propagate_nan=PropagateNan.NONE)`**

   Computes the element-wise maximum of `x` and `y`.

   Parameters
   ----------
   x : Block
       The first input tensor.
   y : Block
       The second input tensor.
   propagate_nan : tl.PropagateNan, optional
       Whether to propagate NaN values. Default is `PropagateNan.NONE`.

   Returns
   -------
   Block
       A tensor containing the element-wise maximum values.

   Notes
   -----
   If either input is of type `bfloat16`, it is promoted to `float32`
   before the operation is performed, as hardware does not support FMAX for
   bfloat16.

   NaN propagation behavior is controlled by the `propagate_nan` parameter.
   See `tl.PropagateNan` for available options.

   This function can also be called as a member function on :py`tensor`,
   as `x.maximum(y)` instead of `maximum(x, y)`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
       y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
       out = tl.maximum(x, y)
       tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), out)

   # Using as member function
   @triton.jit
   def kernel_member(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
       y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
       out = x.maximum(y)
       tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), out)

.. seealso::
   :func:`triton.language.minimum`
   :class:`triton.language.PropagateNan`
```

---

### triton.language.min

```python
min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
```

## triton.language.min


**`min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)`**

   Returns the minimum of all elements in the input tensor along the provided axis.

   Parameters
   ----------
   input : tensor
       The input values.
   axis : int, optional
       The dimension along which the reduction should be done. If None, reduce all dimensions.
   return_indices : bool, optional
       If True, return index corresponding to the minimum value. Default is False.
   return_indices_tie_break_left : bool, optional
       If True, in case of a tie (i.e., multiple elements have the same minimum value), return the left-most index for values that aren't NaN. Default is True.
   keep_dims : bool, optional
       If True, keep the reduced dimensions with length 1. Default is False.

   Returns
   -------
   tensor or tuple of tensors
       If `return_indices` is False, returns a tensor containing the minimum values. If `return_indices` is True, returns a tuple of (min_values, min_indices).

   Notes
   -----
   The reduction operation is associative and commutative.

   This function promotes bfloat16 inputs to float32. For integer or floating-point dtypes with bitwidth less than 32, inputs are promoted to int32 or float32 respectively before reduction.

   This function can also be called as a member function on :py`tensor`, as `x.min(...)` instead of `min(x, ...)`.

   Examples
   --------
```python
   import triton.language as tl

   # Find minimum along axis 0
   x = tl.arange(0, 16).reshape(4, 4)
   min_val = tl.min(x, axis=0)

   # Find minimum across all dimensions
   min_val = tl.min(x, axis=None)

   # Get minimum values and indices
   min_val, min_idx = tl.min(x, axis=1, return_indices=True)

   # Keep reduced dimensions
   min_val = tl.min(x, axis=0, keep_dims=True)
```

---

### triton.language.minimum

```python
minimum(x, y, propagate_nan: 'constexpr' = <PROPAGATE_NAN.NONE: 0>, _semantic=None)
```

## triton.language.minimum


**`minimum(x, y, propagate_nan=PropagateNan.NONE)`**

   Computes the element-wise minimum of two tensors.

   Parameters
   ----------
   x : Block
       The first input tensor.
   y : Block
       The second input tensor.
   propagate_nan : tl.PropagateNan, optional
       Whether to propagate NaN values. Default is `PropagateNan.NONE`.
       Use `PropagateNan.ALL` to propagate NaNs from either input.

   Returns
   -------
   Block
       A tensor containing the element-wise minimum values. For each element,
       returns `x` if `x < y`, otherwise `y`. NaN handling depends on
       `propagate_nan`.

   Notes
   -----
   bfloat16 inputs are automatically promoted to float32 before comparison,
   as hardware does not support FMIN for bfloat16.

   When `propagate_nan=PropagateNan.NONE` (default), NaN values are treated
   as greater than any finite value. When `propagate_nan=PropagateNan.ALL`,
   if either input is NaN, the output is NaN.

   .. seealso:: `triton.language.PropagateNan`

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
       x = tl.load(x_ptr + tl.arange(0, BLOCK))
       y = tl.load(y_ptr + tl.arange(0, BLOCK))
       out = tl.minimum(x, y)
       tl.store(out_ptr + tl.arange(0, BLOCK), out)

   # With NaN propagation
   @triton.jit
   def kernel_nan(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
       x = tl.load(x_ptr + tl.arange(0, BLOCK))
       y = tl.load(y_ptr + tl.arange(0, BLOCK))
       out = tl.minimum(x, y, propagate_nan=tl.PropagateNan.ALL)
       tl.store(out_ptr + tl.arange(0, BLOCK), out)
```

---

### triton.language.mul

```python
mul(x, y, sanitize_overflow: 'constexpr' = True, _semantic=None)
```

## triton.language.mul


.. autofunction:: mul

Element-wise multiplication of two tensors or scalars.

### Parameters
x : tensor or scalar
    The first input operand.
y : tensor or scalar
    The second input operand.
sanitize_overflow : constexpr, optional
    If True (default), enables overflow sanitization during multiplication.

### Returns
tensor
    The element-wise product of `x` and `y`.

### Notes
This function can also be called as a member function on :py`tensor`,
as `x.mul(y)` instead of `tl.mul(x, y)`.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, z_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.load(y_ptr + offsets)
     z = tl.mul(x, y)
     tl.store(z_ptr + offsets, z)
```

---

### triton.language.multiple_of

```python
multiple_of(input, values, _semantic=None)
```

## triton.language.multiple_of


.. autofunction:: multiple_of

Let the compiler know that the values in `input` are all multiples of `values`.

This is a compiler hint that can help optimize memory access patterns by informing
the compiler about alignment properties of the input tensor values.

### Parameters
input : tensor
    The input tensor whose values are known to be multiples of `values`.
values : constexpr or list of constexpr
    One or more compile-time constant integer values. Each element must be
    `constexpr[int]`. The compiler will assume all values in `input` are
    multiples of these constants.

### Returns
tensor
    The input tensor, unchanged but with alignment metadata attached for
    compiler optimization.

### Notes
This function does not modify the actual values in the tensor. It only provides
metadata to the compiler that can be used for optimization passes such as
vectorization, alignment checks, or memory coalescing.

All elements in `values` must be compile-time constants (`constexpr`) with
integer values. Passing runtime values will raise a `TypeError`.

### Examples
```python
 @triton.jit
 def kernel(ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     # Tell compiler that offsets are multiples of 16
     offsets = tl.multiple_of(offsets, tl.constexpr(16))
     values = tl.load(ptr + offsets)
     # ... rest of kernel
```

---

### triton.language.num_programs

```python
num_programs(axis, _semantic=None)
```

## triton.language.num_programs


.. autofunction:: num_programs

Returns the number of program instances launched along the given axis.

### Parameters
axis : int
    The axis of the 3D launch grid. Must be 0, 1, or 2.

### Returns
int
    The number of programs launched along the specified axis.

### Notes
This function is typically used within JIT-compiled Triton kernels to query
the grid dimensions. It returns a compile-time constant when the axis is
known at compile time.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(
     output_ptr,
     M,
     N,
     BLOCK_SIZE_M: tl.constexpr,
     BLOCK_SIZE_N: tl.constexpr,
 ):
     # Get the number of program instances along axis 0
     num_programs_m = tl.num_programs(0)
     
     # Get the number of program instances along axis 1
     num_programs_n = tl.num_programs(1)
     
     # Compute total number of programs
     total_programs = num_programs_m * num_programs_n
     
     # Use program_id to determine which block this instance handles
     pid_m = tl.program_id(0)
     pid_n = tl.program_id(1)
```

---

### triton.language.pair_uniform_to_normal

```python
pair_uniform_to_normal(u1, u2)
```

**`triton.language.pair_uniform_to_normal(u1, u2)`**

   Apply the Box-Muller transform to convert pairs of uniform random variables
   into standard normal random variables.

   Parameters
   ----------
   u1 : tl.tensor
       First input tensor containing uniform random values in $[0, 1)$.
   u2 : tl.tensor
       Second input tensor containing uniform random values in $[0, 1)$.

   Returns
   -------
   tuple
       A tuple `(z0, z1)` of tensors containing samples from the standard
       normal distribution $\mathcal{N}(0, 1)$.

   Notes
   -----
   This function implements the Box-Muller transform. To avoid numerical
   instability when computing the logarithm, `u1` is clamped to a minimum
   value of $1.0e-7$.

   Examples
   --------
```python
   @triton.jit
   def kernel(...):
       u1 = tl.rand(seed, offset)
       u2 = tl.rand(seed, offset + 1)
       n1, n2 = tl.pair_uniform_to_normal(u1, u2)
```

---

### triton.language.permute

```python
permute(input, *dims, _semantic=None)
```

Permutes the dimensions of a tensor.

### Parameters
input : Block
    The input tensor.
dims : int or tuple of ints
    The desired ordering of dimensions. For example, `(2, 1, 0)` reverses
    the order of dims in a 3D tensor. Can be passed as a tuple or as
    individual parameters.

### Returns
tensor
    The permuted tensor with dimensions reordered according to `dims`.

### Notes
This function can be called as a free function or as a member function on
`tensor`, as `x.permute(...)` instead of `permute(x, ...)`.

`trans()` is equivalent to this function, except when `dims` is empty,
it tries to swap the last two axes.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
     # Load 3D tensor
     x = tl.load(x_ptr + tl.arange(0, M)[:, None, None] * N * K
                 + tl.arange(0, N)[None, :, None] * K
                 + tl.arange(0, K)[None, None, :])

     # Reverse dimension order: (M, N, K) -> (K, N, M)
     y = tl.permute(x, 2, 1, 0)

     # Equivalent using tuple
     y = tl.permute(x, (2, 1, 0))

     # Equivalent using member function
     y = x.permute(2, 1, 0)

     tl.store(y_ptr, y)
```

---

### triton.language.philox

```python
philox(seed, c0, c1, c2, c3, n_rounds: triton.language.core.constexpr = constexpr[10])
```

**`triton.language.philox(seed, c0, c1, c2, c3, n_rounds=10)`**

   Apply Philox pseudo-random number generator rounds.

   Parameters
   ----------
   seed : tensor
       The seed for generating random numbers.
   c0 : tensor
       First counter tensor. Determines the integer width (32 or 64 bits).
   c1 : tensor
       Second counter tensor.
   c2 : tensor
       Third counter tensor.
   c3 : tensor
       Fourth counter tensor.
   n_rounds : constexpr, optional
       Number of mixing rounds. Default is 10.

   Returns
   -------
   tuple of tensors
       A tuple of four tensors representing the updated random state.

   Notes
   -----
   The integer width (32 or 64 bits) is inferred from the dtype of `c0`.
   The `seed` is cast to `uint64` internally.
   This is a low-level primitive; consider using `randint` or `rand` for typical use cases.

   Examples
   --------
```python
   @triton.jit
   def kernel(...):
       offsets = tl.program_id(0) * BLOCK
       r0, r1, r2, r3 = tl.philox(seed, offsets, 0, 0, 0)
```

---

### triton.language.philox_impl

```python
philox_impl(c0, c1, c2, c3, k0, k1, n_rounds: triton.language.core.constexpr = constexpr[10])
```

**`triton.language.philox_impl(c0, c1, c2, c3, k0, k1, n_rounds: triton.language.core.constexpr = 10)`**

    Run `n_rounds` rounds of Philox for state (c0, c1, c2, c3) and key (k0, k1).

    Parameters
    ----------
    c0 : tl.tensor
        Counter state variable.
    c1 : tl.tensor
        Counter state variable.
    c2 : tl.tensor
        Counter state variable.
    c3 : tl.tensor
        Counter state variable.
    k0 : tl.tensor
        Key variable.
    k1 : tl.tensor
        Key variable.
    n_rounds : tl.constexpr, optional
        Number of rounds, default is 10.

    Returns
    -------
    c0 : tl.tensor
        Updated counter state.
    c1 : tl.tensor
        Updated counter state.
    c2 : tl.tensor
        Updated counter state.
    c3 : tl.tensor
        Updated counter state.

    Notes
    -----
    Supports `tl.uint32` and `tl.uint64` dtypes. Constants are selected
    based on input dtype.

    Examples
    --------
```python
     import triton.language as tl

     @triton.jit
     def kernel(...):
         # Initialize state and key
         c0 = tl.full((1,), 0, dtype=tl.uint32)
         c1 = tl.full((1,), 0, dtype=tl.uint32)
         c2 = tl.full((1,), 0, dtype=tl.uint32)
         c3 = tl.full((1,), 0, dtype=tl.uint32)
         k0 = tl.full((1,), 0, dtype=tl.uint32)
         k1 = tl.full((1,), 0, dtype=tl.uint32)
         # Run Philox
         c0, c1, c2, c3 = tl.philox_impl(c0, c1, c2, c3, k0, k1)
```

---

### triton.language.pi32_t

.. py:data:: triton.language.pi32_t

   Pointer type instance for 32-bit integers.

   Represents the type of a pointer to 32-bit integer data (:py:data:`tl.int32`) in GPU memory. This constant is used to annotate pointer types or perform type introspection within Triton kernels.

   .. rubric:: Notes

   Equivalent to `tl.pointer_type(tl.int32)`. Prefer using type inference or :py`tl.make_block_ptr()` for pointer construction in most cases.

   .. rubric:: Examples

```python
   import triton.language as tl

   # Check type of a pointer argument
   @triton.jit
   def kernel(x_ptr: tl.pi32_t, size: tl.int32):
       # x_ptr is guaranteed to be a pointer to i32
       pid = tl.program_id(0)
       val = tl.load(x_ptr + pid)
       tl.store(x_ptr + pid, val + 1)
```

---

### triton.language.pointer_type

```python
pointer_type(element_ty: 'dtype', address_space: 'int' = 1, const: 'bool' = False)
```

## pointer_type

**`triton.language.pointer_type(element_ty, address_space=1, const=False)`**

   Represents a pointer type in Triton IR.

   Pointer types are used to describe memory addresses in Triton kernels. They
   encode the element type, address space, and optional const qualifier.

   Parameters
   ----------
   element_ty : dtype
       The data type of the pointed-to element. Must be a :py`dtype`
       instance (e.g., :py:data:`tl.int32`, :py:data:`tl.float32`).
   address_space : int, optional
       The address space of the pointer. Default is 1 (global memory).
       Different address spaces correspond to different memory regions
       (e.g., global, shared, constant memory).
   const : bool, optional
       Whether the pointer points to constant data. Default is False.
       Const pointers cannot be used with :py`tl.store()` operations.

   Attributes
   ----------
   element_ty : dtype
       The element type of the pointer.
   address_space : int
       The address space number.
   const : bool
       Whether this is a const pointer.
   name : str
       String representation (e.g., `pointer<int32>` or
       `const_pointer<float32>`).

   Methods
   -------
   to_ir(builder)
       Convert to LLVM IR pointer type.
   is_ptr()
       Returns True (identifies as pointer type).
   is_const()
       Returns whether the pointer is const.
   mangle()
       Returns the mangled name for code generation.

   Notes
   -----
   Pointer types are fundamental for memory operations in Triton. The
   :py:data:`address_space` parameter typically follows GPU conventions where
   1 represents global memory. Const pointers provide type safety by preventing
   stores through the pointer.

   Pointer types can be compared for equality, which checks element type,
   address space, and const qualifier.

   Examples
   --------
   >>> import triton.language as tl
   >>> ptr_ty = tl.pointer_type(tl.int32)
   >>> print(ptr_ty)
   pointer<int32>
   >>> const_ptr_ty = tl.pointer_type(tl.float32, const=True)
   >>> print(const_ptr_ty)
   const_pointer<float32>
   >>> ptr_ty.is_ptr()
   True
   >>> const_ptr_ty.is_const()
   True

   See Also
   --------
   triton.language.dtype : Base data type class.
   triton.language.load : Load from pointer.
   triton.language.store : Store to pointer.

---

### triton.language.program_id

```python
program_id(axis, _semantic=None)
```

## program_id


**`program_id(axis)`**

   Returns the ID of the current program instance along the given axis.

   Parameters
   ----------
   axis : int
       The axis of the 3D launch grid. Must be 0, 1, or 2.

   Returns
   -------
   pid : tl.tensor
       The program ID along the specified axis. Scalar int32 value.

   Notes
   -----
   This function is typically used in SPMD (Single Program Multiple Data) kernels
   to determine which program instance is executing. The program ID can be used
   to compute memory offsets or select which data to process.

   The 3D launch grid corresponds to the grid dimensions specified when launching
   the kernel. Each axis represents one dimension of the grid.

   Examples
   --------
   >>> @triton.jit
   ... def kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
   ...     pid = tl.program_id(0)
   ...     offset = pid * BLOCK_SIZE
   ...     mask = offset + tl.arange(0, BLOCK_SIZE) < N
   ...     x = tl.load(x_ptr + offset, mask=mask)
   ...     tl.store(x_ptr + offset, x * 2.0, mask=mask)

---

### triton.language.rand

```python
rand(seed, offset, n_rounds: triton.language.core.constexpr = constexpr[10])
```

## triton.language.rand


.. jitfunction:: rand(seed, offset, n_rounds=10)

Generate random numbers from a uniform distribution.

Given a `seed` scalar and an `offset` block, returns a block
of random `float32` values sampled from the uniform distribution
$U(0, 1)$.

### Parameters
seed : int
    The seed for generating random numbers.
offset : tl.tensor
    The offsets to generate random numbers for. Typically used to ensure
    uniqueness across program instances.
n_rounds : tl.constexpr, optional
    Number of rounds for the Philox PRNG. Default is 10.

### Returns
tl.tensor
    Block of random values with dtype `float32`.

### Notes
Uses the Philox pseudo-random number generator. The output is numerically
stable and uniformly distributed in the half-open interval $[0, 1)$.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(X, seed):
     pid = tl.program_id(0)
     offset = pid + tl.arange(0, 128)
     random_vals = tl.rand(seed, offset)
     # Store or use random_vals...
```

---

### triton.language.rand4x

```python
rand4x(seed, offsets, n_rounds: triton.language.core.constexpr = constexpr[10])
```

**`triton.language.rand4x(seed, offsets, n_rounds=10)`**

   Given a `seed` scalar and an `offsets` block,
   returns 4 blocks of random `float32` in $U(0, 1)$.

   Parameters
   ----------
   seed
       The seed for generating random numbers.
   offsets
       The offsets to generate random numbers for.
   n_rounds : tl.constexpr, optional
       Number of rounds for Philox PRNG. Default is 10.

   Returns
   -------
   u1 : tl.block
       First block of random float32 values.
   u2 : tl.block
       Second block of random float32 values.
   u3 : tl.block
       Third block of random float32 values.
   u4 : tl.block
       Fourth block of random float32 values.

   Notes
   -----
   Uses the Philox pseudo-random number generator.
   This function is more efficient than calling `triton.language.rand()`
   four times.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(rng_seed, offsets):
       u1, u2, u3, u4 = tl.rand4x(rng_seed, offsets)
       # Use u1, u2, u3, u4
```

---

### triton.language.randint

```python
randint(seed, offset, n_rounds: triton.language.core.constexpr = constexpr[10])
```

**`triton.language.randint(seed, offset, n_rounds=10)`**

    Given a `seed` scalar and an `offset` block, returns a single
    block of random `int32` values using the Philox PRNG.

    Parameters
    ----------
    seed : tl.int32 or tl.int64
        The seed for generating random numbers. Must be a scalar.
    offset : tl.int32 or tl.int64
        The offsets to generate random numbers for. Must be a block.
    n_rounds : tl.constexpr, optional
        Number of Philox rounds. Default is 10.

    Returns
    -------
    ret : tl.int32
        Block of random integers.

    Notes
    -----
    This function uses the Philox pseudo-random number generator.
    If you need multiple streams of random numbers, using `randint4x()`
    is likely to be faster than calling `randint()` four times.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(X, stride, n_elements,
                BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(axis=0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         mask = offs < n_elements
         seed = 123456
         random_vals = tl.randint(seed, offs)
         tl.store(X + offs, random_vals, mask=mask)
```

---

### triton.language.randint4x

```python
randint4x(seed, offset, n_rounds: triton.language.core.constexpr = constexpr[10])
```

.. jit_function:: triton.language.randint4x(seed, offset, n_rounds=10)

   Given a `seed` scalar and an `offset` block, returns four
   blocks of random `int32`.

   This is the maximally efficient entry point to Triton's Philox pseudo-random
   number generator.

   Parameters
   ----------
   seed : tl.tensor
       The seed for generating random numbers.
   offset : tl.tensor
       The offsets to generate random numbers for. Supports both 32-bit and
       64-bit integers.
   n_rounds : tl.constexpr, optional
       Number of rounds for the Philox PRNG. Default is 10.

   Returns
   -------
   r0 : tl.tensor
       First block of random `int32`.
   r1 : tl.tensor
       Second block of random `int32`.
   r2 : tl.tensor
       Third block of random `int32`.
   r3 : tl.tensor
       Fourth block of random `int32`.

   Notes
   -----
   For generating multiple streams of random numbers, using `randint4x`
   is more efficient than calling `randint` four times.

   Examples
   --------
```python
    @triton.jit
    def kernel(X, seed, offset):
        r0, r1, r2, r3 = tl.randint4x(seed, offset)
        # use r0, r1, r2, r3
```

---

### triton.language.randn

```python
randn(seed, offset, n_rounds: triton.language.core.constexpr = constexpr[10])
```

**`triton.language.randn(seed, offset, n_rounds=10)`**

   Given a `seed` scalar and an `offset` block, returns a block of random `float32` in $\mathcal{N}(0, 1)$.

   Parameters
   ----------
   seed :
       The seed for generating random numbers.
   offset :
       The offsets to generate random numbers for.
   n_rounds : tl.constexpr, optional
       Number of rounds for the Philox PRNG. Default is 10.

   Returns
   -------
   out : tl.block
       A block of random `float32` values sampled from the standard normal distribution.

   Notes
   -----
   This function uses the Philox pseudo-random number generator combined with the Box-Muller transform to generate normally distributed random numbers.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(axis=0)
       offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       seed = 123
       random_noise = tl.randn(seed, offset)
       # Use random_noise in computations
       ...
```

---

### triton.language.randn4x

```python
randn4x(seed, offset, n_rounds: triton.language.core.constexpr = constexpr[10])
```

Given a `seed` scalar and an `offset` block, returns 4 blocks of random `float32` values sampled from a standard normal distribution $\mathcal{N}(0, 1)$.

### Parameters
seed : int
    The seed for generating random numbers.
offset : tl.tensor
    The offsets to generate random numbers for.
n_rounds : constexpr, optional
    Number of rounds for the Philox PRNG. Default is 10.

### Returns
n1 : tl.tensor
    First block of random numbers.
n2 : tl.tensor
    Second block of random numbers.
n3 : tl.tensor
    Third block of random numbers.
n4 : tl.tensor
    Fourth block of random numbers.

### Notes
This function uses the Box-Muller transform to convert uniform random numbers
into normally distributed random numbers. It is more efficient than calling
`randn` four times separately.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(X_ptr, stride, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     seed = 123456
     n1, n2, n3, n4 = tl.randn4x(seed, offset)
     # Use n1, n2, n3, n4 ...
```

---

### triton.language.range

```python
range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False)
```

## triton.language.range


**`range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False)`**

   Special iterator for creating loops in `@triton.jit` functions with compiler optimization hints.

   This class provides similar semantics to Python's built-in `range` but allows users to pass additional attributes to guide compiler optimizations such as pipelining, unrolling, and warp specialization.

   Parameters
   ----------
   arg1 : int or constexpr
       If `arg2` is None, this is the end value (start defaults to 0). Otherwise, this is the start value.
   arg2 : int or constexpr, optional
       The end value of the range. If None, `arg1` is treated as the end value.
   step : int or constexpr, optional
       The step value between iterations. Defaults to 1.
   num_stages : int, optional
       Pipeline the loop into this many stages, allowing `num_stages` iterations to be in flight simultaneously. This differs from passing `num_stages` as a kernel argument, which only pipelines loads feeding into `dot` operations.
   loop_unroll_factor : int, optional
       Tells the Triton IR loop unroller how many times to unroll the loop. Values less than 2 imply no unrolling.
   disallow_acc_multi_buffer : bool, optional
       If True, prevent the accumulator of dot operations in the loop from being multi-buffered. Default is False.
   flatten : bool, optional
       Automatically flatten the loop nest starting at this loop to create a single flattened loop. The compiler will attempt to pipeline the flattened loop to avoid stage stalling. Default is False.
   warp_specialize : bool, optional
       Enable automatic warp specialization on the loop. The compiler will partition memory, MMA, and vector operations into separate async partitions, increasing the total number of warps required. Only supported on Blackwell GPUs for simple matmul loops. Default is False.
   disable_licm : bool, optional
       Prevent the compiler from hoisting loop invariant code outside the loop. Useful to avoid creating long liveranges within a loop. Default is False.

   Returns
   -------
   range
       An iterator object for use in `for` loops within JIT-compiled functions.

   Raises
   ------
   RuntimeError
       If `__iter__` or `__next__` is called outside of a `@triton.jit` decorated function.

   Notes
   -----
   This iterator can only be used within functions decorated with `@triton.jit`. Attempting to iterate over it in regular Python code will raise a RuntimeError.

   Warp specialization is currently only supported on Blackwell GPUs and works best with simple matmul loops. Support for arbitrary loops may be expanded in future versions.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(...):
        # Basic usage similar to Python range
        for i in tl.range(10):
            ...

        # With pipelining
        for i in tl.range(10, num_stages=3):
            ...

        # With loop unrolling
        for i in tl.range(0, 100, step=1, loop_unroll_factor=4):
            ...

        # Multiple optimization hints
        for i in tl.range(0, N, num_stages=2, loop_unroll_factor=2, flatten=True):
            ...
```

---

### triton.language.ravel

```python
ravel(x, can_reorder=False)
```

## triton.language.ravel


**`ravel(x, can_reorder=False)`**

    Returns a contiguous flattened view of `x`.

    Parameters
    ----------
    x : Block
        The input tensor to flatten.
    can_reorder : bool, optional
        Whether the compiler is allowed to reorder elements during flattening.
        Default is `False`.

    Returns
    -------
    Block
        A 1D tensor containing all elements of `x` in contiguous memory order.

    Notes
    -----
    This function can also be called as a member function on :py`tensor`,
    as `x.ravel(...)` instead of `ravel(x, ...)`.

    The output tensor has shape `[x.numel]` where `numel` is the total number
    of elements in the input tensor.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
         # Load a 2D block
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         x = tl.reshape(x, [4, 4])  # Shape: [4, 4]

         # Flatten to 1D
         flat = tl.ravel(x)  # Shape: [16]

         # Store the flattened result
         tl.store(output_ptr + tl.arange(0, 16), flat)

     # Can also use member function syntax
     flat = x.ravel()  # Equivalent to tl.ravel(x)
```

---

### triton.language.reduce

```python
reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None)
```

## triton.language.reduce


**`reduce(input, axis, combine_fn, keep_dims=False)`**

    Applies the `combine_fn` to all elements in `input` tensors along the provided `axis`.

    Parameters
    ----------
    input : tensor or tuple of tensors
        The input tensor(s) to reduce.
    axis : int or None
        The dimension along which the reduction should be done. If `None`, reduce all dimensions.
    combine_fn : callable
        A function to combine two groups of scalar tensors. Must be marked with `@triton.jit`.
    keep_dims : bool, optional
        If `True`, keep the reduced dimensions with length 1. Default is `False`.

    Returns
    -------
    tensor or tuple of tensors
        The reduced tensor(s) after applying the combine function along the specified axis.

    Notes
    -----
    This function can also be called as a member function on :py`tensor`, as
    `x.reduce(...)` instead of `reduce(x, ...)`.

    The reduction operation should be associative and commutative for correct results.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def add_fn(a, b):
         return a + b

     @triton.jit
     def kernel(X, Y, BLOCK_SIZE: tl.constexpr):
         x = tl.load(X + tl.arange(0, BLOCK_SIZE))
         # Sum all elements
         result = tl.reduce(x, axis=0, combine_fn=add_fn)
         tl.store(Y, result)

     # Can also be called as a member function
     @triton.jit
     def kernel2(X, Y, BLOCK_SIZE: tl.constexpr):
         x = tl.load(X + tl.arange(0, BLOCK_SIZE))
         result = x.reduce(axis=0, combine_fn=add_fn)
         tl.store(Y, result)
```

---

### triton.language.reduce_or

```python
reduce_or(input, axis, keep_dims=False)
```

## reduce_or


**`reduce_or(input, axis, keep_dims=False)`**

   Computes the bitwise OR reduction of all elements in the input tensor along the specified axis.

   Parameters
   ----------
   input : Tensor
       The input tensor to reduce. Must be of integer type.
   axis : int
       The dimension along which the reduction should be performed. If None, reduces all dimensions.
   keep_dims : bool, optional
       If True, keeps the reduced dimensions with length 1. Default is False.

   Returns
   -------
   Tensor
       A tensor containing the bitwise OR reduction of the input along the specified axis.

   Notes
   -----
   This function only supports integer tensors. The reduction operation is associative and commutative.

   This function can also be called as a member function on :py`tensor`, as `x.reduce_or(...)` instead of `reduce_or(x, ...)`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
       # Reduce OR across all elements
       result = tl.reduce_or(x, axis=0)
       tl.store(out_ptr, result)

   # Can also be called as a tensor method
   result = x.reduce_or(axis=0)
```

---

### triton.language.reshape

```python
reshape(input, *shape, can_reorder=False, _semantic=None, _generator=None)
```

**`triton.language.reshape(input, *shape, can_reorder=False)`**

    Returns a tensor with the same number of elements as `input` but with the
    provided shape.

    Parameters
    ----------
    input : tl.tensor
        The input tensor to reshape.
    *shape : int or tuple of ints
        The new shape dimensions. Can be passed as a tuple or as individual
        parameters.
    can_reorder : bool, optional
        If True, allows the compiler to reorder elements during reshaping.
        Only set to True if element order does not matter (e.g., result is only
        used in reduction ops). Default is False.

    Returns
    -------
    tl.tensor
        A tensor with the same elements as `input` but with the new shape.

    Notes
    -----
    This function can also be called as a member function on `tensor`,
    as `x.reshape(...)` instead of `reshape(x, ...)`.

    The total number of elements must remain the same before and after reshaping.

    When `can_reorder=True`, this is equivalent to :py`view()`.

    Examples
    --------
```python
     # These are equivalent
     reshape(x, (32, 32))
     reshape(x, 32, 32)

     # Can also use member function syntax
     x.reshape(32, 32)

     # Allow element reordering for reduction ops
     reshape(x, 64, can_reorder=True)
```

---

### triton.language.rsqrt

```python
rsqrt(x, _semantic=None)
```

## triton.language.rsqrt


**`rsqrt(x, _semantic=None)`**

   Computes the element-wise inverse square root of `x`.

   Parameters
   ----------
   x : Block
       The input tensor. Must be a floating-point type.

   Returns
   -------
   Block
       A tensor containing `1 / sqrt(x)` for each element in `x`.
       The output has the same shape and dtype as the input.

   Notes
   -----
   This function is only supported for floating-point types (`fp32`, `fp64`).

   This function can also be called as a member function on :py`tensor`,
   as `x.rsqrt()` instead of `tl.rsqrt(x)`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
       y = tl.rsqrt(x)
       tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.sigmoid

```python
sigmoid(x)
```

## triton.language.sigmoid


**`sigmoid(x)`**

   Computes the element-wise sigmoid of `x`.

   The sigmoid function is defined as `1 / (1 + exp(-x))`.

   Parameters
   ----------
   x : Block
       The input tensor or scalar values.

   Returns
   -------
   Block
       A tensor containing the sigmoid of each element in `x`.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.sigmoid()` instead of `sigmoid(x)`.

   The sigmoid function maps input values to the range (0, 1) and is commonly
   used as an activation function in neural networks.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(axis=0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       output = tl.sigmoid(x)
       tl.store(output_ptr + offsets, output, mask=mask)
```

---

### triton.language.sin

```python
sin(x, _semantic=None)
```

## triton.language.sin


**`sin(x, _semantic=None)`**

   Computes the element-wise sine of `x`.

   Parameters
   ----------
   x : Block
       The input tensor. Must be of floating-point type (`fp32` or `fp64`).

   Returns
   -------
   Block
       A tensor containing the sine of each element in `x`.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.sin()` instead of `tl.sin(x)`.

   The sine function is computed in radians. Input values are expected to be
   in the range where sine is defined for floating-point types.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(X_ptr + offsets)
       y = tl.sin(x)
       tl.store(Y_ptr + offsets, y)

   # Or using the member function syntax:
   @triton.jit
   def kernel_member(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(X_ptr + offsets)
       y = x.sin()
       tl.store(Y_ptr + offsets, y)
```

---

### triton.language.slice

```python
slice(start, stop, step)
```

## slice


**`slice(start, stop, step)`**

    Represents a slice object for tensor indexing operations in Triton kernels.

    Parameters
    ----------
    start : int or constexpr
        The starting index of the slice.
    stop : int or constexpr
        The stopping index of the slice (exclusive).
    step : int or constexpr
        The step value for the slice.

    Attributes
    ----------
    start
        The starting index.
    stop
        The stopping index.
    step
        The step value.
    type : slice_type
        The slice type object.

    Notes
    -----
    This class mirrors Python's built-in :py`slice` but is designed for
    use within Triton JIT-compiled functions. It is primarily used internally
    for tensor indexing operations via :py`tensor.__getitem__()`.

    Examples
    --------
```python
     import triton.language as tl

     @triton.jit
     def kernel(...):
         # Slice objects are typically created implicitly during indexing
         # Example: tensor[0:10:2] creates a slice(0, 10, 2)
         pass

     # Direct instantiation (less common)
     s = tl.slice(0, 10, 2)
     print(s.start)  # 0
     print(s.stop)   # 10
     print(s.step)   # 2
```

---

### triton.language.softmax

```python
softmax(x, dim=None, keep_dims=False, ieee_rounding=False)
```

## softmax


**`softmax(x, dim=None, keep_dims=False, ieee_rounding=False)`**

   Computes the element-wise softmax of the input tensor along a specified dimension.

   The softmax function normalizes the input along the given dimension such that the
   output values are in the range (0, 1) and sum to 1 along that dimension.

   Parameters
   ----------
   x : Block
       The input tensor. Must be a floating-point type.
   dim : int, optional
       The dimension along which to compute softmax. If None, defaults to 0.
   keep_dims : bool, optional
       If True, the output tensor has the same shape as the input with the reduced
       dimension kept as size 1. If False (default), the reduced dimension is removed.
   ieee_rounding : bool, optional
       If True, use IEEE rounding mode for the final division. If False (default),
       use GPU native rounding which may be faster but less precise.

   Returns
   -------
   Block
       The softmax output tensor with the same dtype as the input.

   Notes
   -----
   The softmax is computed as:

   .. math::

      \text{softmax}(x)_i = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}

   The implementation subtracts the maximum value before exponentiation for numerical
   stability, preventing overflow when input values are large.

   This function can also be called as a member function on tensors:
   `x.softmax(dim, keep_dims, ieee_rounding)` instead of
   `softmax(x, dim, keep_dims, ieee_rounding)`.

   Examples
   --------
   >>> import triton
   >>> import triton.language as tl

   Compute softmax along the last dimension:

```python
   @triton.jit
   def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + pid * BLOCK_SIZE + offsets)
       y = tl.softmax(x, dim=0)
       tl.store(output_ptr + pid * BLOCK_SIZE + offsets, y)

Using the member function syntax:

.. code-block:: python

   @triton.jit
   def kernel(x_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + pid * BLOCK_SIZE + offsets)
       y = x.softmax(dim=0)
       tl.store(output_ptr + pid * BLOCK_SIZE + offsets, y)

Keep the reduced dimension:

.. code-block:: python

   @triton.jit
   def kernel(x_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
       pid_m = tl.program_id(0)
       pid_n = tl.program_id(1)
       offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
       offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
       x = tl.load(x_ptr + offsets_m[:, None] * N + offsets_n[None, :])
       y = tl.softmax(x, dim=1, keep_dims=True)
       tl.store(output_ptr + offsets_m[:, None] * N + offsets_n[None, :], y)
```

---

### triton.language.sort

```python
sort(x, dim: 'core.constexpr' = None, descending: 'core.constexpr' = constexpr[0])
```

## sort

Sorts a tensor along a specified dimension using bitonic sort.

### Parameters
x : Tensor
    The input tensor to be sorted.
dim : constexpr, optional
    The dimension along which to sort the tensor. If None, the tensor is
    sorted along the last dimension. Currently, only sorting along the last
    dimension is supported.
descending : constexpr, optional
    If True, sorts in descending order. If False (default), sorts in
    ascending order.

### Returns
Tensor
    A sorted tensor with the same shape as the input.

### Notes
This function uses a bitonic sort algorithm which requires the dimension
size to be a power of 2. Only the last/minor dimension is currently
supported. The input tensor must have a shape that is compatible with
bitonic sorting constraints.

### Examples
```python
 import triton.language as tl
 
 # Sort a 1D tensor in ascending order
 x = tl.arange(0, 16)
 sorted_x = tl.sort(x)
 
 # Sort in descending order
 sorted_x_desc = tl.sort(x, descending=True)
 
 # Sort along the last dimension of a 2D tensor
 x_2d = tl.arange(0, 32).reshape([2, 16])
 sorted_2d = tl.sort(x_2d, dim=1)
```

---

### triton.language.split

```python
split(a, _semantic=None, _generator=None) -> 'tuple[tensor, tensor]'
```

## split


**`split(a, _semantic=None, _generator=None)`**

   Split a tensor in two along its last dimension, which must have size 2.

   Parameters
   ----------
   a : tensor
       The tensor to split. The last dimension must have size 2.

   Returns
   -------
   tuple[tensor, tensor]
       A tuple of two tensors. Each output tensor has the same shape as the
       input except the last dimension is removed. If the input is 1D with
       shape (2,), returns two scalars.

   Notes
   -----
   `split` is the inverse of :py`join()`. Given two tensors of shape
   (4, 8), :py`join()` produces a tensor of shape (4, 8, 2), and
   :py`split()` recovers the original two tensors.

   Triton requires tensors to have power-of-two sizes. To split into more
   than two pieces, use multiple calls to this function (possibly combined
   with :py`reshape()`).

   This function can also be called as a member function on
   :py`tensor`, as `x.split()` instead of `split(x)`.

   Examples
   --------
   Split a 3D tensor along its last dimension:

```python
   @triton.jit
   def kernel(...):
       x = tl.full((4, 8, 2), 1.0, dtype=tl.float32)
       left, right = tl.split(x)
       # left.shape == (4, 8), right.shape == (4, 8)

Split a 1D tensor (returns scalars):

.. code-block:: python

   @triton.jit
   def kernel(...):
       x = tl.arange(0, 2).to(tl.float32)
       a, b = tl.split(x)
       # a and b are scalars

Using as a member function:

.. code-block:: python

   @triton.jit
   def kernel(...):
       x = tl.full((4, 8, 2), 1.0, dtype=tl.float32)
       left, right = x.split()
```

---

### triton.language.sqrt

```python
sqrt(x, _semantic=None)
```

## triton.language.sqrt


**`sqrt(x, _semantic=None)`**

   Computes the element-wise square root of `x`.

   Parameters
   ----------
   x : tl.tensor
       The input tensor. Must have floating-point dtype (`fp32` or `fp64`).

   Returns
   -------
   tl.tensor
       A tensor of the same shape as `x` containing the square root of each element.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.sqrt()` instead of `tl.sqrt(x)`.

   Only floating-point types (`fp32`, `fp64`) are supported.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def sqrt_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.sqrt(x)
       tl.store(y_ptr + offsets, y, mask=mask)
```

---

### triton.language.sqrt_rn

```python
sqrt_rn(x, _semantic=None)
```

## sqrt_rn

**`sqrt_rn(x, _semantic=None)`**

Computes the element-wise precise square root (rounding to nearest wrt the IEEE standard) of `x`.

### Parameters
x : Block
    The input values. Must be `fp32` dtype.

### Returns
tensor
    A tensor containing the precise square root of each element in `x`.

### Notes
This function is only supported for `fp32` data type. The result is rounded to nearest according to the IEEE standard, providing more accurate results than the default :py`sqrt()` operation.

This function can also be called as a member function on :py`tensor`,
as `x.sqrt_rn()` instead of `sqrt_rn(x)`.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
     y = tl.sqrt_rn(x)
     tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.squeeze

```python
squeeze(x, dim: 'core.constexpr')
```

## squeeze

Remove a dimension of size 1 from a tensor.

### Parameters

x : Tensor
    Input tensor.
dim : constexpr
    Dimension to squeeze. Must have size 1.

### Returns

Tensor
    Output tensor with the specified dimension removed.

### Notes

This operation requires that `x.shape[dim] == 1`. If the dimension
does not have size 1, compilation will fail with a static assertion error.

The `dim` parameter must be a compile-time constant (`constexpr`).

### Examples

```python
 import triton.language as tl

 # Remove dimension 1 from a tensor of shape (4, 1, 8)
 x = tl.zeros([4, 1, 8], dtype=tl.float32)
 y = tl.squeeze(x, 1)  # y has shape (4, 8)

 # Squeeze the last dimension
 z = tl.zeros([4, 8, 1], dtype=tl.float32)
 w = tl.squeeze(z, 2)  # w has shape (4, 8)
```

---

### triton.language.static_assert

```python
static_assert(cond, msg='', _semantic=None)
```

**`static_assert(cond, msg='', _semantic=None)`**

   Assert a condition at compile time.

   Parameters
   ----------
   cond : constexpr
       The condition to assert. Must be evaluable at compile time.
   msg : str, optional
       Error message to display if the assertion fails. Default is empty string.

   Returns
   -------
   None

   Notes
   -----
   This assertion is evaluated at compile time during kernel compilation, not at
   runtime. It does not require the `TRITON_DEBUG` environment variable to
   be set, unlike :py`device_assert()` which performs runtime assertions.

   If the condition evaluates to false at compile time, compilation will fail
   with the provided error message.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(BLOCK_SIZE: tl.constexpr):
        tl.static_assert(BLOCK_SIZE == 1024, "BLOCK_SIZE must be 1024")
        # Kernel code here

See Also
--------
:py:func:`device_assert`, :py:func:`static_print`
```

---

### triton.language.static_print

```python
static_print(*values, sep: 'str' = ' ', end: 'str' = '\n', file=None, flush=False, _semantic=None)
```

## static_print

Print values at compile time during kernel compilation.

### Parameters
*values : variadic
    Values to print at compile time.
sep : str, optional
    Separator between values (default is space).
end : str, optional
    String appended after all values (default is newline).
file : optional
    File-like object (unused in Triton).
flush : bool, optional
    Flush flag (unused in Triton).
_semantic : optional
    Internal semantic argument (auto-provided by Triton).

### Returns
None

### Notes
This function prints values during kernel compilation, not at runtime. It is
distinct from Python's builtin `print`, which maps to
:py`triton.language.device_print()` and executes on the GPU device at
runtime.

Use :py`static_print()` for debugging compile-time constants such as
:py`constexpr` values and :py:obj:`tl.constexpr` parameters.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(BLOCK_SIZE: tl.constexpr):
     tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
     # ... kernel code ...

 # Prints "BLOCK_SIZE=1024" during compilation
 kernel[(1,)](BLOCK_SIZE=1024)
```

---

### triton.language.static_range

```python
static_range(arg1, arg2=None, step=None)
```

## static_range

Iterator for compile-time loop unrolling in JIT-compiled Triton kernels.

### Parameters

arg1 : constexpr
    If `arg2` is None, this is the end value (start defaults to 0).
    If `arg2` is provided, this is the start value.
arg2 : constexpr, optional
    The end value. If None, `arg1` is used as end with start=0.
step : constexpr, optional
    The step value. Defaults to 1 if not provided.

### Returns

static_range
    An iterator object that can only be used within `@triton.jit` decorated
    functions.

### Notes

This is a special iterator used to implement similar semantics to Python's
`range` in the context of `triton.jit` functions. All arguments
must be `constexpr` values (compile-time constants). The compiler uses
this to aggressively unroll loops.

Attempting to call `__iter__` or `__next__` outside of a JIT
function will raise a RuntimeError.

### Examples

```python
 @triton.jit
 def kernel(...):
     # Loop from 0 to 9
     for i in tl.static_range(10):
         ...

     # Loop from 5 to 15
     for i in tl.static_range(5, 15):
         ...

     # Loop from 0 to 20 with step 2
     for i in tl.static_range(0, 20, 2):
         ...
```

---

### triton.language.store

```python
store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='', _semantic=None)
```

Store a tensor of data into memory locations defined by `pointer`.

### Parameters
pointer : triton.PointerType or block of triton.PointerType
    The memory location where the elements of `value` are stored. Can be a
    single element pointer, an N-dimensional tensor of pointers, or a block
    pointer defined by `make_block_ptr`.
value : Block
    The tensor of elements to be stored. Implicitly broadcast to
    `pointer.shape` and typecast to `pointer.dtype.element_ty`.
mask : Block of triton.int1, optional
    If `mask[idx]` is false, do not store `value[idx]` at `pointer[idx]`.
    Must be None when using block pointers.
boundary_check : tuple of ints, optional
    Tuple of integers indicating the dimensions which should perform boundary
    checks. Can only be specified when using block pointers.
cache_modifier : str, optional
    Changes cache option in NVIDIA PTX. Should be one of {"", ".wb", ".cg",
    ".cs", ".wt"}, where ".wb" stands for cache write-back all coherent levels,
    ".cg" stands for cache global, ".cs" stands for cache streaming, ".wt"
    stands for cache write-through. See `cache operator
    <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_
    for more details.
eviction_policy : str, optional
    Changes eviction policy in NVIDIA PTX. Should be one of {"", "evict_first",
    "evict_last"}.

### Notes
The behavior of `store` depends on the type of `pointer`:

(1) If `pointer` is a single element pointer, a scalar is stored. In this
    case, `mask` must also be scalar, and `boundary_check` must be empty.

(2) If `pointer` is an N-dimensional tensor of pointers, an N-dimensional
    block is stored. In this case, `mask` is implicitly broadcast to
    `pointer.shape`, and `boundary_check` must be empty.

(3) If `pointer` is a block pointer defined by `make_block_ptr`, a block
    of data is stored. In this case, `mask` must be None, and
    `boundary_check` can be specified to control the behavior of out-of-bound
    access.

This function can also be called as a member function on :py`tensor`,
as `x.store(...)` instead of `store(x, ...)`.

### Examples
```python
 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     # Create offsets for this program instance
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     
     # Load data from global memory
     x = tl.load(x_ptr + offsets)
     
     # Perform computation
     y = x * 2.0
     
     # Store result to global memory
     tl.store(y_ptr + offsets, y)

 @triton.jit
 def kernel_with_mask(x_ptr, y_ptr, mask_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     
     # Load mask to control which elements are stored
     mask = tl.load(mask_ptr + offsets)
     
     x = tl.load(x_ptr + offsets)
     y = x * 2.0
     
     # Only store where mask is true
     tl.store(y_ptr + offsets, y, mask=mask)

 @triton.jit
 def kernel_block_ptr(input_ptr, output_ptr, M, N,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
     # Create block pointers for 2D tensor access
     input_block = tl.make_block_ptr(
         base=input_ptr,
         shape=(M, N),
         strides=(N, 1),
         offsets=(0, 0),
         block_shape=(BLOCK_M, BLOCK_N),
         order=(1, 0)
     )
     
     output_block = tl.make_block_ptr(
         base=output_ptr,
         shape=(M, N),
         strides=(N, 1),
         offsets=(0, 0),
         block_shape=(BLOCK_M, BLOCK_N),
         order=(1, 0)
     )
     
     # Load block of data
     x = tl.load(input_block)
     
     # Store block of data with boundary checking
     tl.store(output_block, x, boundary_check=(0, 1))
```

---

### triton.language.store_tensor_descriptor

```python
store_tensor_descriptor(desc: 'tensor_descriptor_base', offsets: 'Sequence[constexpr | tensor]', value: 'tensor', _semantic=None) -> 'tensor'
```

Store a block of data to a tensor descriptor.

### Parameters
desc : tensor_descriptor_base
    The tensor descriptor object representing a region of global memory.
    Typically created by :py`triton.language.make_tensor_descriptor()`.
offsets : Sequence[constexpr | tensor]
    The starting offsets for each dimension of the tensor. Must be a
    sequence with length equal to the tensor's rank. Offsets must be
    multiples of 16 bytes for TMA compatibility.
value : tensor
    The tensor containing the data to store. Must match the block shape
    specified when creating the descriptor.
_semantic : optional
    Internal parameter for semantic handling. Do not set manually.

### Returns
tensor
    The stored value tensor (returned for SSA form compatibility).

### Notes
Tensor descriptors enable efficient memory operations through the Tensor
Memory Accelerator (TMA) on supported NVIDIA GPUs. Stores outside the
tensor bounds are silently ignored.

Offsets must be 16-byte aligned for correct TMA operation. The value
tensor shape must match the `block_shape` specified in
:py`triton.language.make_tensor_descriptor()`.

This function requires the kernel to be decorated with `@triton.jit`.

### Examples
```python
 @triton.jit
 def kernel(in_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
     # Create tensor descriptors for input and output
     in_desc = tl.make_tensor_descriptor(
         in_ptr,
         shape=[M, N],
         strides=[N, 1],
         block_shape=[BLOCK_M, BLOCK_N],
     )
     out_desc = tl.make_tensor_descriptor(
         out_ptr,
         shape=[M, N],
         strides=[N, 1],
         block_shape=[BLOCK_M, BLOCK_N],
     )

     # Compute program offsets
     m_offset = tl.program_id(0) * BLOCK_M
     n_offset = tl.program_id(1) * BLOCK_N

     # Load data using descriptor
     data = in_desc.load([m_offset, n_offset])

     # Process data
     result = tl.abs(data)

     # Store data using descriptor
     tl.store_tensor_descriptor(out_desc, [m_offset, n_offset], result)
```

---

### triton.language.sub

```python
sub(x, y, sanitize_overflow: 'constexpr' = True, _semantic=None)
```

## triton.language.sub


**`sub(x, y, sanitize_overflow=True)`**

    Computes the element-wise subtraction of `x - y`.

    Parameters
    ----------
    x : tensor or scalar
        The minuend (left operand).
    y : tensor or scalar
        The subtrahend (right operand).
    sanitize_overflow : constexpr, optional
        If True, sanitizes integer overflow. Default is True.

    Returns
    -------
    tensor or scalar
        The result of `x - y`.

    Notes
    -----
    This function can also be called as a member function on :py`tensor`,
    as `x.sub(y)` instead of `sub(x, y)`.

    Examples
    --------
```python
     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         y = tl.load(y_ptr + tl.arange(0, BLOCK))
         result = tl.sub(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK), result)
```

---

### triton.language.sum

```python
sum(input, axis=None, keep_dims=False, dtype: 'core.constexpr' = None)
```

**`sum(input, axis=None, keep_dims=False, dtype=None)`**

   Returns the sum of all elements in the input tensor along the provided axis.

   Parameters
   ----------
   input : tensor
       The input values.
   axis : int, optional
       The dimension along which the reduction should be done. If None, reduce all dimensions.
   keep_dims : bool, optional
       If True, keep the reduced dimensions with length 1.
   dtype : tl.dtype, optional
       The desired data type of the returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. If not specified, integer and bool dtypes are upcasted to `tl.int32` and float dtypes are upcasted to at least `tl.float32`.

   Returns
   -------
   tensor
       A tensor containing the sum of elements.

   Notes
   -----
   The reduction operation is associative and commutative.
   This function can also be called as a member function on tensor, as `x.sum(...)`.

   Examples
   --------
```python
   import triton.language as tl

   # Sum all elements
   result = tl.sum(input_tensor)

   # Sum along axis 0
   result = tl.sum(input_tensor, axis=0)
```

---

### triton.language.swizzle2d

```python
swizzle2d(i, j, size_i, size_j, size_g)
```

## triton.language.swizzle2d


**`swizzle2d(i, j, size_i, size_j, size_g)`**

   Transform row-major matrix indices to column-major indices per row group.

   Parameters
   ----------
   i : int
       Row index in the original row-major matrix.
   j : int
       Column index in the original row-major matrix.
   size_i : int
       Number of rows in the matrix.
   size_j : int
       Number of columns in the matrix.
   size_g : int
       Number of rows per group for the swizzling transformation.

   Returns
   -------
   new_i : int
       Transformed row index in the column-major layout.
   new_j : int
       Transformed column index in the column-major layout.

   Notes
   -----
   This function reorganizes memory access patterns to improve coalescing on GPU
   hardware. It partitions the matrix into groups of `size_g` rows and within
   each group transforms from row-major to column-major ordering.

   The transformation is useful for optimizing memory access patterns in GPU
   kernels, particularly when dealing with 2D tiling strategies.

   Examples
   --------
   >>> import triton.language as tl
   >>>
   >>> # Example: 4x4 matrix with group size 2
   >>> # Original row-major layout:
   >>> # [[0,  1,  2,  3 ],
   >>> #  [4,  5,  6,  7 ],
   >>> #  [8,  9,  10, 11],
   >>> #  [12, 13, 14, 15]]
   >>>
   >>> # Transformed column-major per group:
   >>> # [[0,  2,  4,  6 ],
   >>> #  [1,  3,  5,  7 ],
   >>> #  [8,  10, 12, 14],
   >>> #  [9,  11, 13, 15]]
   >>>
   >>> @triton.jit
   >>> def kernel(i_ptr, j_ptr, size_i, size_j, size_g):
   ...     i = tl.load(i_ptr)
   ...     j = tl.load(j_ptr)
   ...     new_i, new_j = tl.swizzle2d(i, j, size_i, size_j, size_g)
   ...     # Use new_i, new_j for memory access
   >>>

---

### triton.language.tensor

```python
tensor(handle, type: 'dtype')
```

## class triton.language.tensor

Represents an N-dimensional array of values or pointers.

The `tensor` is the fundamental data structure in Triton programs. Most
functions in :py`triton.language` operate on and return tensors.

### Parameters
handle : ir.value
    Internal IR handle (not typically passed by user code)
type : dtype
    Tensor type, which can be a scalar dtype or block_type

### Attributes
dtype : dtype
    The scalar data type of the tensor elements
shape : tuple of constexpr
    The shape of the tensor (empty tuple for scalars)
numel : constexpr
    Total number of elements in the tensor
type : dtype
    The full tensor type (may be block_type)
handle : ir.value
    Internal IR handle

### Notes
Tensors are typically created through Triton operations rather than direct
construction. User code should use functions like :py`triton.language.load()`,
:py`triton.language.arange()`, or :py`triton.language.full()` to create
tensors.

Most named member functions are duplicates of free functions in
:py`triton.language`. For example, `triton.language.sqrt(x)` is
equivalent to `x.sqrt()`.

The `tensor` class defines magic/dunder methods for operator overloading,
supporting expressions like `x + y`, `x << 2`, `x == y`, etc.

Member functions added via the `_tensor_member_fn` decorator include
operations for shape manipulation (`reshape`, `broadcast_to`, `trans`),
reductions (`sum`, `max`, `min`), atomic operations (`atomic_add`,
`atomic_cas`), and mathematical functions (`exp`, `log`, `sqrt`).

### Examples
```python
 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK + tl.arange(0, BLOCK)
     x = tl.load(x_ptr + offsets)  # x is a tensor
     y = tl.sqrt(x)                # Returns a tensor
     tl.store(y_ptr + offsets, y)

 @triton.jit
 def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
     pid_m = tl.program_id(0)
     pid_n = tl.program_id(1)
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
     a = tl.load(a_ptr + offs_m[:, None] * K + offs_n[None, :])  # 2D tensor
     b = tl.load(b_ptr + offs_m[:, None] * K + offs_n[None, :])  # 2D tensor
     c = tl.dot(a, b)                                           # Matrix product tensor
```

---

### triton.language.tensor_descriptor

```python
tensor_descriptor(handle, shape: 'List[tensor]', strides: 'List[tensor]', block_type: 'block_type')
```

**`tensor_descriptor(handle, shape, strides, block_type)`**

    A descriptor representing a tensor in global memory.

    Parameters
    ----------
    handle : ir.value
        IR handle for the descriptor.
    shape : list[tensor]
        List of tensors representing the global shape of the tensor.
    strides : list[tensor]
        List of tensors representing the strides of the tensor.
    block_type : block_type
        The block type for load/store operations.

    Notes
    -----
    This class is typically not instantiated directly by user code. Instead, use
    :py`triton.language.make_tensor_descriptor()` to create tensor descriptors.

    Tensor descriptors enable efficient memory access patterns on GPUs with TMA
    (Tensor Memory Accelerator) support, such as NVIDIA Hopper architecture.
    They allow loading and storing blocks of data with hardware-accelerated
    memory operations.

    Currently only 2-5 dimensional tensors are supported.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     import torch

     @triton.jit
     def kernel(in_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
         desc = tl.make_tensor_descriptor(
             in_ptr,
             shape=[M, N],
             strides=[N, 1],
             block_shape=[BLOCK_M, BLOCK_N],
         )

         m_offset = tl.program_id(0) * BLOCK_M
         n_offset = tl.program_id(1) * BLOCK_N

         data = desc.load([m_offset, n_offset])
         # ... process data ...
         desc.store([m_offset, n_offset], data)

     # Usage
     M, N = 256, 256
     x = torch.randn(M, N, device="cuda")
     BLOCK_M, BLOCK_N = 32, 32
     kernel[(M // BLOCK_M, N // BLOCK_N)](x, M, N, BLOCK_M, BLOCK_N)
```

---

### triton.language.to_tensor

```python
to_tensor(x, _semantic=None)
```

## to_tensor

Convert input values to a tensor.

### Parameters
x : scalar, constexpr, or tensor
    The input value to convert to a tensor. Can be a Python scalar,
    :py`constexpr`, or :py`tensor`.
_semantic : optional
    Internal parameter for semantic operations. Users should not provide
    this argument.

### Returns
tensor
    A tensor containing the input value. If the input is already a tensor,
    it is returned unchanged. If the input is a scalar or constexpr, it is
    converted to a 0-dimensional tensor.

### Notes
This function is primarily used internally by Triton operations. Most users
will not need to call it directly, as arithmetic operations and other builtin
functions automatically convert their arguments to tensors.

### Examples
```python
 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
     # Load creates a tensor from pointers
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     
     # Scalar constants are automatically converted to tensors
     # when used in operations (internally calls to_tensor)
     y = x + 1.0
     
     # Explicit conversion (rarely needed)
     z = tl.to_tensor(2.0)
     
     tl.store(y_ptr + tl.arange(0, BLOCK), y + z)
```

---

### triton.language.topk

```python
topk(x, k: 'core.constexpr', dim: 'core.constexpr' = None, descending: 'core.constexpr' = True)
```

## triton.language.topk


**`topk(x, k, dim=None, descending=True)`**

    Returns the `k` largest (or smallest) elements of the input tensor along the specified dimension.

    The elements are returned in sorted order (largest first when `descending=True`).

    Parameters
    ----------
    x : tensor
        The input tensor.
    k : constexpr
        The number of top elements to return. Must be a power of two.
    dim : constexpr, optional
        The dimension along which to find the top `k` elements. If None, uses the last dimension. Currently only the last dimension is supported.
    descending : constexpr, optional
        If True (default), returns `k` largest elements. If False, returns `k` smallest elements.

    Returns
    -------
    tensor
        A tensor containing the `k` largest (or smallest) elements along the specified dimension, in sorted order.

    Notes
    -----
    The value of `k` must be a power of two for the underlying bitonic sort algorithm to work correctly.

    Only the last (minor) dimension is currently supported for the `dim` parameter.

    Examples
    --------
    Get top 4 elements from a 1D tensor:

```python
     import triton.language as tl

     x = tl.arange(0, 16)
     top4 = tl.topk(x, 4)  # Returns [15, 14, 13, 12]

 Get bottom 4 elements (smallest):

 .. code-block:: python

     x = tl.arange(0, 16)
     bottom4 = tl.topk(x, 4, descending=False)  # Returns [0, 1, 2, 3]
```

---

### triton.language.trans

```python
trans(input: 'tensor', *dims, _semantic=None)
```

## triton.language.trans


**`trans(input, *dims, _semantic=None)`**

    Permutes the dimensions of a tensor.

    If the parameter `dims` is not specified, the function defaults to
    swapping the last two axes, thereby performing an (optionally batched)
    2D transpose.

    Parameters
    ----------
    input : tensor
        The input tensor to transpose.
    *dims : int or tuple of ints, optional
        The desired ordering of dimensions. For example, `(2, 1, 0)`
        reverses the order of dims in a 3D tensor. If not specified,
        defaults to swapping the last two axes.

    Returns
    -------
    tensor
        The transposed tensor with dimensions permuted according to `dims`.

    Notes
    -----
    `dims` can be passed as a tuple or as individual parameters:

```python
     # These are equivalent
     trans(x, (2, 1, 0))
     trans(x, 2, 1, 0)

 :py:func:`permute` is equivalent to this function, except it doesn't
 have the special case when no permutation is specified.

 This function can also be called as a member function on :py:class:`tensor`,
 as ``x.trans(...)`` instead of ``trans(x, ...)``.

 Examples
 --------
 Transpose a 2D tensor (swap last two axes by default):

 .. code-block:: python

     @triton.jit
     def kernel(X, Y, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
         offs_m = tl.arange(0, BLOCK_M)
         offs_n = tl.arange(0, BLOCK_N)
         x = tl.load(X + offs_m[:, None] * N + offs_n[None, :])
         y = tl.trans(x)  # Swaps the last two axes
         tl.store(Y + offs_n[:, None] * M + offs_m[None, :], y)

 Transpose a 3D tensor with explicit dimension ordering:

 .. code-block:: python

     @triton.jit
     def kernel(X, Y, D, M, N, BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
         offs_d = tl.arange(0, BLOCK_D)
         offs_m = tl.arange(0, BLOCK_M)
         offs_n = tl.arange(0, BLOCK_N)
         x = tl.load(X + offs_d[:, None, None] * M * N + 
                        offs_m[None, :, None] * N + 
                        offs_n[None, None, :])
         y = tl.trans(x, 2, 1, 0)  # Reverses dimension order
         tl.store(Y + offs_n[:, None, None] * M * D + 
                     offs_m[None, :, None] * D + 
                     offs_n[None, None, :], y)
```

---

### triton.language.tuple

```python
tuple(args: 'Sequence', type: 'Optional[tuple_type]' = None)
```

## class triton.language.tuple

Base class for tuple values that exist in the Triton IR.

### Parameters
args : Sequence
    Sequence of values to store in the tuple. Each value must be a Triton IR
    value (e.g., :py`tensor`, :py`constexpr`, or nested
    :py`tuple`).
type : tuple_type, optional
    Optional type specification for the tuple. If not provided, the type is
    inferred from the values in `args`.

### Attributes
values : list
    List of values stored in the tuple.
type : tuple_type
    The type of the tuple, including field names if applicable.

### Notes
The :py`tuple` class represents structured collections of values in
Triton kernels. Unlike Python tuples, these exist at compile-time in the
Triton IR and support:

- Indexing via :py`__getitem__()` with :py`constexpr` indices
- Attribute access if the tuple type has named fields
- Iteration over elements
- Arithmetic operations (addition, multiplication)
- Equality comparison

Tuples are commonly used to return multiple values from Triton functions or
to group related data. When a tuple type has named fields, elements can be
accessed via attribute notation (e.g., `t.field_name`).

### Examples
Creating and using tuples in a Triton kernel:

```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(...):
     # Create a tuple of tensors
     t = tl tuple([x, y, z])

     # Access elements by index
     first = t[0]
     second = t[1]

     # Iterate over tuple elements
     for val in t:
         ...

     # Return multiple values as a tuple
     return tl tuple([result1, result2])

 # Unpack returned tuple
 r1, r2 = kernel[...]

```
Named field access (when tuple type has fields defined):

```python
 @triton.jit
 def kernel(...):
     # Tuple with named fields
     point = tl tuple([x, y], type=tl tuple_type([tl.int32, tl.int32],
                                                  fields=["x", "y"]))
     # Access by field name
     px = point.x
     py = point.y

```
### See Also
tuple_type : Type specification for tuples with optional field names.
split : Split a tensor into a tuple of tensors.
join : Join tensors into a single tensor (inverse of split).

---

### triton.language.uint16

.. py:data:: triton.language.uint16

   Unsigned 16-bit integer data type.

   This dtype represents a 16-bit unsigned integer in Triton IR. It is
   equivalent to CUDA `uint16_t` and is used to specify element types
   for tensors or to cast values within GPU kernels.

   Examples
   --------
```python
   import triton.language as tl

   @triton.jit
   def kernel(ptr):
       val = tl.full((1,), 42, dtype=tl.uint16)
       tl.store(ptr, val)
```

---

### triton.language.uint32

.. py:data:: triton.language.uint32

   32-bit unsigned integer data type.

   Represents an unsigned 32-bit integer element type for use in Triton
   GPU kernels. This constant is an instance of `triton.language.dtype`.

   Notes
   -----
   Corresponds to `torch.uint32` in PyTorch and `numpy.uint32` in NumPy.
   Use this dtype to define tensor element types or pointer base types
   within kernel definitions.

   Examples
   --------
   Use `uint32` to specify data types in kernel operations:

```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(out_ptr):
        # Initialize a uint32 value
        val = tl.full((1,), 42, dtype=tl.uint32)
        tl.store(out_ptr, val)
```

---

### triton.language.uint64

64-bit unsigned integer data type.

Represents an unsigned 64-bit integer type for use in Triton GPU kernels.
This dtype is used to define tensor element types or cast scalars to 64-bit
unsigned precision.

### See Also
tl.int64 : 64-bit signed integer data type.
tl.uint32 : 32-bit unsigned integer data type.

### Notes
Corresponds to `u64` in CUDA PTX. Use when 64-bit address arithmetic or
large integer values are required. Throughput may be lower than 32-bit
integers on some architectures.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(ptr, n_elements: tl.constexpr):
     pid = tl.program_id(0)
     offset = pid * 32
     mask = offset + tl.arange(0, 32) < n_elements
     data = tl.load(ptr + offset, mask=mask)
     data_u64 = data.to(tl.uint64)
     tl.store(ptr + offset, data_u64, mask=mask)
```

---

### triton.language.uint8

.. py:data:: triton.language.uint8

   8-bit unsigned integer data type.

   Instance representing the unsigned 8-bit integer type in Triton. Used to
   specify data types for pointers, tensors, and scalars in JIT-compiled
   kernels.

   Notes
   -----
   This type maps to `uint8_t` in CUDA. It is distinct from
   :py:data:`triton.language.int8` (signed) and :py:data:`triton.language.float8`
   (floating point).

   Examples
   --------
   Define a pointer to uint8 data and perform load/store operations.

```python
   import triton
   import triton.language as tl

   @triton.jit
   def uint8_kernel(ptr):
       # Load single uint8 value
       val = tl.load(ptr)
       # Store single uint8 value
       tl.store(ptr, val + 1)

   # Pointer type construction
   uint8_ptr = tl.pointer_type(tl.uint8)
```

---

### triton.language.uint_to_uniform_float

```python
uint_to_uniform_float(x)
```

**`triton.language.uint_to_uniform_float(x)`**

   Numerically stable function to convert a random uint into a random float
   uniformly sampled in [0, 1).

   Parameters
   ----------
   x : tl.tensor
      Input tensor of integer type. Supported dtypes are `tl.uint32`,
      `tl.int32`, `tl.uint64`, or `tl.int64`.

   Returns
   -------
   tl.tensor
      Tensor of floating-point values uniformly sampled in the range [0, 1).

   Notes
   -----
   The function handles both 32-bit and 64-bit integers using specific scaling
   factors to ensure the result remains strictly less than 1.0. Typically used
   internally by random number generation functions like `rand` and
   `randn`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(ptr, seed, offset):
       rand_int = tl.randint(seed, offset)
       rand_float = tl.uint_to_uniform_float(rand_int)
       tl.store(ptr, rand_float)
```

---

### triton.language.umulhi

```python
umulhi(x, y, _semantic=None)
```

## umulhi


**`umulhi(x, y, _semantic=None)`**

   Computes the element-wise most significant N bits of the 2N-bit product of `x` and `y`.

   Parameters
   ----------
   x : Block
       The first input tensor. Must be of type `int32`, `int64`, `uint32`, or `uint64`.
   y : Block
       The second input tensor. Must be of type `int32`, `int64`, `uint32`, or `uint64`.

   Returns
   -------
   Block
       A tensor containing the most significant N bits of the 2N-bit product of `x` and `y`. The dtype matches the input dtype.

   Notes
   -----
   For N-bit inputs, the full product is 2N bits. This function returns the upper N bits of that product. This is useful for high-precision multiplication where the full 2N-bit result is needed but only the high bits are required.

   Supported dtypes: `int32`, `int64`, `uint32`, `uint64`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offs)
       y = tl.load(y_ptr + offs)
       # Compute high 32 bits of 64-bit product
       hi = tl.umulhi(x, y)
       tl.store(out_ptr + offs, hi)
```

---

### triton.language.unsqueeze

```python
unsqueeze(x, dim: 'core.constexpr')
```

## unsqueeze

Insert a new axis of size 1 at the specified dimension.

### Parameters
x : Block
    The input tensor.
dim : constexpr
    The dimension at which to insert the new axis. Must be a compile-time constant.

### Returns
Block
    A view of the input tensor with an additional dimension of size 1 inserted at `dim`.

### Notes
This operation is equivalent to inserting a dimension of size 1 into the shape tuple at position `dim`. The total number of elements remains unchanged.

### Examples
```python
 import triton.language as tl

 @triton.jit
 def kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.arange(0, BLOCK_SIZE)  # shape: (BLOCK_SIZE,)
     y = tl.unsqueeze(x, 0)        # shape: (1, BLOCK_SIZE)
     z = tl.unsqueeze(x, 1)        # shape: (BLOCK_SIZE, 1)
```

---

### triton.language.view

```python
view(input, *shape, _semantic=None)
```

## triton.language.view

**`view(input, *shape, _semantic=None)`**

    Returns a tensor with the same elements as `input` but a different shape.

    Parameters
    ----------
    input : tensor
        The input tensor to reshape.
    shape : int or tuple of ints
        The desired shape. Can be passed as a tuple or as individual parameters.

    Returns
    -------
    tensor
        A view of the input tensor with the new shape.

    Notes
    -----
    The order of the elements may not be preserved. This function is deprecated;
    use :py`triton.language.reshape()` with `can_reorder=True` instead.

    `shape` can be passed as a tuple or as individual parameters.

    This function can also be called as a member function on :py`tensor`,
    as `x.view(...)` instead of `view(x, ...)`.

    Examples
    --------
```python
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         # Reshape from 1D to 2D
         y = tl.view(x, (BLOCK_SIZE // 2, 2))
         tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)

     # These are equivalent
     view(x, (32, 32))
     view(x, 32, 32)
```

---

### triton.language.void

.. py:data:: triton.language.void

    The void data type.

    Represents the absence of a value. Primarily used in function signatures
    to indicate that a kernel or device function does not return a value.

    Notes
    -----
    This is an instance of `triton.language.dtype`. It cannot be used
    to allocate memory or perform arithmetic operations.

    Examples
    --------
```python
     import triton.language as tl

     # Reference the void type
     print(tl.void)
```

---

### triton.language.where

```python
where(condition, x, y, _semantic=None)
```

## triton.language.where


**`where(condition, x, y)`**

    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters
    ----------
    condition : tensor
        Where True, yield `x`, otherwise yield `y`. Must be a block of
        `triton.bool`.
    x : tensor
        Values selected at indices where `condition` is True.
    y : tensor
        Values selected at indices where `condition` is False. Must have the
        same data type as `x`.

    Returns
    -------
    out : tensor
        A tensor with elements from `x` where `condition` is True, and from
        `y` where `condition` is False. The shape is broadcasted from the
        inputs.

    Notes
    -----
    Both `x` and `y` are always evaluated regardless of the value of
    `condition`. This means that if either `x` or `y` involves memory
    operations (e.g., loads), those operations will execute even if their
    values are not selected.

    To avoid unintended memory operations, use the `mask` arguments in
    `triton.language.load()` and `triton.language.store()` instead.

    The shapes of `x` and `y` are both broadcast to the shape of
    `condition`. `x` and `y` must have the same data type.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offs)
         y = tl.load(y_ptr + offs)
         # Select x where x > 0, otherwise select y
         out = tl.where(x > 0, x, y)
         tl.store(out_ptr + offs, out)
```

---

### triton.language.xor_sum

```python
xor_sum(input, axis=None, keep_dims=False)
```

Compute the XOR sum of elements along a given axis.

### Parameters
input : tensor
    The input tensor. Must be of integer type.
axis : int, optional
    The dimension along which the reduction is performed. If `None`, reduce all dimensions.
keep_dims : bool, optional
    If `True`, the reduced axes are retained as dimensions of size 1.

### Returns
tensor
    The result of the XOR reduction.

### Notes
This function can also be called as a member function on :py`triton.language.tensor`,
as `x.xor_sum(...)` instead of `xor_sum(x, ...)`.

The reduction operation is associative and commutative.

### Examples
```python
 import triton.language as tl

 # Create a 1D tensor of integers
 x = tl.arange(0, 8)
 # Compute XOR sum across all elements
 result = tl.xor_sum(x)
```

---

### triton.language.zeros

```python
zeros(shape, dtype)
```

## triton.language.zeros

**`triton.language.zeros(shape, dtype)`**

   Returns a tensor filled with the scalar value 0.

   Parameters
   ----------
   shape : tuple of ints
       Shape of the new array, e.g., `(8, 16)` or `(8,)`.
   dtype : DType
       Data-type of the new array, e.g., `tl.float16`.

   Returns
   -------
   Tensor
       A tensor filled with zeros of the specified shape and dtype.

   Examples
   --------
```python
   import triton.language as tl

   # Create a 1D tensor of 8 zeros
   z1 = tl.zeros((8,), tl.int32)

   # Create a 2D tensor of 8x16 zeros
   z2 = tl.zeros((8, 16), tl.float16)
```

---

### triton.language.zeros_like

```python
zeros_like(input)
```

## zeros_like


**`zeros_like(input)`**

   Returns a tensor of zeros with the same shape and dtype as the input tensor.

   Parameters
   ----------
   input : Tensor
       Input tensor whose shape and dtype determine the output tensor properties.

   Returns
   -------
   Tensor
       Tensor filled with zeros, having the same shape and dtype as `input`.

   See Also
   --------
   zeros : Create a tensor filled with zeros given explicit shape and dtype.
   ones_like : Create a tensor of ones with the same shape and dtype as input.
   full : Create a tensor filled with a given scalar value.

   Examples
   --------
```python
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       # Load input tensor
       x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
       
       # Create zeros tensor with same shape as x
       zeros = tl.zeros_like(x)
       
       # Store result
       tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), zeros)
```

---
