# triton.experimental.gluon

Gluon core — experimental low-level JIT decorator and configuration.

*3 APIs documented.*

---

## triton.experimental.gluon

### triton.experimental.gluon.constexpr_function

```python
constexpr_function(fn)
```

## constexpr_function


.. autofunction:: constexpr_function

**`constexpr_function(fn)`**

   Wraps a Python function for compile-time evaluation on constexpr arguments.

   Decorates an arbitrary Python function so it can be called at compile-time
   within a Gluon kernel on `constexpr` arguments, returning a
   `constexpr` result.

   Parameters
   ----------
   fn : callable
       The Python function to wrap. Must be callable with constexpr arguments
       and return a value that can be converted to a Triton constexpr.

   Returns
   -------
   ConstexprFunction
       A wrapped function object that can be invoked at compile-time within
       Gluon kernels.

   Notes
   -----
   The wrapped function executes during kernel compilation, not at runtime.
   All arguments must be `constexpr` values when called from within a
   Gluon kernel. The function body must use only Python constructs supported
   at compile-time.

   When called from host code or another constexpr function (without the
   `_semantic` keyword argument), the function executes normally and
   returns the raw Python result.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.constexpr_function
   def compute_offset(block_size: int, stride: int) -> int:
       return block_size * stride

   @gluon.jit
   def kernel(x_ptr, BLOCK_SIZE: ttgl.constexpr):
       offset = compute_offset(BLOCK_SIZE, 4)
       # offset is a constexpr value computed at compile-time
```

---

### triton.experimental.gluon.jit

```python
jit(fn: 'Optional[T]' = None, *, version=None, repr: 'Optional[Callable]' = None, launch_metadata: 'Optional[Callable]' = None, do_not_specialize: 'Optional[Iterable[int | str]]' = None, do_not_specialize_on_alignment: 'Optional[Iterable[int | str]]' = None, debug: 'Optional[bool]' = None, noinline: 'Optional[bool]' = None) -> 'Union[GluonJITFunction[T], Callable[[T], JITFunction[T]]]'
```

**`triton.experimental.gluon.jit(fn=None, *, version=None, repr=None, launch_metadata=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None, noinline=None)`**

   Decorator for JIT-compiling a function using the Gluon experimental API.

   Gluon provides explicit control over layouts, shared memory, barriers, and hardware-specific features (NVIDIA Hopper/Blackwell, AMD CDNA3/CDNA4). Kernels should be decorated with `@gluon.jit` rather than `@triton.jit`.

   Parameters
   ----------
   fn : Callable, optional
      The function to be JIT-compiled. If provided, the decorator is applied immediately. If omitted, returns a decorator factory.
   version : int, optional
      Version identifier for the kernel.
   repr : Callable, optional
      Custom function to generate string representation of the kernel.
   launch_metadata : Callable, optional
      Function to attach metadata to the kernel launch.
   do_not_specialize : Iterable[int or str], optional
      Arguments (by index or name) on which specialization should be disabled.
   do_not_specialize_on_alignment : Iterable[int or str], optional
      Arguments (by index or name) on which alignment specialization should be disabled.
   debug : bool, optional
      Enable debug mode for compilation.
   noinline : bool, optional
      Prevent inlining of the function.

   Returns
   -------
   GluonJITFunction or Callable
      If `fn` is provided, returns the compiled `GluonJITFunction`. Otherwise, returns a decorator callable.

   Notes
   -----
   When a JIT'd function is called, arguments are implicitly converted to pointers if they have a `.data_ptr()` method and a `.dtype` attribute.

   The function will be compiled and run on the GPU. It will only have access to:

   * Python primitives
   * Builtins within the `triton` package
   * Arguments to this function
   * Other JIT'd functions

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, n):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid)
       ttgl.store(y_ptr + pid, x)
```

---

### triton.experimental.gluon.must_use_result

```python
must_use_result(x, s=True)
```

## must_use_result


.. autofunction:: must_use_result

**`must_use_result(x, s=True)`**

   Mark a value or function to require its result be used.

   If the result of the decorated function or marked value is unused, an error
   will be raised at compile time. This helps prevent accidental omission of
   operations that have no side effects.

   Parameters
   ----------
   x : object or str
       The value to mark, or a string message when used as a decorator factory.
   s : bool, optional
       Whether the result must be used (default True).

   Returns
   -------
   object
       The input value with the `_must_use_result` attribute set, or a
       decorator function when `x` is a string.

   Notes
   -----
   This function supports two usage patterns:

   1. **Direct marking**: Call `must_use_result(obj)` to mark an object's
      result as required.

   2. **Decorator**: Use `@must_use_result` or `@must_use_result("message")`
      to decorate a function. When a string is provided, it serves as the error
      message.

   The marker sets the `_must_use_result` attribute on the object, which the
   compiler checks to ensure the result is consumed.

   Examples
   --------
   Mark a value to require its result be used:

```python
   import triton.experimental.gluon as gluon

   @gluon.jit
   def kernel(ptr):
       value = gluon.load(ptr)
       # This will raise an error if the result is unused
       gluon.must_use_result(value)

Use as a decorator on a function:

.. code-block:: python

   import triton.experimental.gluon as gluon

   @gluon.must_use_result
   @gluon.jit
   def compute(x):
       return x * 2

   # Calling compute without using the result will raise an error

Use with a custom error message:

.. code-block:: python

   import triton.experimental.gluon as gluon

   @gluon.must_use_result("Result of advance must be assigned")
   @gluon.jit
   def move_ptr(ptr, offset):
       return gluon.advance(ptr, offset)
```

---
