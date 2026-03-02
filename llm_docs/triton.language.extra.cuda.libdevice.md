# triton.language.extra.cuda.libdevice

NVIDIA libdevice math functions — full set of GPU-accelerated math operations (trig, rounding, conversion, special functions).

*197 APIs documented.*

---

## triton.language.extra.cuda.libdevice

### triton.language.extra.cuda.libdevice.abs

```python
abs(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.abs


**`abs(arg0, _semantic=None)`**

    Compute the absolute value of each element in the input tensor.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor. Supported dtypes are int32, int64, fp32, and fp64.
    _semantic : optional
        Internal parameter used by the Triton compiler. Do not set manually.

    Returns
    -------
    out : tl.tensor
        Tensor containing the absolute values of the input elements.
        The output dtype matches the input dtype.

    Notes
    -----
    This function is an extern wrapper that dispatches to CUDA libdevice
    intrinsics based on the input dtype:

    - `int32`: calls `__nv_abs`
    - `int64`: calls `__nv_llabs`
    - `fp32`: calls `__nv_fabsf`
    - `fp64`: calls `__nv_fabs`

    The operation is pure (no side effects) and element-wise.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def abs_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = libdevice.abs(x)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.acos

```python
acos(arg0, _semantic=None)
```

## acos

Compute the arc cosine of the input element-wise.

### Parameters
arg0 : tensor
    Input tensor. Must be of floating-point type (`fp32` or `fp64`).
_semantic : optional
    Internal parameter used by Triton compiler. Do not set manually.

### Returns
tensor
    Output tensor of the same type as `arg0`. Contains the arc cosine
    of each element in the input, in radians.

### Notes
This function wraps CUDA libdevice math functions:

- `__nv_acosf` for 32-bit floating-point (`fp32`)
- `__nv_acos` for 64-bit floating-point (`fp64`)

The output range is $[0, \pi]$ radians. Input values should be in the
domain $[-1, 1]$; results are undefined for values outside this range.

This operation is pure (has no side effects).

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = libdevice.acos(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.acosh

```python
acosh(arg0, _semantic=None)
```

## acosh

Compute the inverse hyperbolic cosine of the input element-wise.

### Parameters
arg0 : tensor
    Input tensor. Must be of floating-point type (`fp32` or `fp64`).
_semantic : optional
    Internal semantic parameter. Do not set manually.

### Returns
tensor
    Output tensor of the same shape as `arg0`. Contains the inverse
    hyperbolic cosine of each element. Returns `fp32` for `fp32`
    input, `fp64` for `fp64` input.

### Notes
This function calls NVIDIA libdevice's `__nv_acoshf` (for `fp32`) or
`__nv_acosh` (for `fp64`) intrinsics. The input values must be
greater than or equal to 1.0 for the result to be defined in the real
number domain.

This function is marked as pure (has no side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offs)
     y = tl.acosh(x)
     tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.add_rd

```python
add_rd(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.add_rd


**`add_rd(arg0, arg1, _semantic=None)`**

    Perform floating-point addition with round-down (toward negative infinity) rounding mode.

    This function calls the CUDA libdevice `__nv_fadd_rd` (for `fp32`) or
    `__nv_dadd_rd` (for `fp64`) intrinsic, which computes `arg0 + arg1`
    with IEEE 754 round-down rounding semantics.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be of type `fp32` or `fp64`.
    arg1 : tl.tensor
        Second input tensor. Must be of type `fp32` or `fp64`.
    _semantic :
        Internal parameter used by Triton compiler. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor containing the element-wise sum of `arg0` and `arg1` with
        round-down rounding. Has the same dtype as the inputs.

    Notes
    -----
    The `_rd` suffix indicates round-down rounding mode (toward negative
    infinity). This differs from the default round-to-nearest-even mode used
    by standard floating-point operations.

    Supported input dtypes:

    - `fp32`: calls `__nv_fadd_rd`
    - `fp64`: calls `__nv_dadd_rd`

    Both inputs must have the same dtype. Mixed precision is not supported.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         y = tl.load(y_ptr + tl.arange(0, BLOCK))
         # Perform addition with round-down rounding mode
         result = tl.extra.cuda.libdevice.add_rd(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK), result)
```

---

### triton.language.extra.cuda.libdevice.add_rn

```python
add_rn(arg0, arg1, _semantic=None)
```

## add_rn


**`add_rn(arg0, arg1, _semantic=None)`**

    Performs floating-point addition with round-to-nearest rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be floating-point type (fp32 or fp64).
    arg1 : tl.tensor
        Second input tensor. Must be floating-point type (fp32 or fp64).
        Must have the same dtype as `arg0`.
    _semantic : optional
        Internal parameter, do not set directly.

    Returns
    -------
    result : tl.tensor
        Tensor containing the element-wise sum of `arg0` and `arg1` with
        round-to-nearest-even rounding mode.

    Notes
    -----
    This function dispatches to CUDA libdevice intrinsics:

    - `__nv_fadd_rn` for 32-bit floating-point (fp32)
    - `__nv_dadd_rn` for 64-bit floating-point (fp64)

    The `rn` suffix indicates round-to-nearest-even rounding mode, which is
    the default IEEE 754 rounding mode. This provides deterministic rounding
    behavior for reproducible numerical results.

    Both inputs must have matching dtypes. Mixed precision operations are not
    supported. The function is marked as pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def add_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(axis=0)
         offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = tl.load(y_ptr + offsets)
         result = tl.extra.cuda.libdevice.add_rn(x, y)
         tl.store(out_ptr + offsets, result)

 See Also
 --------
 tl.add : Standard Triton addition operation
 tl.extra.cuda.libdevice : CUDA libdevice function collection
```

---

### triton.language.extra.cuda.libdevice.add_ru

```python
add_ru(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.add_ru


**`add_ru(arg0, arg1, _semantic=None)`**

    Adds two floating-point values with round-up rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be of type `fp32` or `fp64`.
    arg1 : tl.tensor
        Second input tensor. Must be of type `fp32` or `fp64`.
    _semantic : optional
        Internal semantic argument. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor containing the element-wise sum of `arg0` and `arg1` with
        round-up rounding mode. Same dtype as inputs.

    Notes
    -----
    This function calls CUDA libdevice functions:

    - `__nv_fadd_ru` for `fp32` inputs
    - `__nv_dadd_ru` for `fp64` inputs

    Round-up rounding mode rounds results toward positive infinity. This is
    useful for interval arithmetic and error bound calculations.

    Both inputs must have the same dtype. Mixed precision is not supported.

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
         # Add with round-up rounding mode
         result = tl.extra.cuda.libdevice.add_ru(x, y)
         tl.store(out_ptr + offs, result)
```

---

### triton.language.extra.cuda.libdevice.add_rz

```python
add_rz(arg0, arg1, _semantic=None)
```

## add_rz


**`add_rz(arg0, arg1, _semantic=None)`**

   Adds two floating-point values with round-to-zero rounding mode.

   Parameters
   ----------
   arg0 : tl.tensor
       First input tensor. Must be fp32 or fp64.
   arg1 : tl.tensor
       Second input tensor. Must be fp32 or fp64.
   _semantic
       Internal parameter. Do not set.

   Returns
   -------
   result : tl.tensor
       Element-wise sum of `arg0` and `arg1` with round-to-zero rounding.
       Same dtype as inputs.

   Notes
   -----
   This function calls CUDA libdevice `__nv_fadd_rz` (for fp32) or
   `__nv_dadd_rz` (for fp64). Round-to-zero rounding truncates the
   result towards zero, which differs from the default round-to-nearest
   behavior.

   This function must be used inside a :py`@triton.jit()` decorated kernel.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        result = tl.extra.cuda.libdevice.add_rz(x, y)
        tl.store(out_ptr + offsets, result)
```

---

### triton.language.extra.cuda.libdevice.asin

```python
asin(arg0, _semantic=None)
```

asin(arg0, _semantic=None)

Compute the inverse sine (arcsine) of each element in the input tensor.

### Parameters
arg0 : tensor
    Input tensor. Must be of floating-point type (`fp32` or `fp64`).
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
tensor
    Output tensor of the same shape as `arg0`. Contains the inverse sine
    of each element, in radians. The return dtype matches the input dtype
    (`fp32` or `fp64`).

### Notes
This function wraps the CUDA libdevice `__nv_asinf` (for `fp32`) and
`__nv_asin` (for `fp64`) functions. The output values are in the range
`[-pi/2, pi/2]`. Input values should be in the domain `[-1, 1]`;
behavior is undefined for values outside this range.

The operation is pure (no side effects) and is available only in
JIT-compiled Triton kernels decorated with `@triton.jit`.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.asin(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.asinh

```python
asinh(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.asinh


**`asinh(arg0, _semantic=None)`**

   Compute the inverse hyperbolic sine of the input element-wise.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor. Must have floating-point dtype (`fp32` or `fp64`).
   _semantic : optional
       Internal parameter for semantic propagation. Do not set manually.

   Returns
   -------
   result : tl.tensor
       Tensor containing the inverse hyperbolic sine of each element in `arg0`.
       Has the same dtype and shape as the input.

   Notes
   -----
   This function is an external wrapper around CUDA libdevice functions:
   `__nv_asinhf` for `fp32` and `__nv_asinh` for `fp64`.

   This function can also be called as a member function on :py`tensor`,
   as `x.asinh()` instead of `asinh(x)`.

   The inverse hyperbolic sine is defined as:

   .. math::

      \text{asinh}(x) = \ln(x + \sqrt{x^2 + 1})

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def asinh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       input = tl.load(input_ptr + offsets, mask=mask)
       output = tl.extra.cuda.libdevice.asinh(input)
       tl.store(output_ptr + offsets, output, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.atan

```python
atan(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.atan

Compute the element-wise arctangent of a tensor.

### Parameters
arg0 : tl.tensor
    Input tensor. Must be of floating-point type (`fp32` or `fp64`).

### Returns
result : tl.tensor
    Output tensor of the same type as `arg0`. Contains the arctangent
    values in radians, in the range `[-pi/2, pi/2]`.

### Notes
This function dispatches to CUDA libdevice intrinsics: `__nv_atanf` for
`fp32` inputs and `__nv_atan` for `fp64` inputs. The operation is
pure (no side effects) and supports both scalar and block tensor inputs.

The arctangent function computes the inverse tangent of each element,
returning angles in radians.

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def atan_kernel(X_ptr, Y_ptr, BLOCK: tl.constexpr):
     offsets = tl.arange(0, BLOCK)
     x = tl.load(X_ptr + offsets)
     y = libdevice.atan(x)
     tl.store(Y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.atan2

```python
atan2(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.atan2


**`atan2(arg0, arg1, _semantic=None)`**

    Compute the element-wise arc tangent of `arg0 / arg1` using the signs of both arguments to determine the correct quadrant.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor (y-coordinate). Must be floating-point type (fp32 or fp64).
    arg1 : tl.tensor
        Second input tensor (x-coordinate). Must be floating-point type (fp32 or fp64).
    _semantic : optional
        Internal semantic argument. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of angles in radians. Has the same dtype as inputs (fp32 or fp64).

    Notes
    -----
    This function is a wrapper around CUDA libdevice's `__nv_atan2f` (fp32) and `__nv_atan2` (fp64) intrinsics.

    The result is in the range `[-pi, pi]`. Unlike `arctan(arg0 / arg1)`, `atan2` correctly handles all quadrants and the case where `arg1` is zero.

    Special cases:
    - `atan2(0, 0)` returns `0`
    - `atan2(+0, -0)` returns `pi`
    - `atan2(-0, -0)` returns `-pi`
    - `atan2(+inf, +inf)` returns `pi/4`
    - `atan2(+inf, -inf)` returns `3*pi/4`

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, n):
         pid = tl.program_id(0)
         x = tl.load(x_ptr + pid)
         y = tl.load(y_ptr + pid)
         angle = tl.extra.cuda.libdevice.atan2(y, x)
         tl.store(out_ptr + pid, angle)
```

---

### triton.language.extra.cuda.libdevice.atanh

```python
atanh(arg0, _semantic=None)
```

## atanh

Compute the inverse hyperbolic tangent of the input element-wise.

### Parameters
arg0 : tensor
    Input tensor. Must have floating-point dtype (`fp32` or `fp64`).
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
tensor
    Output tensor of the same shape as `arg0`. Contains the inverse
    hyperbolic tangent of each element.

### Notes
This function calls CUDA libdevice functions:

- `__nv_atanhf` for `fp32` inputs
- `__nv_atanh` for `fp64` inputs

The inverse hyperbolic tangent is defined as:

.. math::

    \text{atanh}(x) = \frac{1}{2} \ln\left(\frac{1+x}{1-x}\right)

The domain is $x \in (-1, 1)$. Values outside this range return NaN.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def atanh_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.atanh(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.brev

```python
brev(arg0, _semantic=None)
```

## brev

Bit-reverse the input integer.

### Parameters
arg0 : tensor of int32 or int64
    Input tensor whose bits will be reversed.

### Returns
tensor of int32 or int64
    Output tensor with bits reversed. The dtype matches the input dtype.

### Notes
This function calls the CUDA libdevice functions `__nv_brev` for int32
and `__nv_brevll` for int64. The operation reverses the order of bits
in the binary representation of each element. For example, the binary
pattern `000...001` becomes `100...000`.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offs)
     y = tl.extra.cuda.libdevice.brev(x)
     tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.byte_perm

```python
byte_perm(arg0, arg1, arg2, _semantic=None)
```

## triton.language.extra.cuda.libdevice.byte_perm


**`byte_perm(arg0, arg1, arg2, _semantic=None)`**

    Perform byte permutation on a 32-bit integer.

    Rearranges the bytes of `arg0` according to the selection pattern
    encoded in `arg1` and `arg2`. Each nibble (4 bits) in the selector
    arguments specifies which byte position (0-3) to select from `arg0`.

    Parameters
    ----------
    arg0 : tl.int32
        The 32-bit integer whose bytes are to be permuted.
    arg1 : tl.int32
        First selector value. Each nibble specifies a byte index (0-3)
        for the lower two output bytes.
    arg2 : tl.int32
        Second selector value. Each nibble specifies a byte index (0-3)
        for the upper two output bytes.
    _semantic : optional
        Internal semantic argument. Do not set manually.

    Returns
    -------
    out : tl.int32
        The permuted 32-bit integer.

    Notes
    -----
    This function maps to the CUDA libdevice function `__nv_byte_perm`.
    The selector arguments encode byte indices in their nibbles (4-bit
    values). For each output byte position, the corresponding nibble in
    the selector arguments specifies which input byte (0-3) to select.

    Byte indices must be in the range 0-3. Invalid indices result in
    undefined behavior.

    This is a pure function with no side effects.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK_SIZE
         val = tl.load(ptr + offset)
         # Select bytes in reverse order (3, 2, 1, 0)
         selector_low = 0x01020300  # bytes 0,1,2,3 -> positions 0,1,2,3
         selector_high = 0x00000000
         permuted = tl.extra.cuda.libdevice.byte_perm(val, selector_low, selector_high)
         tl.store(ptr + offset, permuted)
```

---

### triton.language.extra.cuda.libdevice.cbrt

```python
cbrt(arg0, _semantic=None)
```

Compute the cube root of the input tensor.

### Parameters
arg0 : tensor
    Input tensor of floating-point type (`fp32` or `fp64`).
_semantic :
    Internal parameter, do not use directly.

### Returns
tensor
    Tensor of the same dtype as `arg0` containing the cube root of each element.

### Notes
This function calls the CUDA libdevice functions `__nv_cbrtf` for `fp32` and
`__nv_cbrt` for `fp64`. The operation is element-wise and pure (no side effects).

Only `fp32` and `fp64` dtypes are supported.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def cbrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(axis=0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
     x = tl.load(input_ptr + offsets, mask=mask)
     y = tl.extra.cuda.libdevice.cbrt(x)
     tl.store(output_ptr + offsets, y, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.ceil

```python
ceil(arg0, _semantic=None)
```

## ceil

Compute the ceiling of the input element-wise.

### Parameters
arg0 : tensor
    Input tensor. Must be a floating-point type (`fp32` or `fp64`).

### Returns
tensor
    Tensor of the same shape as `arg0` containing the ceiling values.
    The output dtype matches the input dtype.

### Notes
This function is CUDA-specific and wraps the NVIDIA libdevice functions
`__nv_ceil` (for `fp64`) and `__nv_ceilf` (for `fp32`). The ceiling
operation rounds each element to the nearest integer not less than the input
value (i.e., rounds toward positive infinity).

This is an extern function that compiles directly to GPU machine code.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.load(X_ptr + tl.arange(0, BLOCK_SIZE))
     y = tl.ceil(x)
     tl.store(Y_ptr + tl.arange(0, BLOCK_SIZE), y)

 # Usage with fp32 input
 import torch
 x = torch.tensor([1.2, 2.7, -0.5, 3.0], dtype=torch.float32, device='cuda')
 # Result: [2.0, 3.0, -0.0, 3.0]
```

---

### triton.language.extra.cuda.libdevice.clz

```python
clz(arg0, _semantic=None)
```

## clz

Count leading zeros.

Counts the number of zero bits preceding the most significant bit in the binary
representation of the input integer.

### Parameters
arg0 : tensor
    Input tensor of integer type. Supported dtypes are `int32` and `int64`.

### Returns
out : tensor
    Tensor of `int32` dtype containing the count of leading zeros for each
    element in the input. For each element, the return value is the number of
    zero bits before the most significant one bit.

### Notes
This function is a wrapper around CUDA libdevice functions:

- `__nv_clz` for `int32` inputs
- `__nv_clzll` for `int64` inputs

The result is always returned as `int32` regardless of input type. For an
input of zero, the result is the bitwidth of the input type (32 for int32,
64 for int64).

This is a pure function with no side effects.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     leading_zeros = tl.extra.cuda.libdevice.clz(x)
     tl.store(out_ptr + tl.arange(0, BLOCK), leading_zeros)

 # Example: for int32 value 0x00000001 (binary: 0...01)
 # clz returns 31 (31 leading zeros before the 1 bit)
```

---

### triton.language.extra.cuda.libdevice.copysign

```python
copysign(arg0, arg1, _semantic=None)
```

## copysign

**`triton.language.extra.cuda.libdevice.copysign(arg0, arg1, _semantic=None)`**

   Copies the sign of `arg1` to `arg0`.

   Returns a value with the magnitude of `arg0` and the sign of `arg1`.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor or scalar. Must be of floating-point type (`fp32` or `fp64`).
       The magnitude of the result is taken from this argument.
   arg1 : tl.tensor
       Input tensor or scalar. Must be of floating-point type (`fp32` or `fp64`).
       The sign of the result is taken from this argument.
   _semantic
       Internal argument. Do not set manually.

   Returns
   -------
   tl.tensor
       A tensor with the magnitude of `arg0` and the sign of `arg1`.

   Notes
   -----
   This function maps to CUDA libdevice functions:

   * `__nv_copysignf` for `fp32` inputs.
   * `__nv_c cast` for `fp64` inputs.

   Both arguments must have the same dtype. The operation is pure (no side effects).

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def copysign_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n

        x = tl.load(x_ptr + offset, mask=mask)
        y = tl.load(y_ptr + offset, mask=mask)

        # Copy the sign of y to x
        result = tl.extra.cuda.libdevice.copysign(x, y)

        tl.store(out_ptr + offset, result, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.cos

```python
cos(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.cos


**`cos(arg0, _semantic=None)`**

    Compute the element-wise cosine of the input tensor.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor. Must be of floating-point type (`fp32` or `fp64`).
    _semantic : optional
        Internal parameter used by Triton compiler. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of the same shape as `arg0` containing the cosine of each element.
        The output dtype matches the input dtype (`fp32` input produces `fp32`
        output, `fp64` input produces `fp64` output).

    Notes
    -----
    This function dispatches to CUDA libdevice intrinsics:

    - `__nv_cosf` for `fp32` inputs
    - `__nv_cos` for `fp64` inputs

    The operation is marked as pure (no side effects), enabling compiler optimizations.

    Input values are expected to be in radians.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def cosine_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = libdevice.cos(x)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.cosh

```python
cosh(arg0, _semantic=None)
```

## cosh

Compute the hyperbolic cosine of each element in the input tensor.

### Parameters
arg0 : tensor
    Input tensor. Must have floating-point dtype (fp32 or fp64).
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
tensor
    Element-wise hyperbolic cosine of the input. Same dtype as input.

### Notes
This function dispatches to CUDA libdevice:

- `__nv_coshf` for fp32 inputs
- `__nv_cosh` for fp64 inputs

The operation is pure (no side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     y = tl.extra.cuda.libdevice.cosh(x)
     tl.store(y_ptr + tl.arange(0, BLOCK), y)
```

---

### triton.language.extra.cuda.libdevice.cospi

```python
cospi(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.cospi


**`cospi(arg0, _semantic=None)`**

    Compute the cosine of the input argument multiplied by $\pi$.

    Computes $\cos(\pi \cdot x)$ for each element in the input tensor.
    This function uses CUDA libdevice intrinsics for efficient computation.

    Parameters
    ----------
    arg0 : tensor
        Input tensor of floating-point type. Supported dtypes are `fp32` and
        `fp64`.

    Returns
    -------
    tensor
        Output tensor of the same dtype as `arg0`, containing the cosine
        values.

    Notes
    -----
    This function is CUDA-specific and relies on libdevice intrinsics:
    `__nv_cospif` for `fp32` and `__nv_cospi` for `fp64`.

    The function is pure (no side effects) and supports elementwise operation
    on tensors.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def cospi_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         mask = offsets < n
         x = tl.load(x_ptr + offsets, mask=mask)
         y = tl.extra.cuda.libdevice.cospi(x)
         tl.store(y_ptr + offsets, y, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.cyl_bessel_i0

```python
cyl_bessel_i0(arg0, _semantic=None)
```

## cyl_bessel_i0

Compute the modified Bessel function of the first kind of order 0.

### Parameters
arg0 : tensor
    Input tensor of floating-point values. Must be `fp32` or `fp64` dtype.
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
result : tensor
    Tensor of the same dtype as `arg0` containing the computed I0 values.

### Notes
This function is an extern operation that maps to CUDA libdevice functions:
`__nv_cyl_bessel_i0f` for `fp32` and `__nv_cyl_bessel_i0` for `fp64`.

The modified Bessel function of the first kind of order 0 is defined as:

.. math::

    I_0(x) = \sum_{k=0}^{\infty} \frac{(x/2)^{2k}}{(k!)^2}

This function is pure (no side effects) and supports elementwise computation
on tensors.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.cyl_bessel_i0(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.cyl_bessel_i1

```python
cyl_bessel_i1(arg0, _semantic=None)
```

### cyl_bessel_i1

**`triton.language.extra.cuda.libdevice.cyl_bessel_i1(arg0, _semantic=None)`**

   Compute the modified Bessel function of the first kind of order 1 (I₁).

   Parameters
   ----------
   arg0 : tensor
       Input tensor of floating-point type (`fp32` or `fp64`).
   _semantic : optional
       Internal parameter, do not set manually.

   Returns
   -------
   result : tensor
       Output tensor of the same dtype as input. Contains $I_1(arg0)$ for each element.

   Notes
   -----
   This function dispatches to CUDA libdevice functions:

   - `__nv_cyl_bessel_i1f` for `fp32` inputs
   - `__nv_cyl_bessel_i1` for `fp64` inputs

   The modified Bessel function of the first kind of order 1 is defined as:

   .. math::

       I_1(x) = \sum_{k=0}^{\infty} \frac{1}{k!(k+1)!} \left(\frac{x}{2}\right)^{2k+1}

   For large positive arguments, $I_1(x)$ grows exponentially. For negative arguments, $I_1(-x) = -I_1(x)$.

   Examples
   --------
```python
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def bessel_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = libdevice.cyl_bessel_i1(x)
        tl.store(y_ptr + offsets, y)

See Also
--------
cyl_bessel_i0 : Modified Bessel function of the first kind of order 0
cyl_bessel_j1 : Bessel function of the first kind of order 1
exp : Exponential function
```

---

### triton.language.extra.cuda.libdevice.div_rd

```python
div_rd(arg0, arg1, _semantic=None)
```

## div_rd


**`div_rd(arg0, arg1, _semantic=None)`**

    Divide arg0 by arg1 with round-down (toward negative infinity) rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        The dividend. Must be floating-point type (fp32 or fp64).
    arg1 : tl.tensor
        The divisor. Must be floating-point type (fp32 or fp64).
    _semantic : optional
        Internal parameter, do not set manually.

    Returns
    -------
    result : tl.tensor
        The quotient arg0 / arg1 with round-down rounding mode. Same dtype as inputs.

    Notes
    -----
    This function calls CUDA libdevice functions `__nv_fdiv_rd` for fp32 and
    `__nv_ddiv_rd` for fp64. The rounding mode is round-down (toward negative
    infinity), which differs from the default round-to-nearest-even mode.

    Only available on CUDA targets.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, n):
         pid = tl.program_id(0)
         x = tl.load(x_ptr + pid)
         y = tl.load(y_ptr + pid)
         result = tl.extra.cuda.libdevice.div_rd(x, y)
         tl.store(out_ptr + pid, result)
```

---

### triton.language.extra.cuda.libdevice.div_rn

```python
div_rn(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.div_rn


**`div_rn(arg0, arg1, _semantic=None)`**

    Divide arg0 by arg1 with round-to-nearest rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Dividend tensor. Must be floating point (fp32 or fp64).
    arg1 : tl.tensor
        Divisor tensor. Must be floating point (fp32 or fp64) with same dtype as arg0.
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor containing arg0 / arg1 with round-to-nearest rounding. Same dtype as inputs.

    Notes
    -----
    This function calls CUDA libdevice division intrinsics:

    - `__nv_fdiv_rn` for fp32 inputs
    - `__nv_ddiv_rn` for fp64 inputs

    The `_rn` suffix indicates round-to-nearest rounding mode. Both inputs must have
    the same floating point dtype. Division by zero follows IEEE 754 semantics.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         mask = offs < n
         x = tl.load(x_ptr + offs, mask=mask)
         y = tl.load(y_ptr + offs, mask=mask)
         out = tl.extra.cuda.libdevice.div_rn(x, y)
         tl.store(out_ptr + offs, out, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.div_ru

```python
div_ru(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.div_ru


**`div_ru(arg0, arg1, _semantic=None)`**

    Divide `arg0` by `arg1` and round the result towards positive infinity.

    Parameters
    ----------
    arg0 : tl.tensor
        Numerator. Must be floating-point type (`fp32` or `fp64`).
    arg1 : tl.tensor
        Denominator. Must be floating-point type (`fp32` or `fp64`).
        Must have the same dtype as `arg0`.
    _semantic : optional
        Internal parameter. Do not set manually.

    Returns
    -------
    tl.tensor
        Tensor of the same dtype as inputs containing `arg0 / arg1` rounded
        towards positive infinity.

    Notes
    -----
    This function calls CUDA libdevice functions:

    - `__nv_fdiv_ru` for `fp32` inputs
    - `__nv_ddiv_ru` for `fp64` inputs

    The rounding mode is "round up" (towards +inf). For example, `div_ru(2.5, 1.0)`
    returns `3.0`.

    Only `fp32` and `fp64` dtypes are supported. Both arguments must have
    the same dtype.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(X, Y, Z, BLOCK: tl.constexpr):
         offsets = tl.arange(0, BLOCK)
         x = tl.load(X + offsets)
         y = tl.load(Y + offsets)
         z = tl.extra.cuda.libdevice.div_ru(x, y)
         tl.store(Z + offsets, z)
```

---

### triton.language.extra.cuda.libdevice.div_rz

```python
div_rz(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.div_rz


**`div_rz(arg0, arg1, _semantic=None)`**

    Divide `arg0` by `arg1` with round-towards-zero rounding mode.

    Parameters
    ----------
    arg0 : tensor
        The dividend. Must be floating-point type (`fp32` or `fp64`).
    arg1 : tensor
        The divisor. Must be floating-point type (`fp32` or `fp64`).
    _semantic : optional
        Internal parameter for semantic propagation. Do not set manually.

    Returns
    -------
    result : tensor
        The quotient `arg0 / arg1` rounded towards zero. Has the same dtype
        as the inputs.

    Notes
    -----
    This function calls NVIDIA libdevice functions:

    - `__nv_fdiv_rz` for `fp32` inputs
    - `__nv_ddiv_rz` for `fp64` inputs

    Round-towards-zero (rz) truncates the result toward zero, equivalent to
    C-style integer division behavior for floating-point values.

    Both inputs must have the same dtype. Mixed precision is not supported.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
         # Division with round-towards-zero
         result = tl.extra.cuda.libdevice.div_rz(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

---

### triton.language.extra.cuda.libdevice.double2float_rd

```python
double2float_rd(arg0, _semantic=None)
```

## double2float_rd


**`double2float_rd(arg0, _semantic=None)`**

    Convert double-precision floating-point value to single-precision with round-down rounding.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of double-precision floating-point values (fp64).

    Returns
    -------
    tl.tensor
        Tensor of single-precision floating-point values (fp32) rounded toward negative infinity.

    Notes
    -----
    This function invokes the CUDA libdevice function `__nv_double2float_rd` which performs
    conversion from 64-bit to 32-bit floating-point format with rounding toward negative infinity
    (round-down mode). This differs from the default round-to-nearest-even behavior of standard
    type casting.

    The function is pure (no side effects) and operates element-wise on the input tensor.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def kernel(x_ptr, y_ptr, n):
         pid = tl.program_id(0)
         offset = pid * 4
         x = tl.load(x_ptr + offset + tl.arange(0, 4))
         # Convert fp64 to fp32 with round-down rounding
         y = libdevice.double2float_rd(x)
         tl.store(y_ptr + offset + tl.arange(0, 4), y)
```

---

### triton.language.extra.cuda.libdevice.double2float_rn

```python
double2float_rn(arg0, _semantic=None)
```

## double2float_rn


**`double2float_rn(arg0, _semantic=None)`**

    Convert double-precision floating-point value to single-precision with round-to-nearest-even rounding.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of double-precision floating-point values (`fp64`).

    Returns
    -------
    tl.tensor
        Tensor of single-precision floating-point values (`fp32`).

    Notes
    -----
    This function wraps the CUDA libdevice function `__nv_double2float_rn`.
    The rounding mode is round-to-nearest, ties-to-even (RN).
    This function is pure (no side effects).

    Only available on CUDA targets.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         offset = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offset)  # fp64 tensor
         y = tl.extra.cuda.libdevice.double2float_rn(x)  # fp32 tensor
         tl.store(y_ptr + offset, y)
```

---

### triton.language.extra.cuda.libdevice.double2float_ru

```python
double2float_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2float_ru


**`double2float_ru(arg0, _semantic=None)`**

    Convert double-precision floating-point value to single-precision with round-up rounding.

    This function invokes the CUDA libdevice `__nv_double2float_ru` intrinsic, which
    converts a 64-bit floating-point value to a 32-bit floating-point value using
    round-up (toward positive infinity) rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of double-precision floating-point values (`fp64`).

    _semantic : optional
        Internal Triton semantic parameter. Do not set directly.

    Returns
    -------
    tl.tensor
        Tensor of single-precision floating-point values (`fp32`).

    Notes
    -----
    This is a pure function with no side effects. The rounding mode is round-up
    (toward positive infinity), which differs from the default round-to-nearest
    behavior of standard type casting.

    This function is only available on CUDA targets with libdevice support.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offset).to(tl.float64)
         y = tl.extra.cuda.libdevice.double2float_ru(x)
         tl.store(y_ptr + offset, y)
```

---

### triton.language.extra.cuda.libdevice.double2float_rz

```python
double2float_rz(arg0, _semantic=None)
```

## double2float_rz


**`double2float_rz(arg0, _semantic=None)`**

    Convert double-precision floating-point value to single-precision with round-towards-zero.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of double-precision (fp64) floating-point values.

    Returns
    -------
    tl.tensor
        Tensor of single-precision (fp32) floating-point values.

    Notes
    -----
    This function calls the CUDA libdevice function `__nv_double2float_rz` which performs
    conversion from 64-bit to 32-bit floating-point with round-towards-zero (RTZ) rounding mode.
    This is useful when deterministic rounding behavior is required, as opposed to the default
    round-to-nearest-even mode.

    The function is pure and has no side effects. Only fp64 input type is supported.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def kernel(ptr_in, ptr_out, BLOCK_SIZE: tl.constexpr):
         offset = tl.arange(0, BLOCK_SIZE)
         x = tl.load(ptr_in + offset)  # fp64 tensor
         y = libdevice.double2float_rz(x)  # convert to fp32 with RTZ
         tl.store(ptr_out + offset, y)
```

---

### triton.language.extra.cuda.libdevice.double2hiint

```python
double2hiint(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2hiint


**`double2hiint(arg0, _semantic=None)`**

   Extract the high 32 bits of a double-precision floating-point value as a signed integer.

   Parameters
   ----------
   arg0 : tl.tensor
      Input tensor of type `tl.float64`. The double-precision floating-point values
      whose high 32 bits will be extracted.
   _semantic : optional
      Internal semantic argument. Do not set manually.

   Returns
   -------
   tl.tensor
      Tensor of type `tl.int32` containing the high 32 bits of each input double
      interpreted as a signed integer.

   Notes
   -----
   This function wraps the CUDA libdevice intrinsic `__nv_double2hiint`. It is
   commonly used for type-punning or bit-level manipulation of floating-point
   values. The operation is pure (no side effects) and compiles to a single
   GPU instruction.

   For the low 32 bits, use `double2loint()`. To reconstruct a double from
   high and low parts, use `hiloint2double()`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def extract_high_bits_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK + tl.arange(0, BLOCK)
       x = tl.load(x_ptr + offsets).to(tl.float64)
       high_bits = tl.libdevice.double2hiint(x)
       tl.store(out_ptr + offsets, high_bits)
```

---

### triton.language.extra.cuda.libdevice.double2int_rd

```python
double2int_rd(arg0, _semantic=None)
```

## double2int_rd


**`double2int_rd(arg0, _semantic=None)`**

   Convert double-precision floating-point value to 32-bit integer with round-down rounding.

   Parameters
   ----------
   arg0 : tl.tensor
      Input tensor of float64 (fp64) values.
   _semantic : optional
      Internal parameter, do not set manually.

   Returns
   -------
   out : tl.tensor
      Tensor of int32 values, rounded toward negative infinity.

   Notes
   -----
   This function calls the CUDA libdevice function `__nv_double2int_rd` which
   converts double-precision floating-point values to 32-bit signed integers
   using round-down (toward negative infinity) rounding mode.

   The rounding behavior differs from standard truncation for negative non-integer
   values. For example, `-3.7` becomes `-4` (not `-3`).

   This function is CUDA-specific and requires the NVIDIA GPU backend. The operation
   is pure and elementwise, operating independently on each element of the input
   tensor.

   Only `fp64` input type is supported.

   Examples
   --------
```python
   import triton
   import triton.language as tl
   import torch

   @triton.jit
   def convert_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets).to(tl.float64)
       y = tl.extra.cuda.libdevice.double2int_rd(x)
       tl.store(y_ptr + offsets, y)

   # Example usage
   x = torch.tensor([3.7, -3.7, 0.0, -0.0], dtype=torch.float64, device='cuda')
   y = torch.empty_like(x, dtype=torch.int32)
   convert_kernel[(1,)](x, y, BLOCK_SIZE=4)
   # y will be [3, -4, 0, 0]
```

---

### triton.language.extra.cuda.libdevice.double2int_rn

```python
double2int_rn(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2int_rn


**`double2int_rn(arg0, _semantic=None)`**

   Convert double-precision floating-point value to 32-bit integer using round-to-nearest rounding mode.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of dtype `tl.float64`. Values are rounded to the nearest integer.

   _semantic : optional
       Internal semantic parameter. Do not set manually.

   Returns
   -------
   tl.tensor
       Tensor of dtype `tl.int32` containing the converted values.

   Notes
   -----
   This function wraps the CUDA libdevice intrinsic `__nv_double2int_rn`. The rounding mode is round-to-nearest-even (ties round to even). Values outside the representable int32 range result in undefined behavior.

   This function is pure (no side effects) and can be used in device code within `@triton.jit` decorated kernels.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)  # fp64 tensor
       y = tl.extra.cuda.libdevice.double2int_rn(x)  # convert to int32
       tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.double2int_ru

```python
double2int_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2int_ru


**`double2int_ru(arg0, _semantic=None)`**

    Convert a double-precision floating-point value to a 32-bit integer using round-up rounding mode.

    This function invokes the CUDA libdevice intrinsic `__nv_double2int_ru`, which rounds the input
    toward positive infinity (round-up) before converting to integer.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of double-precision floating-point values (`tl.float64`).

    _semantic : optional
        Internal semantic argument; do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of 32-bit signed integers (`tl.int32`) with the same shape as `arg0`.

    Notes
    -----
    The rounding mode "ru" stands for "round up" (toward positive infinity). For example:

    - `3.2` rounds to `4`
    - `-3.2` rounds to `-3`
    - `3.0` rounds to `3`

    This function is only available on CUDA targets and requires the input to be `float64` dtype.
    The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offs).to(tl.float64)
         y = tl.extra.cuda.libdevice.double2int_ru(x)
         tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.double2int_rz

```python
double2int_rz(arg0, _semantic=None)
```

## double2int_rz


**`double2int_rz(arg0, _semantic=None)`**

   Convert double-precision floating-point value to 32-bit signed integer with round-towards-zero.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of double-precision floating-point values (`fp64`).
   _semantic : optional
       Internal semantic argument used by the Triton compiler. Do not pass manually.

   Returns
   -------
   result : tl.tensor
       Tensor of 32-bit signed integers (`int32`).

   Notes
   -----
   This function invokes the CUDA libdevice function `__nv_double2int_rz`.

   The rounding mode is round-towards-zero (truncation), which rounds positive
   values down and negative values up towards zero. This differs from the
   default round-to-nearest-even mode used by standard type casting.

   This function is only available on CUDA targets. The input must be of type
   `fp64` and the output will be `int32`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets).to(tl.float64)
       # Convert fp64 to int32 with round-towards-zero
       y = tl.extra.cuda.libdevice.double2int_rz(x)
       tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.double2ll_rd

```python
double2ll_rd(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2ll_rd


**`double2ll_rd(arg0, _semantic=None)`**

   Convert double-precision floating-point value to 64-bit signed integer with round-down rounding.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of double-precision floating-point values (`tl.float64`).
   _semantic : optional
       Internal semantic argument, do not provide manually.

   Returns
   -------
   tl.tensor
       Tensor of 64-bit signed integers (`tl.int64`) with the same shape as `arg0`.

   Notes
   -----
   This function calls the CUDA libdevice function `__nv_double2ll_rd` which converts
   double-precision floating-point values to 64-bit signed integers using round-down
   (toward negative infinity) rounding mode. This differs from standard truncation
   rounding which rounds toward zero.

   For positive values, round-down and truncation produce the same result. For negative
   non-integer values, round-down produces a more negative result (e.g., `-3.7`
   becomes `-4` with round-down, but `-3` with truncation).

   The function is pure (no side effects) and operates element-wise on tensors.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def convert_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets).to(tl.float64)
        # Convert float64 to int64 with round-down rounding
        out = tl.extra.cuda.libdevice.double2ll_rd(x)
        tl.store(out_ptr + offsets, out)
```

---

### triton.language.extra.cuda.libdevice.double2ll_rn

```python
double2ll_rn(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2ll_rn

Convert double-precision floating-point value to 64-bit signed integer with round-to-nearest rounding.

### Parameters
arg0 : tensor of fp64
    Input double-precision floating-point value to convert.
_semantic : optional
    Internal semantic parameter for Triton type checking. Do not set manually.

### Returns
tensor of int64
    The input value converted to 64-bit signed integer with round-to-nearest
    rounding mode.

### Notes
This function wraps the CUDA libdevice intrinsic `__nv_double2ll_rn`. The
`rn` suffix indicates round-to-nearest-even rounding mode, which is the
default IEEE 754 rounding mode.

For values outside the representable range of int64, the result is undefined.
NaN inputs produce undefined results.

This is a pure function with no side effects.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offset = pid * BLOCK_SIZE
     x = tl.load(x_ptr + offset)
     # Convert fp64 to int64 with round-to-nearest
     y = tl.libdevice.double2ll_rn(x)
     tl.store(y_ptr + offset, y)
```

---

### triton.language.extra.cuda.libdevice.double2ll_ru

```python
double2ll_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2ll_ru


**`double2ll_ru(arg0, _semantic=None)`**

    Convert a double-precision floating-point value to a 64-bit signed integer with round toward zero.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of dtype `tl.float64`. The floating-point values to convert.

    _semantic : optional
        Internal semantic argument. Do not set manually.

    Returns
    -------
    tl.tensor
        Tensor of dtype `tl.int64` containing the converted integer values.

    Notes
    -----
    This function invokes the CUDA libdevice intrinsic `__nv_double2ll_ru` which performs
    conversion with round toward zero (truncate) semantics. Fractional parts are discarded.

    This is a pure function with no side effects.

    Only available on CUDA targets with libdevice support.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offs).to(tl.float64)
         y = tl.extra.cuda.libdevice.double2ll_ru(x)
         tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.double2ll_rz

```python
double2ll_rz(arg0, _semantic=None)
```

## double2ll_rz


**`double2ll_rz(arg0, _semantic=None)`**

   Convert double-precision floating point to 64-bit integer with round towards zero.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of double-precision floating point values (fp64).
   _semantic : optional
       Internal semantic argument, do not set manually.

   Returns
   -------
   tl.tensor
       Tensor of 64-bit signed integers (int64) with values rounded towards zero.

   Notes
   -----
   This function calls the CUDA libdevice function `__nv_double2ll_rz` which
   converts double-precision floating point values to 64-bit signed integers
   using round towards zero (truncate) rounding mode.

   For positive values, this truncates the fractional part. For negative values,
   this also truncates towards zero (e.g., -3.7 becomes -3).

   This function is pure and has no side effects.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offset)  # fp64 values
       y = tl.extra.cuda.libdevice.double2ll_rz(x)  # convert to int64
       tl.store(y_ptr + offset, y)
```

---

### triton.language.extra.cuda.libdevice.double2loint

```python
double2loint(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2loint

Extract the low 32 bits of a double-precision floating-point value as an integer.

### Parameters
arg0 : tensor
    Input tensor of type `fp64` (double-precision floating-point).
_semantic : optional
    Internal semantic parameter for Triton JIT compilation. Do not set manually.

### Returns
tensor
    Tensor of type `int32` containing the low 32 bits of the input double's
    bit representation.

### Notes
This function is a wrapper around the CUDA libdevice function `__nv_double2loint`.
It performs a bit-cast operation, interpreting the lower 32 bits of the IEEE 754
double-precision representation as a signed 32-bit integer. This is useful for
low-level bit manipulation of floating-point values.

The function is pure (no side effects) and operates element-wise on tensors.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def extract_low_bits_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offs)
     # Extract low 32 bits of double representation
     low_bits = tl.extra.cuda.libdevice.double2loint(x)
     tl.store(out_ptr + offs, low_bits)
```

---

### triton.language.extra.cuda.libdevice.double2uint_rd

```python
double2uint_rd(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2uint_rd

Convert double-precision floating-point value to 32-bit unsigned integer with round-down rounding mode.

### Parameters
arg0 : tensor of fp64
    Input tensor containing double-precision floating-point values to convert.

### Returns
tensor of int32
    Tensor containing 32-bit unsigned integer values converted from the input.

### Notes
This function invokes the CUDA libdevice intrinsic `__nv_double2uint_rd` which converts
double-precision floating-point values to unsigned 32-bit integers using round-down
(round toward negative infinity) rounding mode.

The operation is pure (no side effects) and element-wise. Only `fp64` input type
is supported.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offs).to(tl.float64)
     y = tl.extra.cuda.libdevice.double2uint_rd(x)
     tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.double2uint_rn

```python
double2uint_rn(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2uint_rn


**`double2uint_rn(arg0, _semantic=None)`**

   Convert double-precision floating-point value to unsigned 32-bit integer with round-to-nearest rounding.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of double-precision floating-point values (`tl.float64`).
   _semantic : optional
       Internal semantic argument, do not provide manually.

   Returns
   -------
   tl.tensor
       Tensor of unsigned 32-bit integers (`tl.int32`).

   Notes
   -----
   This function wraps the CUDA libdevice intrinsic `__nv_double2uint_rn`.
   The `rn` suffix indicates round-to-nearest-even rounding mode.

   Values outside the representable range of uint32 (0 to 4294967295) produce
   undefined behavior. NaN and infinity inputs also produce undefined results.

   This function is CUDA-specific and requires a GPU backend.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offset)  # fp64 values
       y = tl.extra.cuda.libdevice.double2uint_rn(x)  # convert to uint32
       tl.store(y_ptr + offset, y)
```

---

### triton.language.extra.cuda.libdevice.double2uint_ru

```python
double2uint_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2uint_ru


**`double2uint_ru(arg0, _semantic=None)`**

   Convert a double-precision floating-point value to an unsigned 32-bit integer with round toward positive infinity.

   Parameters
   ----------
   arg0 : tl.tensor
      Input tensor of double-precision floating-point values (`tl.float64`).
   _semantic : optional
      Internal semantic argument; do not set manually.

   Returns
   -------
   tl.tensor
      Tensor of 32-bit integer values (`tl.int32`) with the same shape as `arg0`.

   Notes
   -----
   This function wraps the NVIDIA libdevice function `__nv_double2uint_ru`.
   The rounding mode "ru" indicates round toward positive infinity (round up).
   Values outside the representable range of 32-bit unsigned integers may produce undefined behavior.
   This is a pure function with no side effects.

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
       y = tl.extra.cuda.libdevice.double2uint_ru(x)
       tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.double2uint_rz

```python
double2uint_rz(arg0, _semantic=None)
```

## double2uint_rz

Convert double-precision floating-point values to 32-bit integers using
round-towards-zero rounding mode.

### Parameters
arg0 : tensor
    Input tensor of double-precision floating-point values (fp64).
_semantic : optional
    Internal parameter for Triton semantic analysis. Do not provide manually.

### Returns
tensor
    Tensor of 32-bit signed integers (int32) containing the converted values.

### Notes
This function wraps the CUDA libdevice function `__nv_double2uint_rz`.
The conversion truncates the fractional portion (round towards zero).

Values outside the representable range of int32 produce undefined results.
NaN and infinity inputs produce undefined results.

This function must be called within a `@triton.jit` decorated kernel.
The `_semantic` parameter is handled automatically by the Triton compiler.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def convert_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offset = pid * BLOCK_SIZE
     x = tl.load(x_ptr + offset).to(tl.float64)
     out = tl.extra.cuda.libdevice.double2uint_rz(x)
     tl.store(out_ptr + offset, out)
```

---

### triton.language.extra.cuda.libdevice.double2ull_rd

```python
double2ull_rd(arg0, _semantic=None)
```

## double2ull_rd


**`double2ull_rd(arg0, _semantic=None)`**

    Convert double-precision floating-point value to unsigned 64-bit integer with round-down rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of double-precision floating-point values (`fp64`).

    Returns
    -------
    tl.tensor
        Tensor of unsigned 64-bit integers (`int64`). Each element is the input value
        rounded down (towards negative infinity) and converted to unsigned long long.

    Notes
    -----
    This function calls the CUDA libdevice intrinsic `__nv_double2ull_rd` which
    performs conversion with round-down (towards negative infinity) rounding mode.

    The behavior for negative input values follows unsigned integer conversion
    semantics (two's complement representation). Out-of-range values result in
    undefined behavior.

    This function is pure and has no side effects.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(axis=0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offs).to(tl.float64)
         y = tl.extra.cuda.libdevice.double2ull_rd(x)
         tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.double2ull_rn

```python
double2ull_rn(arg0, _semantic=None)
```

.. py::function:: double2ull_rn(arg0, _semantic=None)

   Convert double-precision floating-point value to unsigned 64-bit integer using round-to-nearest rounding.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of double-precision floating-point values (fp64).
   _semantic : optional
       Internal parameter for Triton semantic analysis. Do not set manually.

   Returns
   -------
   tl.tensor
       Tensor of unsigned 64-bit integers (int64) with the same shape as `arg0`.

   Notes
   -----
   This function invokes the CUDA libdevice intrinsic `__nv_double2ull_rn` which
   converts double-precision floating-point values to unsigned 64-bit integers
   using round-to-nearest-even rounding mode.

   Behavior for special values:
   
   - NaN inputs return 0
   - Infinity inputs return 0
   - Values outside the representable range of uint64 return 0

   This is a pure function with no side effects.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       offs = pid * BLOCK + tl.arange(0, BLOCK)
       x = tl.load(x_ptr + offs).to(tl.float64)
       y = tl.extra.cuda.libdevice.double2ull_rn(x)
       tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.double2ull_ru

```python
double2ull_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2ull_ru

Convert a double-precision floating-point value to an unsigned 64-bit integer with round toward zero.

### Parameters
arg0 : tensor of fp64
    Input double-precision floating-point value(s) to convert.
_semantic : optional
    Internal semantic argument; do not set manually.

### Returns
tensor of int64
    Unsigned 64-bit integer representation of the input values.

### Notes
This function wraps the CUDA libdevice intrinsic `__nv_double2ull_ru`, which
converts double-precision floating-point values to unsigned long long integers
using round toward zero (truncate) rounding mode.

The function is pure and can be safely used in device code without side effects.
Only `fp64` input type is supported.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def convert_kernel(x_ptr, y_ptr, n):
     pid = tl.program_id(0)
     offset = pid * 64
     x = tl.load(x_ptr + offset + tl.arange(0, 64))
     y = tl.extra.cuda.libdevice.double2ull_ru(x)
     tl.store(y_ptr + offset + tl.arange(0, 64), y)
```

---

### triton.language.extra.cuda.libdevice.double2ull_rz

```python
double2ull_rz(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.double2ull_rz


**`double2ull_rz(arg0, _semantic=None)`**

   Convert double-precision floating-point values to unsigned 64-bit integers with round-towards-zero.

   Parameters
   ----------
   arg0 : tensor of fp64
       Input tensor containing double-precision floating-point values to convert.
   _semantic : optional
       Internal semantic argument for Triton compiler. Do not set manually.

   Returns
   -------
   tensor of int64
       Tensor of unsigned 64-bit integers resulting from the conversion.

   Notes
   -----
   This function wraps the CUDA libdevice intrinsic `__nv_double2ull_rz`. The conversion uses round-towards-zero (truncate) rounding mode, equivalent to C-style casting from double to unsigned long long.

   The function is pure (no side effects) and operates element-wise on the input tensor.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)
       # Convert fp64 to int64 with round-towards-zero
       y = tl.extra.cuda.libdevice.double2ull_rz(x)
       tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.double_as_longlong

```python
double_as_longlong(arg0, _semantic=None)
```

## double_as_longlong


**`double_as_longlong(arg0, _semantic=None)`**

    Reinterpret the bit pattern of a float64 value as a 64-bit signed integer.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of type `tl.float64`.

    Returns
    -------
    tl.tensor
        Tensor of type `tl.int64` with the same bit representation as `arg0`.

    Notes
    -----
    This is a bitcast operation, not a numeric conversion. The IEEE 754 binary
    representation of the floating-point value is reinterpreted as a two's
    complement integer without changing the underlying bits.

    This function maps to the CUDA libdevice function `__nv_double_as_longlong`.
    It is equivalent to type punning in C (e.g., `*(long long*)&double_value`).

    The operation is pure and has no side effects.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offset)  # fp64 tensor
         y = tl.extra.cuda.libdevice.double_as_longlong(x)  # int64 tensor
         tl.store(y_ptr + offset, y)

     # Launch kernel with fp64 input
     x = torch.randn(1024, dtype=torch.float64, device='cuda')
     y = torch.empty(1024, dtype=torch.int64, device='cuda')
     kernel[(1,)](x, y, BLOCK_SIZE=1024)
```

---

### triton.language.extra.cuda.libdevice.erf

```python
erf(arg0, _semantic=None)
```

Compute the element-wise error function of the input tensor.

### Parameters
arg0 : tensor
    Input tensor of floating-point type. Supported dtypes are `fp32` and
    `fp64`.

### Returns
tensor
    Tensor of the same dtype as the input, containing the error function
    values element-wise.

### Notes
The error function is defined as:

.. math::

    \\text{erf}(x) = \\frac{2}{\\sqrt{\\pi}} \\int_0^x e^{-t^2} dt

This function dispatches to CUDA libdevice implementations:

- `__nv_erff` for 32-bit floating-point (`fp32`)
- `__nv_erf` for 64-bit floating-point (`fp64`)

The operation is pure (has no side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(X_ptr + offsets)
     y = tl.extra.cuda.libdevice.erf(x)
     tl.store(Y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.erfc

```python
erfc(arg0, _semantic=None)
```

Compute the complementary error function element-wise.

Calculates $1 - \text{erf}(x)$ for each element in the input tensor.

### Parameters
arg0 : tensor
    Input tensor of floating point type. Supports `fp32` and `fp64`.

### Returns
tensor
    Output tensor of the same type as `arg0`.

### Notes
This function is only available on CUDA targets. It invokes the CUDA
libdevice functions `__nv_erfcf` for `fp32` inputs and
`__nv_erfc` for `fp64` inputs.

### Examples
```python
 @triton.jit
 def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK + tl.arange(0, BLOCK)
     mask = offs < n
     x = tl.load(x_ptr + offs, mask=mask)
     y = tl.extra.cuda.libdevice.erfc(x)
     tl.store(y_ptr + offs, y, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.erfcinv

```python
erfcinv(arg0, _semantic=None)
```

## erfcinv

Compute the inverse complementary error function element-wise.

### Parameters
arg0 : tensor
    Input tensor of floating-point type (fp32 or fp64).
_semantic : optional
    Internal semantic argument used by the Triton compiler. Do not set manually.

### Returns
tensor
    Tensor of the same dtype as `arg0` containing the inverse complementary
    error function values.

### Notes
The inverse complementary error function `erfcinv(x)` computes the value
`y` such that `erfc(y) = x`, where `erfc` is the complementary error
function defined as `erfc(x) = 1 - erf(x)`.

Valid input range is (0, 2). For `x` outside this range, the result is
undefined.

This function is implemented using CUDA libdevice intrinsics:
- `__nv_erfcinvf` for 32-bit floating-point (fp32)
- `__nv_erfcinv` for 64-bit floating-point (fp64)

The operation is pure (no side effects) and supports automatic broadcasting.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.erfcinv(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.erfcx

```python
erfcx(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.erfcx


.. autofunction:: erfcx

Compute the scaled complementary error function.

The scaled complementary error function is defined as
$\text{erfcx}(x) = e^{x^2} \cdot \text{erfc}(x)$, where
$\text{erfc}(x) = 1 - \text{erf}(x)$ is the complementary error
function. This scaling avoids overflow for large positive values of
$x$.

### Parameters
arg0 : tensor
    Input tensor. Must be of floating-point type (`fp32` or `fp64`).

### Returns
tensor
    Output tensor of the same dtype as `arg0`, containing the scaled
    complementary error function values.

### Notes
This function dispatches to CUDA libdevice intrinsics:

- `__nv_erfcxf` for `fp32` inputs
- `__nv_erfcx` for `fp64` inputs

The function is pure (no side effects) and supports element-wise
operation on tensors.

For large positive $x$, :math:`\text{erfcx}(x) \approx
\frac{1}{\sqrt{\pi} \cdot x}`, which avoids the overflow that would
occur when computing $e^{x^2}$ and $\text{erfc}(x)$
separately.

### Examples
```python
 import triton
 import triton.language as tl
 import torch

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.erfcx(x)
     tl.store(y_ptr + offsets, y)

 # Example usage
 BLOCK_SIZE = 1024
 x = torch.randn(BLOCK_SIZE, dtype=torch.float32, device='cuda')
 y = torch.empty_like(x)
 kernel[(1,)](x, y, BLOCK_SIZE=BLOCK_SIZE)
```

---

### triton.language.extra.cuda.libdevice.erfinv

```python
erfinv(arg0, _semantic=None)
```

## erfinv

Compute the inverse error function element-wise.

### Parameters
arg0 : tensor
    Input tensor of floating-point type (fp32 or fp64).
_semantic : optional
    Internal parameter for Triton semantics. Do not set manually.

### Returns
tensor
    Tensor of the same dtype as `arg0` containing the inverse error function
    values.

### Notes
This function is implemented using CUDA libdevice intrinsics:

- `__nv_erfinvf` for 32-bit floating-point (fp32)
- `__nv_erfinv` for 64-bit floating-point (fp64)

The inverse error function `erfinv(x)` is the inverse of the error function
`erf(x)`. The domain of `erfinv` is `[-1, 1]`, and the range is
`(-inf, inf)`. Values outside the domain `[-1, 1]` may return NaN.

This is a pure function with no side effects.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(X_ptr + offsets)
     y = tl.extra.cuda.libdevice.erfinv(x)
     tl.store(Y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.exp

```python
exp(arg0, _semantic=None)
```

## exp

Compute the natural exponential of all elements in the input tensor.

### Parameters
arg0 : tensor
    Input tensor. Must have floating-point dtype (`fp32` or `fp64`).

### Returns
tensor
    Element-wise exponential of the input. Has the same dtype as `arg0`.

### Notes
This function dispatches to NVIDIA libdevice's `__nv_expf` for `fp32`
inputs and `__nv_exp` for `fp64` inputs. The operation is pure
(no side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def exp_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.exp(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.exp10

```python
exp10(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.exp10


**`exp10(arg0, _semantic=None)`**

    Compute the base-10 exponential $10^x$ element-wise.

    Parameters
    ----------
    arg0 : tensor
        Input tensor of floating-point type (fp32 or fp64).
    _semantic : optional
        Internal parameter for semantic propagation. Do not set manually.

    Returns
    -------
    tensor
        Tensor of the same dtype as `arg0` containing $10^{arg0}$.

    Notes
    -----
    This function calls CUDA libdevice routines:

    - `__nv_exp10f` for 32-bit floating-point (fp32)
    - `__nv_exp10` for 64-bit floating-point (fp64)

    Only available on CUDA devices with libdevice support. The operation is
    pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def exp10_kernel(x_ptr, out_ptr, n):
         pid = tl.program_id(0)
         offset = pid * 256 + tl.arange(0, 256)
         mask = offset < n
         x = tl.load(x_ptr + offset, mask=mask)
         out = tl.extra.cuda.libdevice.exp10(x)
         tl.store(out_ptr + offset, out, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.exp2

```python
exp2(arg0, _semantic=None)
```

## exp2

Compute 2^x element-wise.

### Parameters
arg0 : tensor
    Input tensor of floating-point values (fp32 or fp64).
_semantic : optional
    Internal semantic parameter (do not set manually).

### Returns
tensor
    Tensor of same shape and dtype as `arg0` containing 2^arg0 for each element.

### Notes
Dispatches to CUDA libdevice functions:

- `__nv_exp2f` for `fp32` inputs
- `__nv_exp2` for `fp64` inputs

The operation is pure (no side effects). Only supported on CUDA devices.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def exp2_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.exp2(x)  # Computes 2^x
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.expm1

```python
expm1(arg0, _semantic=None)
```

## expm1(arg0, _semantic=None)

Compute $e^{arg0} - 1$ element-wise.

### Parameters
arg0 : tensor
    Input tensor of floating-point values. Supports `fp32` and `fp64` dtypes.

### Returns
tensor
    Tensor of the same shape as `arg0` containing $e^{arg0} - 1$.
    Return dtype matches input dtype.

### Notes
This function provides better numerical accuracy than computing `exp(arg0) - 1`
directly for small values of `arg0`, as it avoids catastrophic cancellation
when `arg0` is near zero.

On CUDA GPUs, this function maps to the libdevice functions:

- `__nv_expm1f` for `fp32` inputs
- `__nv_expm1` for `fp64` inputs

The function is pure (no side effects) and can be safely used in any context.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def expm1_kernel(X, Y, BLOCK: tl.constexpr):
     x = tl.load(X + tl.arange(0, BLOCK))
     y = tl.expm1(x)
     tl.store(Y + tl.arange(0, BLOCK), y)
```

---

### triton.language.extra.cuda.libdevice.fast_cosf

```python
fast_cosf(arg0, _semantic=None)
```

## fast_cosf

Compute the cosine of the input element-wise using a fast approximation.

This function invokes CUDA libdevice's `__nv_fast_cosf` intrinsic, providing
a faster but less accurate alternative to :py`tl.cos()`.

### Parameters
arg0 : tensor
    Input tensor of float32 values.

### Returns
out : tensor
    Tensor of float32 values containing the cosine of each element in `arg0`.
    The output has the same shape as the input.

### Notes
This function is only available on CUDA targets. The input must be of type
float32; other dtypes are not supported. The fast cosine approximation may
have reduced precision compared to :py`tl.cos()` but can offer better
performance on some GPU architectures.

The function is pure (has no side effects) and can be safely used in
compiler optimizations.

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     # Compute fast cosine
     y = libdevice.fast_cosf(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.fast_dividef

```python
fast_dividef(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.fast_dividef


**`fast_dividef(arg0, arg1, _semantic=None)`**

    Fast floating-point division for 32-bit floats.

    Performs division `arg0 / arg1` using CUDA's fast math approximation
    (`__nv_fast_fdividef`). This is faster than standard division but may
    have reduced precision.

    Parameters
    ----------
    arg0 : tl.tensor
        Numerator. Must be 32-bit floating point (`tl.float32`).
    arg1 : tl.tensor
        Denominator. Must be 32-bit floating point (`tl.float32`).
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    result : tl.tensor
        The quotient `arg0 / arg1` as 32-bit floating point.

    Notes
    -----
    This function is CUDA-specific and uses the libdevice intrinsic
    `__nv_fast_fdividef`. The fast division approximation provides
    improved performance at the cost of numerical precision compared
    to standard floating-point division.

    Both inputs must be `tl.float32` dtype. Behavior for special
    values (NaN, Inf, zero) follows IEEE-754 fast math semantics.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def fast_div_kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = tl.load(y_ptr + offsets)
         # Fast division (less precise than x / y)
         result = tl.extra.cuda.libdevice.fast_dividef(x, y)
         tl.store(out_ptr + offsets, result)
```

---

### triton.language.extra.cuda.libdevice.fast_exp10f

```python
fast_exp10f(arg0, _semantic=None)
```

## fast_exp10f


**`fast_exp10f(arg0, _semantic=None)`**

    Compute the base-10 exponential function $10^x$ for float32 values.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor. Must be of type `tl.float32`.
    _semantic : optional
        Internal parameter used by Triton compiler. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Output tensor of type `tl.float32` containing $10^{arg0}$.

    Notes
    -----
    This function wraps the CUDA libdevice intrinsic `__nv_fast_exp10f`.
    It provides a fast approximation of the base-10 exponential operation.
    Only `fp32` input and output types are supported.

    This is a pure function with no side effects.

    Example
    -------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def exp10_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = libdevice.fast_exp10f(x)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.fast_expf

```python
fast_expf(arg0, _semantic=None)
```

fast_expf(arg0, _semantic=None)
    Compute the exponential function e^x with reduced accuracy but improved performance.

    Parameters
    ----------
    arg0 : tensor
        Input tensor of type `tl.float32`.
    _semantic
        Internal parameter used by Triton compiler. Do not set manually.

    Returns
    -------
    result : tensor
        Output tensor of type `tl.float32` containing e^arg0.

    Notes
    -----
    This function uses the CUDA libdevice `__nv_fast_expf` intrinsic, which provides
    a fast approximation of the exponential function with lower accuracy than
    :py`tl.exp()` but better performance. The maximum error is approximately 2 ulps.

    Only supports `float32` input and output types. This function is pure (no side
    effects) and can be safely used in compiler optimizations.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def exp_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offs)
         y = libdevice.fast_expf(x)
         tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.fast_log10f

```python
fast_log10f(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.fast_log10f

**`fast_log10f(arg0, _semantic=None)`**

    Compute the base-10 logarithm of `arg0` approximately.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of type `tl.float32`.

    Returns
    -------
    tl.tensor
        Tensor of type `tl.float32` containing the base-10 logarithm of each element.

    Notes
    -----
    This is a fast approximation of `log10` using the CUDA libdevice function
    `__nv_fast_log10f`. The result may have reduced precision compared to
    `tl.libdevice.log10`. Only supports `fp32` input type.

    This function is pure (no side effects).

    Example
    -------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.extra.cuda.libdevice.fast_log10f(x)
         tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.fast_log2f

```python
fast_log2f(arg0, _semantic=None)
```

## fast_log2f


**`fast_log2f(arg0, _semantic=None)`**

    Compute the base-2 logarithm of the input element-wise.

    This is a fast approximation of $log_2$ using CUDA libdevice.

    Parameters
    ----------
    arg0 : tensor
        Input tensor of floating-point values. Must be `fp32` dtype.

    Returns
    -------
    tensor
        Tensor of the same shape as `arg0` containing the base-2 logarithm
        of each element. Returns `fp32` dtype.

    Notes
    -----
    This function calls the CUDA libdevice function `__nv_fast_log2f` which
    provides a faster but less accurate approximation compared to the standard
    `log2` function.

    The function is pure (no side effects) and operates element-wise.

    Only `fp32` input dtype is supported.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def log2_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK + tl.arange(0, BLOCK)
         mask = offset < n
         x = tl.load(x_ptr + offset, mask=mask)
         y = tl.extra.cuda.libdevice.fast_log2f(x)
         tl.store(y_ptr + offset, y, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.fast_logf

```python
fast_logf(arg0, _semantic=None)
```

## fast_logf

Compute the natural logarithm of each element in the input tensor using a fast
approximation.

### Parameters
arg0 : tensor
    Input tensor of type `tl.float32`.
_semantic :
    Internal parameter, do not use.

### Returns
tensor
    Output tensor of type `tl.float32` containing the natural logarithm
    of each element in `arg0`.

### Notes
This function maps to the `__nv_fast_logf` PTX instruction from the CUDA
libdevice library. It provides a fast approximation of the natural logarithm
for 32-bit floating-point values.

Only `fp32` input and output types are supported. This function is pure
(no side effects).

For higher precision logarithm computation, consider using `tl.log` instead.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def log_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.fast_logf(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.fast_powf

```python
fast_powf(arg0, arg1, _semantic=None)
```

## fast_powf

Compute `arg0` raised to the power of `arg1` using a fast approximation.

### Parameters
arg0 : tl.tensor
    Base value. Must be float32.
arg1 : tl.tensor
    Exponent value. Must be float32.

### Returns
result : tl.tensor
    Result of `arg0 ** arg1`. Float32.

### Notes
This function uses NVIDIA's libdevice `__nv_fast_powf` intrinsic for faster
computation at the cost of reduced precision compared to standard power
operations. Only supports float32 inputs and outputs. The operation is pure
(no side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(X, Y, BLOCK: tl.constexpr):
     offsets = tl.arange(0, BLOCK)
     x = tl.load(X + offsets)
     y = tl.load(Y + offsets)
     result = tl.extra.cuda.libdevice.fast_powf(x, y)
     tl.store(X + offsets, result)
```

---

### triton.language.extra.cuda.libdevice.fast_sinf

```python
fast_sinf(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.fast_sinf


.. autofunction:: fast_sinf

Compute the sine of the input element-wise (fast approximation).

This function calls the CUDA libdevice `__nv_fast_sinf` intrinsic, which provides
a fast approximation of the sine function with reduced precision compared to the
standard `sin` operation.

### Parameters
arg0 : tl.tensor
    Input tensor of type `tl.float32`. The sine is computed element-wise.
_semantic : optional
    Internal semantic argument used by Triton compiler. Do not set manually.

### Returns
result : tl.tensor
    Tensor of type `tl.float32` containing the sine of each input element.

### Notes
This is a fast approximation of the sine function. For higher precision, use
:py`triton.language.sin()` instead. The input must be of type `fp32`.

This function is pure (has no side effects) and can be safely used in
constant expressions and compiler optimizations.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.fast_sinf(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.fast_tanf

```python
fast_tanf(arg0, _semantic=None)
```

## fast_tanf

Compute the tangent of the input element-wise using a fast approximation.

### Parameters
arg0 : tensor
    Input tensor of floating-point values. Must be `fp32` dtype.
_semantic : optional
    Internal parameter for Triton semantics. Do not set manually.

### Returns
tensor
    Output tensor of `fp32` dtype containing the tangent of each element.

### Notes
This function calls the CUDA libdevice `__nv_fast_tanf` intrinsic, providing
a fast approximation of the tangent function with reduced precision compared
to the standard `tan` operation.

The function is marked as pure (no side effects) and supports only `fp32`
input and output types.

For higher precision tangent computation, use :py`triton.language.tan()`
if available for your target architecture.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def tan_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.fast_tanf(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.fdim

```python
fdim(arg0, arg1, _semantic=None)
```

fdim(arg0, arg1, _semantic=None)

    Returns the positive difference between two arguments.

    Computes `max(arg0 - arg1, 0)` element-wise.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be floating-point type (fp32 or fp64).
    arg1 : tl.tensor
        Second input tensor. Must be floating-point type (fp32 or fp64).
        Must have the same dtype as arg0.
    _semantic : optional
        Internal parameter for semantic propagation. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of the positive differences. Has the same dtype as the inputs.

    Notes
    -----
    This function is an extern wrapper for CUDA libdevice functions:

    - `__nv_fdimf` for fp32 inputs
    - `__nv_fdim` for fp64 inputs

    The function is pure (no side effects) and supports element-wise operations
    on tensors.

    Examples
    --------
```python
     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         y = tl.load(y_ptr + tl.arange(0, BLOCK))
         # Computes max(x - y, 0) for each element
         result = tl.extra.cuda.libdevice.fdim(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK), result)
```

---

### triton.language.extra.cuda.libdevice.ffs

```python
ffs(arg0, _semantic=None)
```

## ffs

Find the position of the first set bit.


**`ffs(arg0, _semantic=None)`**

   Returns the position of the first (least significant) bit set to 1 in the input.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of integer type (int32 or int64).

   Returns
   -------
   ret : tl.tensor
       Tensor of int32 containing the bit position. Returns 0 if the input is 0,
       otherwise returns a value in the range [1, bitwidth].

   Notes
   -----
   This function wraps the CUDA libdevice functions `__nv_ffs` for int32 and
   `__nv_ffsll` for int64. The result is 1-indexed (the least significant bit
   is position 1, not 0).

   The function is pure and has no side effects.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        # Find first set bit position
        ffs_result = tl.extra.cuda.libdevice.ffs(x)
        tl.store(out_ptr + offsets, ffs_result)
```

---

### triton.language.extra.cuda.libdevice.finitef

```python
finitef(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.finitef


**`finitef(arg0, _semantic=None)`**

    Test whether a floating-point value is finite.

    Returns 1 if `arg0` is finite (not infinity or NaN), and 0 otherwise.
    This function wraps the CUDA libdevice function `__nv_finitef`.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values (fp32).
    _semantic : optional
        Internal semantic argument; do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of int1 (boolean) values. Each element is 1 if the corresponding
        input element is finite, and 0 otherwise.

    Notes
    -----
    This function is pure (has no side effects) and operates elementwise on
    the input tensor. It is typically used within `@triton.jit` decorated
    kernel functions.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         # Check if values are finite
         is_finite = tl.extra.cuda.libdevice.finitef(x)
         tl.store(out_ptr + offsets, is_finite)
```

---

### triton.language.extra.cuda.libdevice.float2int_rd

```python
float2int_rd(arg0, _semantic=None)
```

## float2int_rd


.. autofunction:: float2int_rd

Converts a floating-point value to a signed integer by rounding towards negative infinity.

### Parameters
arg0 : tl.tensor
    Input tensor of 32-bit floating-point values (`tl.float32`).

### Returns
result : tl.tensor
    Tensor of 32-bit signed integers (`tl.int32`) with the same shape as `arg0`.

### Notes
This function wraps the CUDA libdevice function `__nv_float2int_rd` which performs
rounding towards negative infinity (floor rounding). For positive values, this behaves
like truncation. For negative non-integer values, this rounds away from zero
(e.g., `-3.7` becomes `-4`).

This operation is only available on CUDA devices and compiles to the PTX
`cvt.rni.s32.f32` instruction with round-down modifier.

The function is pure (no side effects) and operates element-wise on tensors.

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(axis=0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offs)
     # Round towards negative infinity
     y = libdevice.float2int_rd(x)
     tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.float2int_rn

```python
float2int_rn(arg0, _semantic=None)
```

## float2int_rn


.. autofunction:: float2int_rn

Converts floating-point values to signed integers using round-to-nearest-even rounding mode.

### Parameters
arg0 : tl.tensor
    Input tensor of floating-point values. Must have dtype `tl.float32`.

_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
out : tl.tensor
    Tensor of signed 32-bit integers (`tl.int32`). Each element is the input
    value rounded to the nearest integer, with ties rounded to even.

### Notes
This function wraps the CUDA libdevice intrinsic `__nv_float2int_rn`. The
rounding behavior follows IEEE 754 round-to-nearest-even semantics:

- Values are rounded to the nearest representable integer
- When exactly halfway between two integers, the even integer is chosen
- Values outside the representable int32 range produce undefined behavior

This operation is pure (no side effects) and can be safely used in
device-side computations.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     # Convert float32 to int32 with round-to-nearest
     out = tl.extra.cuda.libdevice.float2int_rn(x)
     tl.store(out_ptr + offsets, out)
```

---

### triton.language.extra.cuda.libdevice.float2int_ru

```python
float2int_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.float2int_ru


**`float2int_ru(arg0, _semantic=None)`**

    Convert floating-point values to 32-bit integers with round-up rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values (fp32).

    Returns
    -------
    tl.tensor
        Tensor of 32-bit signed integers (int32) with values rounded towards
        positive infinity.

    Notes
    -----
    This function wraps the CUDA libdevice intrinsic `__nv_float2int_ru`, which
    performs floating-point to integer conversion with round-up (towards positive
    infinity) rounding mode. For positive values, this behaves like ceiling.
    For negative values, it truncates towards zero.

    The operation is pure (no side effects) and compiles to a single GPU
    instruction on CUDA devices.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offs)
         # Convert float to int with round-up rounding
         y = tl.extra.cuda.libdevice.float2int_ru(x)
         tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.float2int_rz

```python
float2int_rz(arg0, _semantic=None)
```

## float2int_rz


**`float2int_rz(arg0, _semantic=None)`**

   Convert a floating-point value to a signed integer using round-towards-zero.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of float32 values to convert.
   _semantic : optional
       Internal semantic argument, do not set manually.

   Returns
   -------
   tl.tensor
       Tensor of int32 values with the same shape as `arg0`.

   Notes
   -----
   This function calls the CUDA libdevice function `__nv_float2int_rz` which
   converts floating-point values to integers using round-towards-zero semantics.
   Fractional parts are truncated toward zero (e.g., 3.9 becomes 3, -3.9 becomes -3).

   The operation is pure (no side effects) and supports elementwise broadcasting.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, out_ptr, n):
       pid = tl.program_id(0)
       offset = pid * 32
       x = tl.load(x_ptr + offset + tl.arange(0, 32))
       out = tl.extra.cuda.libdevice.float2int_rz(x)
       tl.store(out_ptr + offset + tl.arange(0, 32), out)
```

---

### triton.language.extra.cuda.libdevice.float2ll_rd

```python
float2ll_rd(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.float2ll_rd

Convert a 32-bit floating-point value to a 64-bit signed integer with round-down rounding mode.

### Parameters
arg0 : tl.tensor
    Input tensor of 32-bit floating-point values (`tl.float32`).

_semantic : optional
    Internal semantic argument; do not set manually.

### Returns
tl.tensor
    Tensor of 64-bit signed integers (`tl.int64`) with the same shape as `arg0`.

### Notes
This function wraps the CUDA libdevice intrinsic `__nv_float2ll_rd`, which converts
floating-point values to signed 64-bit integers using round-down (toward negative
infinity) rounding mode.

The function is pure (no side effects) and operates elementwise on the input tensor.

Behavior for special values:
- NaN inputs produce undefined results
- Infinity inputs produce undefined results
- Values outside the int64 representable range produce undefined results

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.float2ll_rd(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.float2ll_rn

```python
float2ll_rn(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.float2ll_rn


.. autofunction:: float2ll_rn

Convert a 32-bit floating-point value to a 64-bit signed integer using round-to-nearest-even rounding mode.

### Parameters
arg0 : tensor of float32
    Input floating-point value(s) to convert.

### Returns
tensor of int64
    Converted integer value(s). Each input element is rounded to the nearest integer, with ties rounded to even.

### Notes
This function wraps the CUDA libdevice function `__nv_float2ll_rn`. The rounding mode is round-to-nearest-even (RN), which is the default IEEE 754 rounding mode.

For values outside the representable range of int64, the result is undefined. NaN inputs produce undefined results.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     # Convert float32 to int64 with round-to-nearest
     y = tl.extra.cuda.libdevice.float2ll_rn(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.float2ll_ru

```python
float2ll_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.float2ll_ru


**`float2ll_ru(arg0, _semantic=None)`**

    Convert a 32-bit floating-point value to a 64-bit signed integer using round-to-nearest-even rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of 32-bit floating-point values (`tl.float32`).
    _semantic : optional
        Internal parameter for semantic analysis. Do not set manually.

    Returns
    -------
    out : tl.tensor
        Tensor of 64-bit signed integers (`tl.int64`) with the same shape as `arg0`.

    Notes
    -----
    This function wraps the CUDA libdevice intrinsic `__nv_float2ll_ru`. The `ru` suffix indicates round-to-nearest-even rounding mode (also known as "round to nearest, ties to even" or IEEE 754 default rounding).

    Values outside the representable range of 64-bit signed integers result in undefined behavior. NaN and infinity inputs also result in undefined behavior.

    This is a pure function with no side effects.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def convert_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         # Convert float32 to int64 with round-to-nearest-even
         out = tl.extra.cuda.libdevice.float2ll_ru(x)
         tl.store(out_ptr + offsets, out)
```

---

### triton.language.extra.cuda.libdevice.float2ll_rz

```python
float2ll_rz(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.float2ll_rz

Convert 32-bit floating-point values to 64-bit signed integers with round-towards-zero rounding.

### Parameters
arg0 : tl.tensor
    Input tensor of 32-bit floating-point values (`tl.float32`).

### Returns
tl.tensor
    Tensor of 64-bit signed integers (`tl.int64`) with the same shape as
    `arg0`.

### Notes
This function invokes the CUDA libdevice intrinsic `__nv_float2ll_rz`. The
rounding mode is "round towards zero" (rz), which truncates the fractional part
of the floating-point value. For positive numbers this rounds down, for negative
numbers this rounds up towards zero.

This is a pure external function that compiles directly to GPU machine code via
the CUDA libdevice library. Behavior is undefined for input values outside the
representable range of 64-bit signed integers ([-2^63, 2^63-1]).

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def convert_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets).to(tl.float32)
     y = libdevice.float2ll_rz(x)
     tl.store(y_ptr + offsets, y)

 # Launch kernel
 BLOCK_SIZE = 1024
 x = torch.randn(BLOCK_SIZE, dtype=torch.float32, device='cuda')
 y = torch.empty(BLOCK_SIZE, dtype=torch.int64, device='cuda')
 convert_kernel[(1,)](x, y, BLOCK_SIZE=BLOCK_SIZE)
```

---

### triton.language.extra.cuda.libdevice.float2uint_rd

```python
float2uint_rd(arg0, _semantic=None)
```

## float2uint_rd

Convert floating-point values to unsigned integers using round-down rounding mode.

### Parameters
arg0 : tensor
    Input tensor of 32-bit floating-point values (`tl.float32`).
_semantic : optional
    Internal semantic argument, typically not provided by users.

### Returns
tensor
    Tensor of 32-bit unsigned integers (`tl.int32`).

### Notes
This function invokes the CUDA libdevice intrinsic `__nv_float2uint_rd`, which
converts each floating-point element to an unsigned integer by rounding toward
negative infinity (round-down mode). Values outside the representable range of
`uint32` produce undefined behavior.

The operation is element-wise and pure (no side effects).

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def float_to_uint_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = libdevice.float2uint_rd(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.float2uint_rn

```python
float2uint_rn(arg0, _semantic=None)
```

## float2uint_rn


**`float2uint_rn(arg0, _semantic=None)`**

    Convert 32-bit floating-point values to 32-bit signed integers using round-to-nearest-even rounding.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of 32-bit floating-point values (`tl.float32`).

    Returns
    -------
    out : tl.tensor
        Output tensor of 32-bit signed integers (`tl.int32`).

    Notes
    -----
    This function wraps the CUDA libdevice function `__nv_float2uint_rn`.
    The conversion uses IEEE 754 round-to-nearest-even rounding mode.
    Out-of-range values produce undefined behavior.
    
    This is a pure function with no side effects, suitable for use in JIT-compiled kernels.
    The `_semantic` parameter is internal and should not be provided by users.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK
         x = tl.load(x_ptr + offset)
         out = tl.extra.cuda.libdevice.float2uint_rn(x)
         tl.store(out_ptr + offset, out)
```

---

### triton.language.extra.cuda.libdevice.float2uint_ru

```python
float2uint_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.float2uint_ru


**`float2uint_ru(arg0, _semantic=None)`**

   Convert 32-bit floating-point values to 32-bit unsigned integers using round-to-nearest-even rounding.

   Calls the CUDA libdevice function `__nv_float2uint_ru` which converts float32 to int32 with rounding toward nearest unsigned integer.

   Parameters
   ----------
   arg0 : tensor
       Input tensor of 32-bit floating-point values (`tl.float32`).

   _semantic : optional
       Internal semantic parameter. Do not set manually.

   Returns
   -------
   tensor
       Tensor of 32-bit signed integers (`tl.int32`) containing the converted values.

   Notes
   -----
   This function is a thin wrapper around CUDA's libdevice intrinsic. It is only available on CUDA targets.

   The `ru` suffix indicates round-to-nearest-even rounding mode for unsigned conversion.

   This operation is pure (no side effects) and can be safely used in device code.

   Examples
   --------
```python
   import triton
   import triton.language as tl
   from triton.language.extra.cuda import libdevice

   @triton.jit
   def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       x = tl.load(x_ptr + offsets)
       # Convert float32 to int32 with rounding
       out = libdevice.float2uint_ru(x)
       tl.store(out_ptr + offsets, out)
```

---

### triton.language.extra.cuda.libdevice.float2uint_rz

```python
float2uint_rz(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.float2uint_rz


**`float2uint_rz(arg0, _semantic=None)`**

    Convert a 32-bit floating-point value to a 32-bit unsigned integer using round-towards-zero rounding.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of 32-bit floating-point values (`tl.float32`).
    _semantic : optional
        Internal semantic argument used by Triton compiler. Do not set manually.

    Returns
    -------
    tl.tensor
        Tensor of 32-bit unsigned integers (`tl.int32`) with the same shape as `arg0`.

    Notes
    -----
    This function wraps the CUDA libdevice intrinsic `__nv_float2uint_rz` which performs
    floating-point to unsigned integer conversion with round-towards-zero (truncation) semantics.
    Fractional components are discarded. Values outside the representable range of `int32`
    produce undefined behavior.

    This is a device-only function that can only be called from within `@triton.jit`
    decorated kernels.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         # Convert float32 to int32 with round-towards-zero
         y = tl.extra.cuda.libdevice.float2uint_rz(x)
         tl.store(out_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.float2ull_rd

```python
float2ull_rd(arg0, _semantic=None)
```

## float2ull_rd

Convert a 32-bit floating-point value to an unsigned 64-bit integer with round-down rounding mode.

### Parameters
arg0 : tl.tensor
    Input tensor of float32 values to convert.
_semantic : optional
    Internal semantic argument, do not set manually.

### Returns
tl.tensor
    Tensor of int64 (uint64) values representing the input floats converted with round-down rounding.

### Notes
This function wraps the CUDA libdevice function `__nv_float2ull_rd` which performs float-to-unsigned-long-long
conversion with round-down (toward negative infinity) rounding mode. The conversion behavior for negative inputs
or values outside the representable range of uint64 is undefined.

This is an extern function that compiles directly to PTX code on NVIDIA GPUs.

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offs).to(tl.float32)
     y = libdevice.float2ull_rd(x)
     tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.float2ull_rn

```python
float2ull_rn(arg0, _semantic=None)
```

## float2ull_rn


**`float2ull_rn(arg0, _semantic=None)`**

    Convert a 32-bit floating-point value to an unsigned 64-bit integer with round-to-nearest rounding.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of 32-bit floating-point values (`tl.float32`).
    _semantic : optional
        Internal parameter for semantic propagation. Do not set manually.

    Returns
    -------
    out : tl.tensor
        Tensor of unsigned 64-bit integers (`tl.int64`).

    Notes
    -----
    This function calls the CUDA libdevice function `__nv_float2ull_rn`.
    The conversion uses round-to-nearest-even rounding mode.

    Values outside the representable range of uint64 (0 to 2^64-1) may produce
    undefined results. NaN and infinity inputs produce undefined behavior.

    This function is pure (no side effects) and can be used in JIT-compiled
    device code.

    The function is only available on CUDA targets with libdevice support.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets).to(tl.float32)
         out = tl.extra.cuda.libdevice.float2ull_rn(x)
         tl.store(out_ptr + offsets, out)
```

---

### triton.language.extra.cuda.libdevice.float2ull_ru

```python
float2ull_ru(arg0, _semantic=None)
```

## float2ull_ru

Convert a floating-point value to an unsigned 64-bit integer with round-up rounding.

### Parameters
arg0 : tl.tensor
    Input tensor of float32 values.
_semantic : optional
    Internal semantic argument, should not be provided by users.

### Returns
tl.tensor
    Tensor of int64 values, converted from float32 with round-up (toward positive
    infinity) rounding mode.

### Notes
This function uses the CUDA libdevice intrinsic `__nv_float2ull_ru` which converts
float32 to unsigned 64-bit integer with round toward positive infinity rounding mode.
The function is marked as pure, meaning it has no side effects.

This is an external function that compiles down to GPU machine code. It must be called
from within a `@triton.jit` decorated kernel function.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(ptr + offsets).to(tl.float32)
     # Convert float32 to uint64 with round-up rounding
     y = tl.extra.cuda.libdevice.float2ull_ru(x)
     tl.store(ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.float2ull_rz

```python
float2ull_rz(arg0, _semantic=None)
```

## float2ull_rz


**`float2ull_rz(arg0, _semantic=None)`**

   Convert a 32-bit floating-point value to an unsigned 64-bit integer using round-towards-zero.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of 32-bit floating-point values (`tl.float32`).

   Returns
   -------
   tl.tensor
       Tensor of unsigned 64-bit integers (`tl.int64`).

   Notes
   -----
   This function wraps the CUDA libdevice intrinsic `__nv_float2ull_rz`. The conversion uses
   round-towards-zero (RTZ) rounding mode, truncating the fractional part of the input.

   Behavior is undefined for:

   - Negative values (result is implementation-defined)
   - NaN inputs
   - Infinity inputs
   - Values outside the representable range of uint64

   This operation is pure and element-wise.

   Examples
   --------
```python
   import triton
   import triton.language as tl
   from triton.language.extra.cuda import libdevice

   @triton.jit
   def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK + tl.arange(0, BLOCK)
       x = tl.load(x_ptr + offsets)
       # Convert float32 to uint64 with round-towards-zero
       out = libdevice.float2ull_rz(x)
       tl.store(out_ptr + offsets, out)
```

---

### triton.language.extra.cuda.libdevice.float_as_int

```python
float_as_int(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.float_as_int


**`float_as_int(arg0, _semantic=None)`**

   Reinterpret the bit pattern of a 32-bit floating-point value as a 32-bit signed integer.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of float32 values.

   _semantic : optional
       Internal semantic argument (do not pass explicitly).

   Returns
   -------
   tl.tensor
       Tensor of int32 values with the same bit pattern as the input floats.

   Notes
   -----
   This function performs a type punning operation equivalent to CUDA's
   `__nv_float_as_int` libdevice intrinsic. It reinterprets the IEEE 754
   binary representation without numerical conversion.

   For example, the float value `1.0` (bit pattern `0x3f800000`) becomes
   the integer `1065353216`.

   This operation is pure and has no side effects.

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        float_vals = tl.load(x_ptr + offsets)
        int_vals = tl.extra.cuda.libdevice.float_as_int(float_vals)
        tl.store(y_ptr + offsets, int_vals)
```

---

### triton.language.extra.cuda.libdevice.float_as_uint

```python
float_as_uint(arg0, _semantic=None)
```

## float_as_uint


**`float_as_uint(arg0, _semantic=None)`**

    Reinterpret the bit pattern of a 32-bit float as a 32-bit unsigned integer.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of 32-bit floating point values (`tl.float32`).

    Returns
    -------
    tl.tensor
        Tensor of 32-bit integers (`tl.int32`) with the same bit pattern as the input floats.

    Notes
    -----
    This function performs a bitwise reinterpretation, not a numeric conversion.
    The underlying CUDA libdevice function `__nv_float_as_uint` is used.
    This operation is pure and has no side effects.

    Equivalent to a bitcast from `float32` to `int32`.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK + tl.arange(0, BLOCK)
         x = tl.load(x_ptr + offsets)
         # Reinterpret float bits as int
         bits = tl.float_as_uint(x)
         tl.store(out_ptr + offsets, bits)
```

---

### triton.language.extra.cuda.libdevice.floor

```python
floor(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.floor


**`floor(arg0, _semantic=None)`**

    Compute the floor of the input element-wise.

    Returns the largest integer value less than or equal to each element of
    `arg0`. This function dispatches to CUDA libdevice intrinsics
    (`__nv_floorf` for `fp32`, `__nv_floor` for `fp64`).

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values. Must have dtype `fp32` or
        `fp64`.
    _semantic : optional
        Internal parameter for Triton semantic analysis. Do not set manually.

    Returns
    -------
    out : tl.tensor
        Tensor of the same dtype as `arg0` containing the floor values.

    Notes
    -----
    This is an external function that compiles to CUDA libdevice calls. The
    operation is pure (no side effects) and supports the following dtypes:

    - `fp32` → `__nv_floorf`
    - `fp64` → `__nv_floor`

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def floor_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         out = tl.extra.cuda.libdevice.floor(x)
         tl.store(out_ptr + tl.arange(0, BLOCK), out)

     # Input: [1.7, 2.3, -0.5, 3.9]
     # Output: [1.0, 2.0, -1.0, 3.0]
```

---

### triton.language.extra.cuda.libdevice.fma

```python
fma(arg0, arg1, arg2, _semantic=None)
```

## triton.language.extra.cuda.libdevice.fma

**`fma(arg0, arg1, arg2, _semantic=None)`**

    Compute fused multiply-add `arg0 * arg1 + arg2` with single rounding.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor (multiplicand). Must be fp32 or fp64.
    arg1 : tl.tensor
        Second input tensor (multiplicand). Must be fp32 or fp64.
    arg2 : tl.tensor
        Third input tensor (addend). Must be fp32 or fp64.
    _semantic : optional
        Internal parameter, do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of type `arg0 * arg1 + arg2` with the same dtype as inputs.

    Notes
    -----
    This operation performs multiplication and addition in a single rounding
    step, providing better numerical accuracy than separate multiply and add
    operations.

    Supported dtypes:

    - `fp32` maps to CUDA libdevice function `__nv_fmaf`
    - `fp64` maps to CUDA libdevice function `__nv_fma`

    All three arguments must have the same dtype.

    Examples
    --------
```python
     @triton.jit
     def kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         offs = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offs)
         y = tl.load(y_ptr + offs)
         z = tl.load(z_ptr + offs)
         # Compute x * y + z with single rounding
         result = tl.libdevice.fma(x, y, z)
         tl.store(out_ptr + offs, result)
```

---

### triton.language.extra.cuda.libdevice.fma_rd

```python
fma_rd(arg0, arg1, arg2, _semantic=None)
```

## triton.language.extra.cuda.libdevice.fma_rd


.. autofunction:: fma_rd

**`fma_rd(arg0, arg1, arg2, _semantic=None)`**

   Compute fused multiply-add with round-down rounding mode.

   Computes `arg0 * arg1 + arg2` with round-down (toward negative infinity)
   rounding applied to the intermediate result. This operation is performed with
   a single rounding step, providing higher precision than separate multiply and
   add operations.

   Parameters
   ----------
   arg0 : tl.tensor
       First input tensor (multiplicand). Must be floating-point type (fp32 or fp64).
   arg1 : tl.tensor
       Second input tensor (multiplicand). Must be floating-point type (fp32 or fp64).
   arg2 : tl.tensor
       Third input tensor (addend). Must be floating-point type (fp32 or fp64).
   _semantic : optional
       Internal parameter for Triton semantics. Do not set manually.

   Returns
   -------
   result : tl.tensor
       Tensor of the same dtype as inputs containing `arg0 * arg1 + arg2`
       computed with round-down rounding mode.

   Notes
   -----
   This function dispatches to CUDA libdevice intrinsic functions:

   - `__nv_fmaf_rd` for fp32 inputs
   - `__nv_fma_rd` for fp64 inputs

   Round-down rounding mode rounds the result toward negative infinity. This is
   useful for interval arithmetic and reproducible numerical computations.

   All input tensors must have the same dtype (either all fp32 or all fp64).
   Tensors are broadcast to a common shape if necessary.

   Examples
   --------
```python
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
        y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
        z = tl.load(z_ptr + tl.arange(0, BLOCK_SIZE))
        result = libdevice.fma_rd(x, y, z)
        tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

---

### triton.language.extra.cuda.libdevice.fma_rn

```python
fma_rn(arg0, arg1, arg2, _semantic=None)
```

## fma_rn


**`fma_rn(arg0, arg1, arg2, _semantic=None)`**

    Compute fused multiply-add with round-to-nearest rounding mode.

    Computes `arg0 * arg1 + arg2` with a single rounding operation,
    providing better numerical accuracy and performance than separate
    multiply and add operations.

    Parameters
    ----------
    arg0 : tensor
        First input tensor (multiplicand). Must be fp32 or fp64 dtype.
    arg1 : tensor
        Second input tensor (multiplier). Must be fp32 or fp64 dtype.
    arg2 : tensor
        Third input tensor (addend). Must be fp32 or fp64 dtype.
    _semantic : optional
        Internal parameter, do not set manually.

    Returns
    -------
    result : tensor
        Tensor of the same dtype as inputs containing `arg0 * arg1 + arg2`.

    Notes
    -----
    This function maps to CUDA libdevice functions:
    
    - `__nv_fmaf_rn` for fp32 inputs
    - `__nv_fma_rn` for fp64 inputs
    
    All three inputs must have the same dtype (either all fp32 or all fp64).
    The operation is performed with round-to-nearest rounding mode.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
         z = tl.load(z_ptr + tl.arange(0, BLOCK_SIZE))
         # Compute x * y + z with fused multiply-add
         result = tl.extra.cuda.libdevice.fma_rn(x, y, z)
         tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

---

### triton.language.extra.cuda.libdevice.fma_ru

```python
fma_ru(arg0, arg1, arg2, _semantic=None)
```

## triton.language.extra.cuda.libdevice.fma_ru


.. autofunction:: fma_ru

Fused multiply-add with round-up rounding mode.

Computes `(arg0 * arg1) + arg2` with round-up rounding semantics.

### Parameters
arg0 : tl.tensor
    First input tensor. Must be fp32 or fp64.
arg1 : tl.tensor
    Second input tensor. Must be fp32 or fp64.
arg2 : tl.tensor
    Third input tensor. Must be fp32 or fp64.

### Returns
result : tl.tensor
    Tensor of the same dtype as inputs containing the fused multiply-add result
    with round-up rounding.

### Notes
This function calls CUDA libdevice functions `__nv_fmaf_ru` for fp32
and `__nv_fma_ru` for fp64. The "ru" suffix indicates round-up
(round toward positive infinity) rounding mode.

All input tensors must have the same dtype (either all fp32 or all fp64).
The operation is performed element-wise on tensors.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, z_ptr, out_ptr, n, BLOCK: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK + tl.arange(0, BLOCK)
     mask = offs < n
     x = tl.load(x_ptr + offs, mask=mask)
     y = tl.load(y_ptr + offs, mask=mask)
     z = tl.load(z_ptr + offs, mask=mask)
     result = tl.extra.cuda.libdevice.fma_ru(x, y, z)
     tl.store(out_ptr + offs, result, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.fma_rz

```python
fma_rz(arg0, arg1, arg2, _semantic=None)
```

## fma_rz


**`fma_rz(arg0, arg1, arg2, _semantic=None)`**

    Compute fused multiply-add with round-towards-zero rounding mode.

    Computes `arg0 * arg1 + arg2` with a single rounding operation using
    round-towards-zero semantics. This operation is performed with higher
    precision than separate multiply and add operations.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor (multiplicand). Must be floating-point type.
    arg1 : tl.tensor
        Second input tensor (multiplicand). Must be floating-point type.
    arg2 : tl.tensor
        Third input tensor (addend). Must be floating-point type.
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor containing the fused multiply-add result with round-towards-zero
        rounding. Has the same dtype as the inputs.

    Notes
    -----
    This function calls CUDA libdevice functions:

    - `__nv_fmaf_rz` for 32-bit floating-point (fp32)
    - `__nv_fma_rz` for 64-bit floating-point (fp64)

    Round-towards-zero (rz) rounding mode truncates the result towards zero,
    which differs from the default round-to-nearest-even mode. This can be
    useful for numerical algorithms requiring specific rounding behavior.

    All input tensors must have the same floating-point dtype (either all fp32
    or all fp64). Mixed precision is not supported.

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
         # Compute x * y + z with round-towards-zero
         result = tl.extra.cuda.libdevice.fma_rz(x, y, z)
         tl.store(out_ptr + offsets, result)
```

---

### triton.language.extra.cuda.libdevice.fmod

```python
fmod(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.fmod


**`fmod(arg0, arg1, _semantic=None)`**

   Compute the floating-point modulo of two tensors element-wise.

   Returns the remainder of `arg0 / arg1` with the same sign as the dividend (`arg0`).
   This is equivalent to the C library function `fmod`.

   Parameters
   ----------
   arg0 : tensor
       The dividend tensor. Must be floating-point type (fp32 or fp64).
   arg1 : tensor
       The divisor tensor. Must be floating-point type (fp32 or fp64) with the same
       dtype as `arg0`.
   _semantic : optional
       Internal parameter for Triton semantic analysis. Do not set manually.

   Returns
   -------
   tensor
       A tensor containing the floating-point modulo of `arg0` and `arg1`.
       The output dtype matches the input dtype (fp32 or fp64).

   Notes
   -----
   This function is an external libdevice operation that compiles to CUDA PTX
   instructions `__nv_fmodf` for fp32 and `__nv_fmod` for fp64.

   The result is computed as `arg0 - arg1 * trunc(arg0 / arg1)`, where
   `trunc` rounds toward zero. This differs from the Python modulo
   operator which rounds toward negative infinity.

   Special cases:
   - `fmod(x, +inf)` = `x`
   - `fmod(x, -inf)` = `x`
   - `fmod(x, 0)` = NaN
   - `fmod(inf, y)` = NaN
   - `fmod(nan, y)` = NaN

   Examples
   --------
```python
   import torch
   import triton
   import triton.language as tl

   @triton.jit
   def fmod_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(axis=0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       mask = offsets < n_elements
       x = tl.load(x_ptr + offsets, mask=mask)
       y = tl.load(y_ptr + offsets, mask=mask)
       out = tl.extra.cuda.libdevice.fmod(x, y)
       tl.store(out_ptr + offsets, out, mask=mask)

   # Example usage
   n_elements = 1024
   x = torch.randn(n_elements, dtype=torch.float32, device='cuda')
   y = torch.randn(n_elements, dtype=torch.float32, device='cuda')
   out = torch.empty_like(x)
   grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
   fmod_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=256)
```

---

### triton.language.extra.cuda.libdevice.hadd

```python
hadd(arg0, arg1, _semantic=None)
```

## hadd

Compute the hardware-accelerated add with rounding of two integers.

### Parameters
arg0 : tensor
    First input tensor of type `int32` or `uint32`.
arg1 : tensor
    Second input tensor of type `int32` or `uint32`. Must have the same
    dtype as `arg0`.
_semantic : optional
    Internal semantic argument (do not pass explicitly).

### Returns
tensor
    Tensor of the same dtype as inputs containing the hardware add result.

### Notes
This function maps to CUDA libdevice intrinsics:

- `int32` inputs use `__nv_hadd`
- `uint32` inputs use `__nv_uhadd`

The operation computes `(arg0 + arg1) >> 1` with hardware rounding,
providing better precision than naive integer division for averaging.
Both inputs must have the same dtype.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offs)
     y = tl.load(y_ptr + offs)
     # Compute hardware-accelerated average
     avg = tl.extra.cuda.libdevice.hadd(x, y)
     tl.store(out_ptr + offs, avg)
```

---

### triton.language.extra.cuda.libdevice.hiloint2double

```python
hiloint2double(arg0, arg1, _semantic=None)
```

## hiloint2double


**`hiloint2double(arg0, arg1, _semantic=None)`**

   Reinterpret two 32-bit integers as a double-precision floating point value.

   Parameters
   ----------
   arg0 : tl.tensor
       First 32-bit integer (high bits of the double representation).
   arg1 : tl.tensor
       Second 32-bit integer (low bits of the double representation).
   _semantic : optional
       Internal parameter, do not set manually.

   Returns
   -------
   tl.tensor
       Double-precision floating point value (`tl.float64`) constructed from
       the bit pattern of `arg0` and `arg1`.

   Notes
   -----
   This function calls the CUDA libdevice intrinsic `__nv_hiloint2double`.
   The two 32-bit integers are concatenated to form a 64-bit bit pattern,
   which is then reinterpreted as an IEEE 754 double-precision floating
   point value.

   - `arg0` provides the high 32 bits
   - `arg1` provides the low 32 bits

   This is a pure function with no side effects. Both input tensors must
   have dtype `tl.int32`.

   Examples
   --------
```python
    import triton
    import triton.language as tl
    from triton.language.extra.cuda.libdevice import hiloint2double

    @triton.jit
    def kernel(ptr, BLOCK: tl.constexpr):
        # Load two int32 values
        high = tl.load(ptr + tl.arange(0, BLOCK))
        low = tl.load(ptr + tl.arange(0, BLOCK) + BLOCK)

        # Reinterpret as double
        value = hiloint2double(high, low)

        # Use the double value
        tl.store(ptr + tl.arange(0, BLOCK) * 2, value)
```

---

### triton.language.extra.cuda.libdevice.hypot

```python
hypot(arg0, arg1, _semantic=None)
```

## hypot


.. autofunction:: hypot

Compute the hypotenuse of two values element-wise.

Calculates $\sqrt{arg0^2 + arg1^2}$ without unnecessary overflow or underflow.

### Parameters
arg0 : tensor
    First input tensor. Must be floating-point type (fp32 or fp64).
arg1 : tensor
    Second input tensor. Must be floating-point type (fp32 or fp64).
    Will be broadcast to match the shape of `arg0` if shapes differ.
_semantic : optional
    Internal parameter. Do not set manually.

### Returns
tensor
    Tensor of the same shape as inputs containing the hypotenuse values.
    dtype is fp32 if inputs are fp32, fp64 if inputs are fp64.

### Notes
This function maps to CUDA libdevice functions:
- `__nv_hypotf` for fp32 inputs
- `__nv_hypot` for fp64 inputs

The computation is numerically stable and avoids intermediate overflow
when squaring large values.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.load(y_ptr + offsets)
     h = tl.extra.cuda.libdevice.hypot(x, y)
     tl.store(out_ptr + offsets, h)
```

---

### triton.language.extra.cuda.libdevice.ilogb

```python
ilogb(arg0, _semantic=None)
```

## ilogb


**`ilogb(arg0, _semantic=None)`**

    Returns the unbiased exponent of a floating-point value.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values. Must be either `tl.float32` or
        `tl.float64` dtype.

    Returns
    -------
    tl.tensor
        Tensor of `tl.int32` values containing the unbiased exponent of each
        input element.

    Notes
    -----
    This function is a wrapper around CUDA libdevice functions:

    - For `fp32` input: calls `__nv_ilogbf`
    - For `fp64` input: calls `__nv_ilogb`

    The unbiased exponent is the exponent value from the IEEE 754 floating-point
    representation, without the bias. For normal numbers, this equals
    `floor(log2(abs(x)))`. Special cases:

    - `ilogb(0)` returns `-2147483648` (INT32_MIN)
    - `ilogb(inf)` returns `2147483647` (INT32_MAX)
    - `ilogb(nan)` returns `2147483647` (INT32_MAX)

    This function is pure (no side effects) and can be used in JIT-compiled
    Triton kernels.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def extract_exponent_kernel(x_ptr, exp_ptr, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK + tl.arange(0, BLOCK)
         x = tl.load(x_ptr + offsets)
         exp = tl.extra.cuda.libdevice.ilogb(x)
         tl.store(exp_ptr + offsets, exp)

     # Usage
     import torch
     x = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32, device='cuda')
     exp = torch.empty_like(x, dtype=torch.int32)
     extract_exponent_kernel[(1,)](x, exp, BLOCK=4)
     # exp will contain [0, 1, 2, 3]
```

---

### triton.language.extra.cuda.libdevice.int2double_rn

```python
int2double_rn(arg0, _semantic=None)
```

## int2double_rn

Convert 32-bit signed integer to double-precision floating-point with round-to-nearest rounding.

### Parameters
arg0 : tl.tensor
    Input tensor of 32-bit signed integers. Must have dtype `tl.int32`.
_semantic : tl.semantic, optional
    Internal semantic object (automatically provided by Triton compiler).

### Returns
tl.tensor
    Tensor of double-precision floating-point values with dtype `tl.float64`.

### Notes
This function wraps the CUDA libdevice intrinsic `__nv_int2double_rn`. The "rn"
suffix indicates round-to-nearest-even rounding mode.

The operation is pure (no side effects) and is applied element-wise on tensors.

Only `int32` input type is supported.

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def kernel(int_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     int_vals = tl.load(int_ptr + offsets)
     double_vals = libdevice.int2double_rn(int_vals)
     tl.store(out_ptr + offsets, double_vals)
```

---

### triton.language.extra.cuda.libdevice.int2float_rd

```python
int2float_rd(arg0, _semantic=None)
```

## int2float_rd


**`int2float_rd(arg0, _semantic=None)`**

    Convert 32-bit signed integer to 32-bit float with round-down rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of int32 values to convert to floating point.

    Returns
    -------
    tl.tensor
        Tensor of fp32 values converted from input integers using round-down
        (toward negative infinity) rounding mode.

    Notes
    -----
    This function wraps the CUDA libdevice function `__nv_int2float_rd`.
    The round-down rounding mode rounds toward negative infinity, which differs
    from the default round-to-nearest-even mode used by standard integer-to-float
    conversions.

    For positive integers, the result is identical to standard conversion.
    For negative integers that cannot be exactly represented in float32, the
    result is rounded toward negative infinity.

    This is an extern function and requires CUDA libdevice support.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def kernel(int_ptr, float_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         int_vals = tl.load(int_ptr + offsets)
         # Convert with round-down rounding mode
         float_vals = libdevice.int2float_rd(int_vals)
         tl.store(float_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.int2float_rn

```python
int2float_rn(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.int2float_rn

**`int2float_rn(arg0)`**

    Convert 32-bit signed integer to single-precision float with round-to-nearest.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of int32 values to convert.

    Returns
    -------
    tl.tensor
        Tensor of fp32 values resulting from the conversion.

    Notes
    -----
    Calls the CUDA libdevice intrinsic `__nv_int2float_rn`. The conversion uses
    round-to-nearest-even rounding mode. Only int32 input type is supported.
    The operation is pure (no side effects) and applies element-wise.

    Example
    -------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(int_ptr, float_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         int_vals = tl.load(int_ptr + offsets)
         float_vals = tl.extra.cuda.libdevice.int2float_rn(int_vals)
         tl.store(float_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.int2float_ru

```python
int2float_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.int2float_ru


**`int2float_ru(arg0, _semantic=None)`**

    Convert a 32-bit signed integer to a 32-bit floating-point value using round-up rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of int32 values to convert.

    Returns
    -------
    tl.tensor
        Tensor of fp32 values representing the converted integers with round-up rounding.

    Notes
    -----
    This function wraps the CUDA libdevice function `__nv_int2float_ru` which performs
    integer-to-float conversion with IEEE 754 round-toward-positive-infinity rounding mode.
    The rounding mode affects the result when the integer cannot be exactly represented
    in floating-point format.

    This is a pure function with no side effects.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(int_ptr, float_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         int_vals = tl.load(int_ptr + offsets)
         float_vals = tl.extra.cuda.libdevice.int2float_ru(int_vals)
         tl.store(float_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.int2float_rz

```python
int2float_rz(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.int2float_rz


**`int2float_rz(arg0, _semantic=None)`**

    Convert signed 32-bit integer to 32-bit float with round-toward-zero rounding.

    Parameters
    ----------
    arg0 : tensor of int32
        Input tensor containing signed 32-bit integers to convert.
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    tensor of fp32
        Tensor containing the converted floating-point values.

    Notes
    -----
    This function uses the CUDA libdevice intrinsic `__nv_int2float_rz` which
    performs integer-to-float conversion with round-toward-zero (RTZ) rounding
    mode. This differs from the default round-to-nearest-even rounding used by
    standard type casting.

    The function is pure (no side effects) and operates element-wise on tensors.

    This function is only available on CUDA targets.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
         offsets = tl.arange(0, BLOCK)
         int_vals = tl.load(input_ptr + offsets)
         float_vals = tl.extra.cuda.libdevice.int2float_rz(int_vals)
         tl.store(output_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.int_as_float

```python
int_as_float(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.int_as_float


**`int_as_float(arg0, _semantic=None)`**

    Reinterpret the bit pattern of a 32-bit integer as a 32-bit floating-point value.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of type `tl.int32`. The bit pattern will be reinterpreted as
        `tl.float32` without modification.
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    tl.tensor
        Tensor of type `tl.float32` with the same bit pattern as `arg0`.

    Notes
    -----
    This function performs a bitcast operation from `int32` to `fp32`. It is
    equivalent to viewing the bits of an integer as a floating-point number. This
    is useful for low-level bit manipulation or implementing custom floating-point
    operations.

    The operation is pure (no side effects) and compiles to the CUDA libdevice
    function `__nv_int_as_float`.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offs)
         # Reinterpret int32 bits as float32
         y = tl.extra.cuda.libdevice.int_as_float(x)
         tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.isfinited

```python
isfinited(arg0, _semantic=None)
```

## isfinited


**`isfinited(arg0, _semantic=None)`**

    Test element-wise for finiteness (not infinity or Not a Number).

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of float64 (fp64) values.

    Returns
    -------
    out : tl.tensor
        Boolean tensor (int1 dtype). Element-wise True where the input is finite
        (neither NaN nor infinity), False otherwise.

    Notes
    -----
    This function wraps the CUDA libdevice function `__nv_isfinited`.
    It operates element-wise on 64-bit floating-point values only.

    The function is pure and has no side effects.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         finite = tl.extra.cuda.libdevice.isfinited(x)
         tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), finite)

 See Also
 --------
 isnand : Test element-wise for Not a Number (float64)
 isinfd : Test element-wise for positive or negative infinity (float64)
 isfinite : Test element-wise for finiteness (general floating types)
```

---

### triton.language.extra.cuda.libdevice.isinf

```python
isinf(arg0, _semantic=None)
```

## isinf


**`isinf(arg0, _semantic=None)`**

    Test element-wise for positive or negative infinity.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor. Must be of floating-point type (fp32 or fp64).

    Returns
    -------
    out : tl.tensor
        Boolean tensor of type int1. Values are 1 where `arg0` is infinity
        (positive or negative), and 0 otherwise.

    Notes
    -----
    Calls CUDA libdevice functions `__nv_isinff` for float32 inputs and
    `__nv_isinfd` for float64 inputs. The operation is pure (no side effects).

    Examples
    --------
```python
     @triton.jit
     def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         inf_mask = tl.extra.cuda.libdevice.isinf(x)
         tl.store(out_ptr + tl.arange(0, BLOCK), inf_mask)
```

---

### triton.language.extra.cuda.libdevice.isnan

```python
isnan(arg0, _semantic=None)
```

## isnan

Test element-wise for NaN and return result as a boolean tensor.

### Parameters
arg0 : tensor
    Input tensor of floating-point type (`fp32` or `fp64`).

### Returns
out : tensor
    Boolean tensor (`int1` dtype) with the same shape as `arg0`.
    Element is `True` if the corresponding input element is NaN,
    `False` otherwise.

### Notes
This function maps to CUDA libdevice functions:
- `__nv_isnanf` for `fp32` inputs
- `__nv_isnand` for `fp64` inputs

The operation is pure (no side effects) and supports element-wise
broadcasting.

### Examples
```python
 import torch
 import triton
 import triton.language as tl

 @triton.jit
 def check_nan_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK + tl.arange(0, BLOCK)
     mask = offsets < n
     x = tl.load(x_ptr + offsets, mask=mask)
     out = tl.isnan(x)
     tl.store(out_ptr + offsets, out, mask=mask)

 # Test with NaN values
 x = torch.tensor([1.0, float('nan'), 2.0, float('nan')], device='cuda')
 out = torch.empty_like(x, dtype=torch.int32)
 check_nan_kernel[(1,)](x, out, 4, BLOCK=4)
 # out will be [0, 1, 0, 1]
```

---

### triton.language.extra.cuda.libdevice.j0

```python
j0(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.j0


.. autofunction:: j0

Compute the Bessel function of the first kind of order 0.

### Parameters
arg0 : tensor
    Input tensor of floating-point values. Must be `fp32` or `fp64`.

### Returns
tensor
    Tensor of the same dtype as `arg0` containing the Bessel function J0(x)
    for each element.

### Notes
This function is an extern wrapper around CUDA libdevice functions:

- `__nv_j0f` for `fp32` inputs
- `__nv_j0` for `fp64` inputs

The Bessel function of the first kind of order 0 is defined as the solution to
Bessel's differential equation. It oscillates with decaying amplitude as x
increases.

This operation is pure (no side effects) and supports elementwise computation
on tensors.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def bessel_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(axis=0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
     x = tl.load(x_ptr + offsets, mask=mask)
     y = tl.extra.cuda.libdevice.j0(x)
     tl.store(y_ptr + offsets, y, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.j1

```python
j1(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.j1


**`j1(arg0, _semantic=None)`**

   Compute the Bessel function of the first kind of order 1.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor. Must be floating-point type (`fp32` or `fp64`).
   _semantic : optional
       Internal parameter for semantic propagation. Do not set manually.

   Returns
   -------
   tl.tensor
       Tensor of the same shape and dtype as `arg0` containing the Bessel
       function J1 values.

   Notes
   -----
   This function calls the CUDA libdevice implementation:

   - `__nv_j1f` for `fp32` inputs
   - `__nv_j1` for `fp64` inputs

   The Bessel function of the first kind of order 1 is defined as:

   .. math::

      J_1(x) = \frac{1}{\pi} \int_0^\pi \cos(\theta - x \sin \theta) d\theta

   This is a pure function with no side effects.

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
       y = tl.extra.cuda.libdevice.j1(x)
       tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.jn

```python
jn(arg0, arg1, _semantic=None)
```

## tn.language.extra.cuda.libdevice.jn


.. autofunction:: jn

Compute the Bessel function of the first kind of order `n`.

### Parameters
arg0 : tl.tensor
    Order of the Bessel function. Must be of type `int32`.
arg1 : tl.tensor
    Argument value. Must be of type `fp32` or `fp64`.
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
result : tl.tensor
    Bessel function value `J_n(arg1)`. Returns `fp32` if
    `arg1` is `fp32`, `fp64` if `arg1` is
    `fp64`.

### Notes
This function is a wrapper around CUDA libdevice functions:

- `__nv_jnf` for single precision (`fp32`)
- `__nv_jn` for double precision (`fp64`)

The Bessel function of the first kind is defined as the solution to
Bessel's differential equation:

.. math::

    x^2 \\frac{d^2y}{dx^2} + x \\frac{dy}{dx} + (x^2 - n^2)y = 0

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def bessel_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
     offsets = tl.arange(0, BLOCK)
     x = tl.load(x_ptr + offsets)
     # Compute J_n(x) for each element
     result = tl.extra.cuda.libdevice.jn(n, x)
     tl.store(out_ptr + offsets, result)
```

---

### triton.language.extra.cuda.libdevice.ldexp

```python
ldexp(arg0, arg1, _semantic=None)
```

## ldexp


**`ldexp(arg0, arg1, _semantic=None)`**

    Compute `arg0 * 2**arg1`.

    Parameters
    ----------
    arg0 : tl.tensor
        Floating-point mantissa. Supported dtypes: fp32, fp64.
    arg1 : tl.tensor
        Integer exponent. Supported dtype: int32.
    _semantic : optional
        Internal parameter, do not set manually.

    Returns
    -------
    result : tl.tensor
        Floating-point value of `arg0 * 2**arg1`. Same dtype as `arg0`.

    Notes
    -----
    This function wraps CUDA libdevice's `__nv_ldexpf` for fp32 inputs
    and `__nv_ldexp` for fp64 inputs. The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, exp_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         exp = tl.load(exp_ptr + offsets)
         y = tl.extra.cuda.libdevice.ldexp(x, exp)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.lgamma

```python
lgamma(arg0, _semantic=None)
```

## lgamma

Compute the natural logarithm of the absolute value of the Gamma function.

### Parameters
arg0 : tensor
    Input tensor. Must have floating-point dtype (`fp32` or `fp64`).
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
tensor
    Element-wise log gamma of `arg0`. Same dtype as input.

### Notes
This function dispatches to CUDA libdevice:

- `__nv_lgammaf` for `fp32` inputs
- `__nv_lgamma` for `fp64` inputs

The Gamma function is defined as $\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} dt$.
For positive integers, $\Gamma(n) = (n-1)!$.

The function is pure (no side effects) and supports automatic broadcasting.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     y = tl.extra.cuda.libdevice.lgamma(x)
     tl.store(y_ptr + tl.arange(0, BLOCK), y)

 # Compute log gamma of values 1.0 to 8.0
 import torch
 x = torch.arange(1, 9, dtype=torch.float32, device='cuda')
 y = torch.empty_like(x)
 kernel[(1,)](x, y, BLOCK=8)
 # y contains log((n-1)!) for n = 1..8
```

---

### triton.language.extra.cuda.libdevice.ll2double_rd

```python
ll2double_rd(arg0, _semantic=None)
```

## ll2double_rd

Convert 64-bit signed integer to double-precision floating point with round-down rounding.

### Parameters
arg0 : tensor of int64
    Input 64-bit signed integer value(s) to convert.
_semantic : optional
    Internal semantic parameter (do not pass manually).

### Returns
tensor of fp64
    Double-precision floating point representation of the input integer(s).

### Notes
This function wraps the CUDA libdevice function `__nv_ll2double_rd` which
converts a 64-bit signed integer to a double-precision floating point value
using round-down (toward negative infinity) rounding mode.

This is an external elementwise operation that is pure (no side effects).
The function is typically used within `@triton.jit` decorated kernel
functions.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def convert_kernel(int_ptr, float_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     int_vals = tl.load(int_ptr + offsets)
     float_vals = tl.extra.cuda.libdevice.ll2double_rd(int_vals)
     tl.store(float_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.ll2double_rn

```python
ll2double_rn(arg0, _semantic=None)
```

## ll2double_rn(arg0, _semantic=None)

Convert a 64-bit integer to double-precision floating-point with round-to-nearest.

### Parameters
arg0 : tensor of int64
    Input tensor containing 64-bit signed integers to convert.
_semantic : optional
    Internal semantic argument used by Triton compiler.

### Returns
tensor of fp64
    Tensor containing the double-precision floating-point representation
    of the input integers, rounded to nearest.

### Notes
This function wraps the CUDA libdevice intrinsic `__nv_ll2double_rn`.
The conversion uses round-to-nearest rounding mode (RN).

This is an external function that compiles directly to GPU machine code.
It must be called from within a `@triton.jit` decorated kernel.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def convert_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
     offsets = tl.arange(0, BLOCK)
     int_vals = tl.load(input_ptr + offsets)
     float_vals = tl.extra.cuda.libdevice.ll2double_rn(int_vals)
     tl.store(output_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.ll2double_ru

```python
ll2double_ru(arg0, _semantic=None)
```

## ll2double_ru


**`ll2double_ru(arg0, _semantic=None)`**

   Convert a 64-bit signed integer to double-precision floating point with round-up rounding mode.

   Parameters
   ----------
   arg0 : tensor of int64
      Input integer value(s) to convert.

   Returns
   -------
   tensor of fp64
      Converted floating point value(s) with round-up rounding applied.

   Notes
   -----
   This function wraps the CUDA libdevice intrinsic `__nv_ll2double_ru`.

   The round-up rounding mode rounds towards positive infinity when the
   integer cannot be exactly represented as a double-precision float.

   This is a pure function with no side effects.

   Only supports `int64` input type.

   Examples
   --------
```python
   import triton
   import triton.language as tl
   from triton.language.extra.cuda import libdevice

   @triton.jit
   def kernel(ptr, BLOCK: tl.constexpr):
       offsets = tl.arange(0, BLOCK)
       int_vals = tl.load(ptr + offsets).to(tl.int64)
       float_vals = libdevice.ll2double_ru(int_vals)
       tl.store(ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.ll2double_rz

```python
ll2double_rz(arg0, _semantic=None)
```

## ll2double_rz


**`ll2double_rz(arg0, _semantic=None)`**

   Convert a 64-bit signed integer to double-precision floating point with round toward zero.

   Parameters
   ----------
   arg0 : tl.int64
       Input 64-bit signed integer value to convert.
   _semantic : optional
       Internal semantic argument; do not set directly.

   Returns
   -------
   tl.fp64
       Double-precision floating point value representing the input integer,
       rounded toward zero.

   Notes
   -----
   This function wraps the CUDA libdevice intrinsic `__nv_ll2double_rz`.
   The rounding mode is round toward zero (rz), which truncates any fractional
   part when converting from integer to floating point representation.

   This is a pure function with no side effects. The conversion is exact for
   all int64 values that can be represented exactly in fp64 format.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(ptr, n):
       pid = tl.program_id(0)
       offset = pid * 8
       int_val = tl.load(ptr + offset).to(tl.int64)
       float_val = tl.extra.cuda.libdevice.ll2double_rz(int_val)
       tl.store(ptr + offset + 4, float_val)
```

---

### triton.language.extra.cuda.libdevice.ll2float_rd

```python
ll2float_rd(arg0, _semantic=None)
```

## ll2float_rd


**`ll2float_rd(arg0, _semantic=None)`**

    Convert a 64-bit signed integer to a 32-bit floating-point value with round-down rounding.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of 64-bit signed integers (`tl.int64`).

    Returns
    -------
    tl.tensor
        Tensor of 32-bit floating-point values (`tl.fp32`).

    Notes
    -----
    This function wraps the CUDA libdevice function `__nv_ll2float_rd`, which
    converts signed 64-bit integers to single-precision floats using round-down
    (towards negative infinity) rounding mode.

    The operation is pure (no side effects) and is executed element-wise on
    input tensors.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(input_ptr, output_ptr, n):
         pid = tl.program_id(0)
         offsets = pid * 8 + tl.arange(0, 8)
         mask = offsets < n
         int_vals = tl.load(input_ptr + offsets, mask=mask)
         float_vals = tl.extra.cuda.libdevice.ll2float_rd(int_vals)
         tl.store(output_ptr + offsets, float_vals, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.ll2float_rn

```python
ll2float_rn(arg0, _semantic=None)
```

## ll2float_rn


**`ll2float_rn(arg0, _semantic=None)`**

   Convert 64-bit integer to 32-bit float with round-to-nearest rounding.

   Parameters
   ----------
   arg0 : tensor
       Input tensor of int64 values to convert.
   _semantic : optional
       Internal Triton parameter, do not set manually.

   Returns
   -------
   tensor
       Tensor of fp32 values resulting from the conversion.

   Notes
   -----
   This function wraps the CUDA libdevice function `__nv_ll2float_rn`.
   The `rn` suffix indicates round-to-nearest-even rounding mode.
   This is an extern function that compiles to GPU machine code.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(ptr, n, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK + tl.arange(0, BLOCK)
       int_vals = tl.load(ptr + offsets).to(tl.int64)
       float_vals = tl.extra.cuda.libdevice.ll2float_rn(int_vals)
       tl.store(ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.ll2float_ru

```python
ll2float_ru(arg0, _semantic=None)
```

## ll2float_ru

Convert a 64-bit signed integer to float32 with round-up rounding mode.

### Parameters
arg0 : tl.int64
    Input 64-bit signed integer tensor or scalar.

### Returns
tl.float32
    Floating-point representation of the input with round-up (toward positive
    infinity) rounding mode.

### Notes
This function wraps the CUDA libdevice function `__nv_ll2float_ru`. The
rounding mode "ru" (round up) rounds toward positive infinity. This operation
is pure (no side effects).

Only CUDA targets support this function. The input must be of type
`tl.int64`.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     int_vals = tl.load(x_ptr + offsets).to(tl.int64)
     float_vals = tl.extra.cuda.libdevice.ll2float_ru(int_vals)
     tl.store(y_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.ll2float_rz

```python
ll2float_rz(arg0, _semantic=None)
```

## ll2float_rz


**`ll2float_rz(arg0, _semantic=None)`**

    Convert 64-bit signed integer to 32-bit float with round-toward-zero.

    Parameters
    ----------
    arg0 : tensor
        Input tensor of int64 values to convert.
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    tensor
        Tensor of fp32 values converted from int64 input with round-toward-zero
        rounding mode.

    Notes
    -----
    This function calls the CUDA libdevice intrinsic `__nv_ll2float_rz`.
    The conversion uses round-toward-zero (truncate) rounding mode, which
    truncates the fractional part toward zero. This differs from the default
    round-to-nearest-even mode.

    The function is pure (no side effects) and operates elementwise on tensors.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         int64_vals = tl.load(input_ptr + offsets).to(tl.int64)
         float_vals = tl.extra.cuda.libdevice.ll2float_rz(int64_vals)
         tl.store(output_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.llrint

```python
llrint(arg0, _semantic=None)
```

## llrint

Round floating-point value to nearest integer (long long).

### Parameters
arg0 : tensor
    Input floating-point tensor. Must be of type `fp32` or `fp64`.
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
out : tensor
    Tensor of type `int64` containing the rounded integer values.

### Notes
This function rounds each element of the input tensor to the nearest
integer value using the current rounding mode, returning the result as
a 64-bit integer. For CUDA targets, this maps to the libdevice functions
`__nv_llrintf` (for `fp32`) and `__nv_llrint` (for `fp64`).

The operation is pure (no side effects) and supports both scalar and
block tensors.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     # Round floating-point values to nearest integer
     y = tl.extra.cuda.libdevice.llrint(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.llround

```python
llround(arg0, _semantic=None)
```

### llround


**`llround(arg0, _semantic=None)`**

    Round the input floating-point value to the nearest integer.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values. Must be fp32 or fp64 dtype.

    Returns
    -------
    ret : tl.tensor
        Tensor of int64 values representing the rounded input.

    Notes
    -----
    This function dispatches to CUDA libdevice intrinsics:
    
    - `__nv_llroundf` for fp32 inputs
    - `__nv_llround` for fp64 inputs
    
    Rounding is performed to the nearest integer value, with halfway cases
    rounded away from zero. The result is always returned as a 64-bit
    signed integer (int64).

    This is an extern function and requires CUDA backend support.

    Example
    -------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def round_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         offset = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offset)
         y = tl.extra.cuda.libdevice.llround(x)
         tl.store(y_ptr + offset, y)
```

---

### triton.language.extra.cuda.libdevice.log

```python
log(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.log


**`log(arg0, _semantic=None)`**

    Compute the natural logarithm element-wise.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor. Must be of floating-point type (`fp32` or `fp64`).
    _semantic : optional
        Internal parameter for semantic propagation. Do not set manually.

    Returns
    -------
    out : tl.tensor
        Tensor of the same shape as `arg0` containing the natural logarithm
        of each element. The dtype matches the input dtype (`fp32` or `fp64`).

    Notes
    -----
    This function dispatches to CUDA libdevice functions:

    - `__nv_logf` for `fp32` inputs
    - `__nv_log` for `fp64` inputs

    The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def log_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.extra.cuda.libdevice.log(x)
         tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.log10

```python
log10(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.log10


Compute the base-10 logarithm of the input element-wise.

```python
 log10(arg0, _semantic=None)

```
### Parameters
arg0 : tensor
    Input tensor or scalar. Must be of floating-point type (`fp32` or `fp64`).
_semantic : optional
    Internal semantic argument. Do not pass explicitly.

### Returns
out : tensor
    Element-wise base-10 logarithm of `arg0`. Same dtype as input.

### Notes
This function dispatches to CUDA libdevice implementations:

- `__nv_log10f` for `fp32` inputs
- `__nv_log10` for `fp64` inputs

The operation is pure (no side effects). Results are undefined for negative
inputs. `log10(0)` returns `-inf`.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def log10_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offs < n
     x = tl.load(x_ptr + offs, mask=mask)
     out = tl.extra.cuda.libdevice.log10(x)
     tl.store(out_ptr + offs, out, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.log1p

```python
log1p(arg0, _semantic=None)
```

## log1p


**`log1p(arg0, _semantic=None)`**

   Compute the natural logarithm of one plus the input.

   Computes $\log(1 + x)$ accurately for small values of $x$, avoiding
   loss of precision that would occur from computing $\log(1 + x)$ directly.

   Parameters
   ----------
   arg0 : tensor
       Input tensor. Must be of floating-point type (fp32 or fp64).
   _semantic : optional
       Internal semantic argument, do not provide manually.

   Returns
   -------
   tensor
       Tensor of the same dtype as `arg0` containing $\log(1 + arg0)$.

   Notes
   -----
   This function is an extern operation that maps to CUDA libdevice functions:

   - `__nv_log1pf` for fp32 inputs
   - `__nv_log1p` for fp64 inputs

   The operation is pure (no side effects).

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs)
        y = tl.extra.cuda.libdevice.log1p(x)
        tl.store(y_ptr + offs, y)
```

---

### triton.language.extra.cuda.libdevice.log2

```python
log2(arg0, _semantic=None)
```

## log2

Compute the base-2 logarithm of the input tensor element-wise.

### Parameters
arg0 : tensor
    Input tensor. Must be of floating-point type (`fp32` or `fp64`).
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
tensor
    Tensor of the same shape as `arg0` containing the base-2 logarithm
    of each element. Returns `fp32` for `fp32` input, `fp64` for
    `fp64` input.

### Notes
This function is implemented using CUDA libdevice functions:

- `__nv_log2f` for 32-bit floating-point inputs
- `__nv_log2` for 64-bit floating-point inputs

The operation is pure (no side effects). Input values must be positive;
behavior for non-positive values follows IEEE-754 semantics (returns NaN
for negative inputs, -inf for zero).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def log2_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.log2(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.logb

```python
logb(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.logb


**`logb(arg0, _semantic=None)`**

    Compute the exponent of a floating-point value.

    Returns the unbiased exponent of `arg0` as a floating-point value.
    For finite `arg0`, the result is `floor(log2(|arg0|))`.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values. Must be `fp32` or `fp64`.
    _semantic : optional
        Internal Triton semantic parameter. Do not set manually.

    Returns
    -------
    out : tl.tensor
        Tensor of the same dtype as `arg0` containing the unbiased exponents.
        For `fp32` input, returns `fp32`; for `fp64` input, returns `fp64`.

    Notes
    -----
    This function is a wrapper around CUDA libdevice functions:

    - `__nv_logbf` for `fp32` inputs
    - `__nv_logb` for `fp64` inputs

    Special values:

    - `logb(+inf)` = `+inf`
    - `logb(-inf)` = `+inf`
    - `logb(0)` = `-inf`
    - `logb(NaN)` = `NaN`

    The function is pure (no side effects) and can be used in JIT-compiled kernels.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
         offsets = tl.arange(0, BLOCK)
         x = tl.load(x_ptr + offsets)
         # Extract exponent from floating-point values
         exp = tl.extra.cuda.libdevice.logb(x)
         tl.store(out_ptr + offsets, exp)

     # Launch kernel
     BLOCK = 1024
     x = torch.randn(BLOCK, device='cuda', dtype=torch.float32)
     out = torch.empty(BLOCK, device='cuda', dtype=torch.float32)
     kernel[(1,)](x, out, BLOCK)
```

---

### triton.language.extra.cuda.libdevice.longlong_as_double

```python
longlong_as_double(arg0, _semantic=None)
```

## longlong_as_double

Reinterpret the bit pattern of a 64-bit integer as a double-precision floating point value.

### Parameters
arg0 : tensor of int64
    Input tensor containing 64-bit integers to reinterpret as doubles.
_semantic : optional
    Internal semantic argument, do not set manually.

### Returns
tensor of float64
    Tensor containing the bitcasted double-precision floating point values.

### Notes
This function performs a bitwise reinterpretation without changing the underlying
bits. It is equivalent to a bitcast operation from int64 to float64. The function
is pure and has no side effects.

This is a CUDA libdevice function that maps to `__nv_longlong_as_double` in
PTX. It is typically used for low-level bit manipulation or when implementing
custom floating point operations.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(ptr_in, ptr_out, BLOCK: tl.constexpr):
     offsets = tl.arange(0, BLOCK)
     int_vals = tl.load(ptr_in + offsets).to(tl.int64)
     float_vals = tl.extra.cuda.libdevice.longlong_as_double(int_vals)
     tl.store(ptr_out + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.mul24

```python
mul24(arg0, arg1, _semantic=None)
```

mul24(arg0, arg1, _semantic=None)

Multiply two 32-bit integers and return the lower 24 bits of the product.

### Parameters
arg0 : tensor
    First input tensor. Must have dtype `int32` or `uint32`.
arg1 : tensor
    Second input tensor. Must have dtype `int32` or `uint32`.
_semantic : optional
    Internal parameter, do not set.

### Returns
tensor
    Tensor containing the lower 24 bits of the product. Returns `int32` if
    inputs are `int32`, `uint32` if inputs are `uint32`.

### Notes
This function calls the CUDA libdevice functions `__nv_mul24` for signed
integers and `__nv_umul24` for unsigned integers. Only the lower 24 bits of
the 32-bit product are returned. This is useful when you need 24-bit precision
multiplication on CUDA devices.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     y = tl.load(y_ptr + tl.arange(0, BLOCK))
     result = tl.extra.cuda.libdevice.mul24(x, y)
     tl.store(out_ptr + tl.arange(0, BLOCK), result)
```

---

### triton.language.extra.cuda.libdevice.mul_rd

```python
mul_rd(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.mul_rd


**`mul_rd(arg0, arg1, _semantic=None)`**

    Multiply two floating-point values with round-down (toward negative infinity) rounding mode.

    Parameters
    ----------
    arg0 : tensor
        First input tensor. Must be of floating-point type (`fp32` or `fp64`).
    arg1 : tensor
        Second input tensor. Must be of floating-point type (`fp32` or `fp64`).
    _semantic : optional
        Internal parameter for Triton semantic handling. Do not set manually.

    Returns
    -------
    result : tensor
        Tensor containing the element-wise product of `arg0` and `arg1` with
        round-down rounding mode applied. Has the same dtype as the inputs.

    Notes
    -----
    This function calls CUDA libdevice functions:

    - `__nv_fmul_rd` for `fp32` inputs
    - `__nv_dmul_rd` for `fp64` inputs

    Round-down rounding mode rounds the result toward negative infinity. This differs
    from the default round-to-nearest-even mode used by standard multiplication.

    Both inputs must have the same dtype. The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
         # Multiply with round-down rounding mode
         result = tl.extra.cuda.libdevice.mul_rd(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

---

### triton.language.extra.cuda.libdevice.mul_rn

```python
mul_rn(arg0, arg1, _semantic=None)
```

## mul_rn


**`mul_rn(arg0, arg1, _semantic=None)`**

   Multiply two floating-point values with round-to-nearest rounding mode.

   Parameters
   ----------
   arg0 : tl.tensor
       First input tensor. Must be of type `tl.float32` or `tl.float64`.
   arg1 : tl.tensor
       Second input tensor. Must be of type `tl.float32` or `tl.float64`.
   _semantic : optional
       Internal parameter for Triton semantics. Do not set manually.

   Returns
   -------
   result : tl.tensor
       Element-wise product of `arg0` and `arg1` with round-to-nearest
       rounding. Same dtype as inputs.

   Notes
   -----
   This function calls CUDA libdevice routines `__nv_fmul_rn` (for fp32)
   and `__nv_dmul_rn` (for fp64). The `rn` suffix indicates round-to-nearest
   rounding mode, which is the default IEEE 754 rounding behavior.

   Both inputs must have the same dtype. Mixed precision is not supported.

   This is an extern function that requires CUDA backend.

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
       # Multiply with round-to-nearest rounding
       result = tl.extra.cuda.libdevice.mul_rn(x, y)
       tl.store(out_ptr + offs, result)
```

---

### triton.language.extra.cuda.libdevice.mul_ru

```python
mul_ru(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.mul_ru


**`mul_ru(arg0, arg1, _semantic=None)`**

    Multiply two floating-point values with round-up rounding mode.

    Performs element-wise multiplication of `arg0` and `arg1` using IEEE 754 
    round-up (toward positive infinity) rounding mode. Implemented via CUDA 
    libdevice functions.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be of floating-point type (fp32 or fp64).
    arg1 : tl.tensor
        Second input tensor. Must be of floating-point type (fp32 or fp64).
    _semantic : optional
        Internal parameter. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor containing the element-wise product with round-up rounding.
        Has the same dtype as the inputs (fp32 or fp64).

    Notes
    -----
    This function is only available on CUDA devices. It maps to CUDA libdevice 
    functions `__nv_fmul_ru` for float32 and `__nv_dmul_ru` for float64.

    The round-up rounding mode rounds results toward positive infinity. This 
    differs from standard multiplication which uses round-to-nearest-even.

    Supported input dtypes: fp32, fp64. Both inputs must have the same dtype.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
         result = tl.extra.cuda.libdevice.mul_ru(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

---

### triton.language.extra.cuda.libdevice.mul_rz

```python
mul_rz(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.mul_rz


.. autofunction:: mul_rz

Multiply two floating-point values with round-to-zero rounding mode.

### Parameters
arg0 : tensor
    First input tensor. Must be floating-point type (`fp32` or `fp64`).
arg1 : tensor
    Second input tensor. Must be floating-point type (`fp32` or `fp64`).
    Must have the same dtype as `arg0`.

### Returns
result : tensor
    Element-wise product of `arg0` and `arg1` with round-to-zero rounding.
    The dtype matches the input dtype.

### Notes
This function calls CUDA libdevice functions `__nv_fmul_rz` for `fp32` and
`__nv_dmul_rz` for `fp64`. Round-to-zero rounding mode truncates the result
towards zero, which can be useful for controlling floating-point precision
in numerical algorithms.

The operation is element-wise for tensor inputs. Both inputs must have
the same dtype.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK + tl.arange(0, BLOCK)
     x = tl.load(x_ptr + offs)
     y = tl.load(y_ptr + offs)
     result = tl.extra.cuda.libdevice.mul_rz(x, y)
     tl.store(out_ptr + offs, result)
```

---

### triton.language.extra.cuda.libdevice.mulhi

```python
mulhi(arg0, arg1, _semantic=None)
```

## mulhi


**`mulhi(arg0, arg1, _semantic=None)`**

    Returns the high bits of the product of two integers.

    Computes the upper portion of a full multiplication (arg0 * arg1). For 32-bit
    inputs, returns the high 32 bits of the 64-bit product. For 64-bit inputs,
    returns the high 64 bits of the 128-bit product.

    Parameters
    ----------
    arg0 : tensor
        First input tensor. Must be int32, uint32, int64, or uint64.
    arg1 : tensor
        Second input tensor. Must have the same dtype as arg0.
    _semantic : optional
        Internal parameter, do not set manually.

    Returns
    -------
    result : tensor
        Tensor containing the high bits of the multiplication. Same dtype as
        inputs.

    Notes
    -----
    This function maps to CUDA libdevice functions:

    - `__nv_mulhi` for int32
    - `__nv_umulhi` for uint32
    - `__nv_mul64hi` for int64
    - `__nv_umul64hi` for uint64

    The operation is pure (no side effects). Both arguments must have matching
    dtypes.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         y = tl.load(y_ptr + tl.arange(0, BLOCK))
         # Get high bits of multiplication
         hi = tl.extra.cuda.libdevice.mulhi(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK), hi)
```

---

### triton.language.extra.cuda.libdevice.nearbyint

```python
nearbyint(arg0, _semantic=None)
```

## nearbyint

Round to the nearest integer value.

### Parameters
arg0 : tensor
    Input tensor of floating-point values. Supported dtypes are `fp32` and
    `fp64`.
_semantic : optional
    Internal parameter, do not set manually.

### Returns
tensor
    Tensor of the same dtype as `arg0` containing the nearest integer values.
    The dtype is preserved (`fp32` returns `fp32`, `fp64` returns `fp64`).

### Notes
This function rounds to the nearest integer value, with ties rounded to even
(rounding mode: round to nearest, ties to even). It is equivalent to the C
`nearbyint` function and maps to CUDA libdevice functions `__nv_nearbyintf`
(for `fp32`) and `__nv_nearbyint` (for `fp64`).

Unlike `rint`, this function does not raise the inexact floating-point
exception when the result differs from the input. The operation is pure (no
side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.nearbyint(x)
     tl.store(y_ptr + offsets, y)

 # Example usage
 # x = [1.5, 2.3, 3.7, -1.5]
 # y = [2.0, 2.0, 4.0, -2.0]  # ties rounded to even
```

---

### triton.language.extra.cuda.libdevice.nextafter

```python
nextafter(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.nextafter


**`nextafter(arg0, arg1, _semantic=None)`**

    Returns the next representable floating-point value after `arg0` in the direction of `arg1`.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values (fp32 or fp64).
    arg1 : tl.tensor
        Input tensor indicating the direction (fp32 or fp64).
    _semantic : optional
        Internal parameter for semantic propagation. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of the same dtype as inputs containing the next representable values.

    Notes
    -----
    This function wraps CUDA libdevice functions:

    - `__nv_nextafterf` for 32-bit floating-point (fp32)
    - `__nv_nextafter` for 64-bit floating-point (fp64)

    Both inputs must have the same dtype. The function is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK + tl.arange(0, BLOCK)
         x = tl.load(x_ptr + offset)
         y = tl.full((BLOCK,), 1.0, dtype=tl.float32)
         result = tl.extra.cuda.libdevice.nextafter(x, y)
         tl.store(y_ptr + offset, result)
```

---

### triton.language.extra.cuda.libdevice.norm3d

```python
norm3d(arg0, arg1, arg2, _semantic=None)
```

norm3d(arg0, arg1, arg2, _semantic=None)

    Compute the 3D Euclidean norm $\sqrt{x^2 + y^2 + z^2}$.

    Parameters
    ----------
    arg0 : tl.tensor
        First component of the 3D vector. Must be floating-point (fp32 or fp64).
    arg1 : tl.tensor
        Second component of the 3D vector. Must be floating-point (fp32 or fp64).
    arg2 : tl.tensor
        Third component of the 3D vector. Must be floating-point (fp32 or fp64).
    _semantic : optional
        Internal parameter, do not use directly.

    Returns
    -------
    result : tl.tensor
        The Euclidean norm of the 3D vector. Has the same dtype as inputs.

    Notes
    -----
    This function calls CUDA libdevice functions `__nv_norm3df` for fp32 inputs
    and `__nv_norm3d` for fp64 inputs. All three arguments must have the same
    dtype. The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def norm_kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         y = tl.load(y_ptr + tl.arange(0, BLOCK))
         z = tl.load(z_ptr + tl.arange(0, BLOCK))
         norm = tl.extra.cuda.libdevice.norm3d(x, y, z)
         tl.store(out_ptr + tl.arange(0, BLOCK), norm)
```

---

### triton.language.extra.cuda.libdevice.norm4d

```python
norm4d(arg0, arg1, arg2, arg3, _semantic=None)
```

## norm4d


**`norm4d(arg0, arg1, arg2, arg3, _semantic=None)`**

   Compute the Euclidean norm (L2 norm) of a 4-dimensional vector.

   Computes $\sqrt{arg0^2 + arg1^2 + arg2^2 + arg3^2}$ element-wise.

   Parameters
   ----------
   arg0 : tl.tensor
       First component of the 4D vector. Must be fp32 or fp64.
   arg1 : tl.tensor
       Second component of the 4D vector. Must have same dtype as arg0.
   arg2 : tl.tensor
       Third component of the 4D vector. Must have same dtype as arg0.
   arg3 : tl.tensor
       Fourth component of the 4D vector. Must have same dtype as arg0.
   _semantic : optional
       Internal parameter, do not set manually.

   Returns
   -------
   result : tl.tensor
       The Euclidean norm of the 4D vector. Has the same dtype as the
       input arguments (fp32 or fp64).

   Notes
   -----
   This function dispatches to CUDA libdevice intrinsics:

   - `__nv_norm4df` for float32 inputs
   - `__nv_norm4d` for float64 inputs

   All input arguments must have the same dtype. Mixed precision is not
   supported. The function is pure (no side effects).

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def normalize_4d_kernel(x_ptr, y_ptr, z_ptr, w_ptr, out_ptr, BLOCK: tl.constexpr):
       x = tl.load(x_ptr + tl.arange(0, BLOCK))
       y = tl.load(y_ptr + tl.arange(0, BLOCK))
       z = tl.load(z_ptr + tl.arange(0, BLOCK))
       w = tl.load(w_ptr + tl.arange(0, BLOCK))
       norm = tl.extra.cuda.libdevice.norm4d(x, y, z, w)
       tl.store(out_ptr + tl.arange(0, BLOCK), norm)

   # Launch kernel
   BLOCK = 1024
   normalize_4d_kernel[(1,)](x_ptr, y_ptr, z_ptr, w_ptr, out_ptr, BLOCK=BLOCK)
```

---

### triton.language.extra.cuda.libdevice.normcdf

```python
normcdf(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.normcdf


.. autofunction:: normcdf

Compute the cumulative distribution function (CDF) of the standard normal distribution.

### Parameters
arg0 : tensor
    Input tensor of floating-point values. Supported dtypes are `fp32` and `fp64`.
_semantic : optional
    Internal semantic argument (do not pass explicitly).

### Returns
tensor
    Tensor of the same dtype as `arg0` containing the CDF values. For each element `x`,
    returns the probability that a standard normal random variable is less than or equal to `x`.

### Notes
This function wraps CUDA libdevice functions:

- `__nv_normcdff` for `fp32` inputs
- `__nv_normcdf` for `fp64` inputs

The standard normal CDF is defined as:

.. math::

    \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-t^2/2} dt

The function is pure (no side effects) and supports elementwise operations on tensors.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(X_ptr, Y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(X_ptr + offsets)
     # Compute standard normal CDF
     y = tl.extra.cuda.libdevice.normcdf(x)
     tl.store(Y_ptr + offsets, y)

 # Usage with float32
 x = torch.randn(1024, device='cuda', dtype=torch.float32)
 y = torch.empty_like(x)
 kernel[(1,)](x, y, BLOCK_SIZE=1024)
```

---

### triton.language.extra.cuda.libdevice.normcdfinv

```python
normcdfinv(arg0, _semantic=None)
```

## normcdfinv

Compute the inverse of the standard normal cumulative distribution function (quantile function).

### Parameters
arg0 : tensor
    Input tensor containing probabilities. Values must be in the open interval 
    (0, 1). Supports `fp32` and `fp64` dtypes.

### Returns
tensor
    Tensor containing z-scores (quantiles) corresponding to the input 
    probabilities. Has the same dtype as `arg0`.

### Notes
This function computes the quantile function (percent point function) of the 
standard normal distribution. For a given probability $p$, it returns 
the value $z$ such that $P(Z \leq z) = p$ where $Z$ is a 
standard normal random variable.

The function is undefined for probabilities outside the open interval (0, 1). 
As probabilities approach 0 or 1, the result approaches negative or positive 
infinity respectively.

This operation is implemented using CUDA libdevice functions:
- `__nv_normcdfinvf` for `fp32` inputs
- `__nv_normcdfinv` for `fp64` inputs

The function is pure (no side effects) and can be safely used in any context.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def normcdfinv_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     probs = tl.load(x_ptr + offsets)
     z_scores = tl.extra.cuda.libdevice.normcdfinv(probs)
     tl.store(y_ptr + offsets, z_scores)

 # Example: convert probabilities to z-scores
 # prob = 0.5 -> z_score = 0.0
 # prob = 0.975 -> z_score ≈ 1.96
```

---

### triton.language.extra.cuda.libdevice.popc

```python
popc(arg0, _semantic=None)
```

popc(arg0, _semantic=None)

    Count the number of set bits (population count) in an integer.

    Parameters
    ----------
    arg0 : tensor
        Input tensor of integer type. Supported dtypes are `int32` and `int64`.
    _semantic
        Internal parameter, do not use directly.

    Returns
    -------
    result : tensor
        Tensor of `int32` containing the population count (number of 1-bits) for each element.

    Notes
    -----
    This function wraps CUDA libdevice population count intrinsics:
    
    - `__nv_popc` for `int32` inputs
    - `__nv_popcll` for `int64` inputs
    
    The operation is pure (no side effects) and elementwise.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def popc_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.extra.cuda.libdevice.popc(x)
         tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)

     # Count set bits in input values
     # Input:  [0b0001, 0b0011, 0b0111, 0b1111]
     # Output: [1,      2,      3,      4     ]
```

---

### triton.language.extra.cuda.libdevice.pow

```python
pow(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.pow


**`pow(arg0, arg1, _semantic=None)`**

    Compute `arg0` raised to the power of `arg1`.

    Parameters
    ----------
    arg0 : tl.tensor
        The base value. Must be floating-point (fp32 or fp64).
    arg1 : tl.tensor
        The exponent. Can be floating-point (fp32 or fp64) or integer (int32).
    _semantic : optional
        Internal semantic argument. Do not set manually.

    Returns
    -------
    result : tl.tensor
        The result of `arg0 ** arg1`. Has the same dtype as `arg0`
        (fp32 or fp64).

    Notes
    -----
    This function dispatches to CUDA libdevice intrinsics based on input dtypes:

    - (fp32, int32) -> `__nv_powif` -> fp32
    - (fp64, int32) -> `__nv_powi` -> fp64
    - (fp32, fp32) -> `__nv_powf` -> fp32
    - (fp64, fp64) -> `__nv_pow` -> fp64

    The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, n):
         pid = tl.program_id(0)
         offset = pid * tl.constexpr(128)
         x = tl.load(x_ptr + offset)
         y = tl.load(y_ptr + offset)
         result = tl.extra.cuda.libdevice.pow(x, y)
         tl.store(out_ptr + offset, result)
```

---

### triton.language.extra.cuda.libdevice.rcbrt

```python
rcbrt(arg0, _semantic=None)
```

## rcbrt


**`rcbrt(arg0, _semantic=None)`**

    Compute the reciprocal cube root of the input element-wise.

    Parameters
    ----------
    arg0 : tensor
        Input tensor of floating-point values (`fp32` or `fp64`).
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    result : tensor
        Tensor of the same shape and dtype as `arg0`, containing
        $1 / \sqrt[3]{x}$ for each element.

    Notes
    -----
    This function dispatches to CUDA libdevice functions:

    - `__nv_rcbrtf` for `fp32` inputs
    - `__nv_rcbrt` for `fp64` inputs

    The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.extra.cuda.libdevice.rcbrt(x)
         tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.rcp64h

```python
rcp64h(arg0, _semantic=None)
```

## rcp64h


**`rcp64h(arg0, _semantic=None)`**

   Compute the reciprocal (1/x) of a 64-bit floating-point value using CUDA libdevice.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of dtype fp64.
   _semantic : optional
       Internal parameter, do not use directly.

   Returns
   -------
   result : tl.tensor
       Tensor of dtype fp64 containing the reciprocal of input values (1/arg0).

   Notes
   -----
   This function is an extern wrapper around the CUDA libdevice function
   `__nv_rcp64h`. The operation is element-wise and marked as pure
   (no side effects).

   Only supports fp64 (double precision) input and output types.

   Examples
   --------
```python
   import triton
   import triton.language as tl
   import torch

   @triton.jit
   def reciprocal_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       offs = pid * BLOCK + tl.arange(0, BLOCK)
       mask = offs < n
       x = tl.load(x_ptr + offs, mask=mask)
       y = tl.extra.cuda.libdevice.rcp64h(x)
       tl.store(y_ptr + offs, y, mask=mask)

   n = 1024
   x = torch.randn(n, dtype=torch.float64, device='cuda')
   y = torch.empty(n, dtype=torch.float64, device='cuda')
   reciprocal_kernel[(triton.cdiv(n, 128),)](x, y, n, BLOCK=128)
```

---

### triton.language.extra.cuda.libdevice.rcp_rd

```python
rcp_rd(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.rcp_rd


.. autofunction:: rcp_rd

Compute reciprocal of `arg0` with round-downward rounding mode.

### Parameters
arg0 : tl.tensor
    Input tensor of floating-point values. Supports `fp32` and 
    `fp64` dtypes.

### Returns
result : tl.tensor
    Tensor containing the reciprocal (1/`arg0`) of each element, 
    computed with round-downward rounding mode. The dtype matches the 
    input dtype.

### Notes
This function dispatches to CUDA libdevice functions:

- `__nv_frcp_rd` for `fp32` inputs
- `__nv_drcp_rd` for `fp64` inputs

The round-downward rounding mode rounds the result toward negative 
infinity, which can be useful for controlling numerical precision in 
IEEE-754 floating-point arithmetic.

This is a pure function with no side effects.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def reciprocal_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
     y = tl.extra.cuda.libdevice.rcp_rd(x)
     tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.rcp_rn

```python
rcp_rn(arg0, _semantic=None)
```

## rcp_rn


**`rcp_rn(arg0, _semantic=None)`**

   Compute reciprocal with round-to-nearest rounding.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of floating-point type (fp32 or fp64).
   _semantic : optional
       Internal semantic argument, do not set manually.

   Returns
   -------
   result : tl.tensor
       Reciprocal of input (1/arg0) with same dtype as input.

   Notes
   -----
   This function calls NVIDIA libdevice device functions:

   - `__nv_frcp_rn` for fp32 inputs
   - `__nv_drcp_rn` for fp64 inputs

   The rounding mode is round-to-nearest (RN). The function is pure
   (no side effects).

   Example
   -------
```python
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def reciprocal_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offsets)
        y = libdevice.rcp_rn(x)
        tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.rcp_ru

```python
rcp_ru(arg0, _semantic=None)
```

## rcp_ru

Compute reciprocal with round-up rounding mode.

### Parameters
arg0 : tensor
    Input tensor of floating-point values. Supported dtypes are fp32 and fp64.
_semantic : optional
    Internal parameter for semantic propagation. Do not set manually.

### Returns
result : tensor
    Tensor containing reciprocals (1.0 / arg0) computed with round-up
    rounding mode. Same dtype as input.

### Notes
This function dispatches to CUDA libdevice intrinsic functions:
`__nv_frcp_ru` for fp32 inputs and `__nv_drcp_ru` for fp64 inputs.
The "ru" suffix indicates IEEE 754 round-up (toward positive infinity)
rounding mode.

This is a pure function with no side effects.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK))
     y = tl.extra.cuda.libdevice.rcp_ru(x)
     tl.store(y_ptr + tl.arange(0, BLOCK), y)
```

---

### triton.language.extra.cuda.libdevice.rcp_rz

```python
rcp_rz(arg0, _semantic=None)
```

## rcp_rz


**`rcp_rz(arg0, _semantic=None)`**

    Compute the reciprocal of `arg0` with round-toward-zero rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor. Must be of type `tl.float32` or `tl.float64`.

    Returns
    -------
    out : tl.tensor
        Reciprocal of `arg0` (1.0 / arg0) with round-toward-zero rounding.
        Same dtype as input.

    Notes
    -----
    This function calls CUDA libdevice functions:

    - `__nv_frcp_rz` for `fp32` inputs
    - `__nv_drcp_rz` for `fp64` inputs

    The round-toward-zero (rz) rounding mode truncates the result towards zero.
    This differs from the default round-to-nearest-even rounding mode.

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
         y = tl.extra.cuda.libdevice.rcp_rz(x)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.remainder

```python
remainder(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.remainder


**`remainder(arg0, arg1, _semantic=None)`**

    Compute the IEEE remainder of dividing `arg0` by `arg1`.

    Parameters
    ----------
    arg0 : tl.tensor
        The dividend. Must be floating-point (fp32 or fp64).
    arg1 : tl.tensor
        The divisor. Must be floating-point (fp32 or fp64) with the same dtype
        as `arg0`.
    _semantic : optional
        Internal parameter for semantic propagation. Do not set manually.

    Returns
    -------
    result : tl.tensor
        The remainder of `arg0 / arg1`. Has the same dtype as the inputs.

    Notes
    -----
    This function wraps CUDA libdevice's `__nv_remainderf` (fp32) and
    `__nv_remainder` (fp64) operations. The result is the IEEE remainder,
    computed as `arg0 - n * arg1` where `n` is the nearest integer
    to the exact value of `arg0 / arg1`.

    The operation is pure (no side effects) and supports element-wise computation
    on tensors.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK))
         y = tl.load(y_ptr + tl.arange(0, BLOCK))
         rem = tl.extra.cuda.libdevice.remainder(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK), rem)
```

---

### triton.language.extra.cuda.libdevice.rhadd

```python
rhadd(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.rhadd

**`rhadd(arg0, arg1, _semantic=None)`**

   Compute rounded half add: `(arg0 + arg1 + 1) >> 1`.

   Parameters
   ----------
   arg0 : tensor
       First input tensor. Must be `int32` or `uint32`.
   arg1 : tensor
       Second input tensor. Must be `int32` or `uint32`. Same dtype as `arg0`.
   _semantic : optional
       Internal semantic argument. Do not set manually.

   Returns
   -------
   tensor
       Tensor of same dtype as inputs containing `(arg0 + arg1 + 1) >> 1`.

   Notes
   -----
   This function maps to CUDA libdevice functions:

   - `__nv_rhadd` for `int32` inputs
   - `__nv_urhadd` for `uint32` inputs

   Computes the average of two integers with rounding toward positive infinity.
   Equivalent to `(arg0 + arg1 + 1) // 2` but avoids overflow in the addition.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
       pid = tl.program_id(0)
       offs = pid * BLOCK + tl.arange(0, BLOCK)
       x = tl.load(x_ptr + offs)
       y = tl.load(y_ptr + offs)
       result = tl.extra.cuda.libdevice.rhadd(x, y)
       tl.store(out_ptr + offs, result)
```

---

### triton.language.extra.cuda.libdevice.rhypot

```python
rhypot(arg0, arg1, _semantic=None)
```

## rhypot

Compute the reciprocal hypotenuse $1 / \sqrt{arg0^2 + arg1^2}$ of two arguments.

### Parameters
arg0 : tensor
    First input tensor. Must be floating-point type (`fp32` or `fp64`).
arg1 : tensor
    Second input tensor. Must be floating-point type (`fp32` or `fp64`).
    Must have the same dtype as `arg0`.
_semantic : optional
    Internal semantic parameter. Do not set manually.

### Returns
tensor
    Reciprocal hypotenuse of the inputs. Returns `fp32` for `fp32` inputs,
    `fp64` for `fp64` inputs.

### Notes
This function dispatches to CUDA libdevice functions `__nv_rhypotf` (for `fp32`)
or `__nv_rhypot` (for `fp64`). The operation is pure (no side effects).

More numerically stable than computing `1 / tl.sqrt(arg0 * arg0 + arg1 * arg1)`
for large input values.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def reciprocal_hypot_kernel(X, Y, Z, BLOCK: tl.constexpr):
     x = tl.load(X + tl.arange(0, BLOCK))
     y = tl.load(Y + tl.arange(0, BLOCK))
     z = tl.libdevice.rhypot(x, y)
     tl.store(Z + tl.arange(0, BLOCK), z)
```

---

### triton.language.extra.cuda.libdevice.rint

```python
rint(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.rint

Round input to the nearest integer value.

### Parameters
arg0 : tensor
    Input tensor of floating-point values (`fp32` or `fp64`).
_semantic : optional
    Internal parameter for semantic context. Do not set manually.

### Returns
tensor
    Tensor of the same shape as `arg0`, with each element rounded to the
    nearest integer. Returns `fp32` for `fp32` input and `fp64` for
    `fp64` input.

### Notes
This function wraps CUDA libdevice intrinsics: `__nv_rintf` for `fp32`
and `__nv_rint` for `fp64`. Rounding follows IEEE 754 standard
(round to nearest, ties to even).

The function is pure (no side effects) and operates element-wise on tensors.

### Examples
```python
 import triton
 import triton.language as tl
 from triton.language.extra.cuda import libdevice

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
     y = libdevice.rint(x)
     tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.rnorm3d

```python
rnorm3d(arg0, arg1, arg2, _semantic=None)
```

## triton.language.extra.cuda.libdevice.rnorm3d


**`rnorm3d(arg0, arg1, arg2, _semantic=None)`**

   Computes the reciprocal of the 3D Euclidean norm.

   Parameters
   ----------
   arg0 : tl.tensor
       First coordinate (x).
   arg1 : tl.tensor
       Second coordinate (y).
   arg2 : tl.tensor
       Third coordinate (z).
   _semantic : optional
       Internal semantic argument; do not set manually.

   Returns
   -------
   tl.tensor
       Reciprocal norm $1 / \sqrt{x^2 + y^2 + z^2}$. Same dtype as inputs.

   Notes
   -----
   This function dispatches to CUDA libdevice:

   - `__nv_rnorm3df` for `fp32` inputs
   - `__nv_rnorm3d` for `fp64` inputs

   All three arguments must have the same dtype (either all `fp32` or all `fp64`).
   The operation is pure (no side effects).

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(X, Y, Z, Out, BLOCK_SIZE: tl.constexpr):
        x = tl.load(X + tl.arange(0, BLOCK_SIZE))
        y = tl.load(Y + tl.arange(0, BLOCK_SIZE))
        z = tl.load(Z + tl.arange(0, BLOCK_SIZE))
        rnorm = tl.extra.cuda.libdevice.rnorm3d(x, y, z)
        tl.store(Out + tl.arange(0, BLOCK_SIZE), rnorm)
```

---

### triton.language.extra.cuda.libdevice.rnorm4d

```python
rnorm4d(arg0, arg1, arg2, arg3, _semantic=None)
```

## triton.language.extra.cuda.libdevice.rnorm4d


**`rnorm4d(arg0, arg1, arg2, arg3, _semantic=None)`**

   Compute the reciprocal of the 4D Euclidean norm.

   Computes `1 / sqrt(arg0^2 + arg1^2 + arg2^2 + arg3^2)` element-wise.

   Parameters
   ----------
   arg0 : tl.tensor
       First component of the 4D vector.
   arg1 : tl.tensor
       Second component of the 4D vector.
   arg2 : tl.tensor
       Third component of the 4D vector.
   arg3 : tl.tensor
       Fourth component of the 4D vector.

   Returns
   -------
   tl.tensor
       Reciprocal norm of the 4D vector. Has the same dtype as the inputs.

   Notes
   -----
   This function dispatches to CUDA libdevice:

   - `__nv_rnorm4df` for float32 inputs
   - `__nv_rnorm4d` for float64 inputs

   All inputs must have the same dtype (either all fp32 or all fp64). The function
   is pure (no side effects).

   Examples
   --------
```python
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, y_ptr, z_ptr, w_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
        y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
        z = tl.load(z_ptr + tl.arange(0, BLOCK_SIZE))
        w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE))
        rnorm = tl.extra.cuda.libdevice.rnorm4d(x, y, z, w)
        tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), rnorm)
```

---

### triton.language.extra.cuda.libdevice.round

```python
round(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.round

**`round(arg0, _semantic=None)`**

    Round a floating-point value to the nearest integer.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values. Supports `fp32` and `fp64`
        dtypes.
    _semantic : optional
        Internal parameter used by Triton compiler. Do not set manually.

    Returns
    -------
    tl.tensor
        Tensor of rounded values with the same dtype as `arg0`.

    Notes
    -----
    This function dispatches to CUDA libdevice routines: `__nv_roundf` for
    `fp32` inputs and `__nv_round` for `fp64` inputs. Rounding follows
    the standard round-to-nearest behavior.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = tl.extra.cuda.libdevice.round(x)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.rsqrt

```python
rsqrt(arg0, _semantic=None)
```

## rsqrt


**`rsqrt(arg0, _semantic=None)`**

    Compute the reciprocal square root $1/\sqrt{x}$ element-wise.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values. Must be `fp32` or `fp64` dtype.
    _semantic : optional
        Internal parameter. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor containing reciprocal square root values. Same dtype as `arg0`.

    Notes
    -----
    This function is an extern wrapper around CUDA libdevice reciprocal square root
    operations. It dispatches to the following PTX functions based on input dtype:

    - `__nv_rsqrtf` for 32-bit floating-point (`fp32`)
    - `__nv_rsqrt` for 64-bit floating-point (`fp64`)

    The operation is marked as pure (no side effects). Behavior for negative inputs
    and zero follows IEEE-754 standards (returns NaN for negative, inf for zero).

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = libdevice.rsqrt(x)
         tl.store(y_ptr + offsets, y)

 See Also
 --------
 tl.sqrt : Compute square root
 tl.libdevice.sqrt : CUDA libdevice square root
```

---

### triton.language.extra.cuda.libdevice.rsqrt_rn

```python
rsqrt_rn(arg0, _semantic=None)
```

**`triton.language.extra.cuda.libdevice.rsqrt_rn(arg0, _semantic=None)`**

   Compute the reciprocal square root of the input with round-to-nearest rounding.

   Parameters
   ----------
   arg0 : tensor
      Input tensor of floating-point values. Only `fp32` dtype is supported.
   _semantic : optional
      Internal semantic argument for Triton JIT compilation. Do not set manually.

   Returns
   -------
   tensor
      Tensor of type `fp32` containing `1 / sqrt(arg0)` for each element.

   Notes
   -----
   This function calls the CUDA libdevice function `__nv_frsqrt_rn` which
   computes reciprocal square root with IEEE round-to-nearest-even rounding mode.

   The result is undefined for negative inputs and zero inputs produce infinity.

   This is a pure function with no side effects.

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
       y = tl.extra.cuda.libdevice.rsqrt_rn(x)
       tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.sad

```python
sad(arg0, arg1, arg2, _semantic=None)
```

## triton.language.extra.cuda.libdevice.sad


**`sad(arg0, arg1, arg2, _semantic=None)`**

    Computes the sum of absolute differences (SAD) plus an accumulator.

    Computes `abs(arg0 - arg1) + arg2` using CUDA libdevice intrinsics.
    Supports both signed and unsigned 32-bit integer operations.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be of type `tl.int32` or `tl.uint32`.
    arg1 : tl.tensor
        Second input tensor. Must match the type of `arg0`.
    arg2 : tl.tensor
        Accumulator tensor. Must be of type `tl.uint32`.
    _semantic : optional
        Internal parameter for type semantics. Do not set directly.

    Returns
    -------
    result : tl.tensor
        Tensor of type `tl.int32` if inputs are signed, or `tl.uint32`
        if inputs are unsigned. Contains `abs(arg0 - arg1) + arg2` for each
        element.

    Notes
    -----
    This function maps to CUDA libdevice intrinsics:

    - `__nv_sad` for signed 32-bit integers (`int32`, `int32`, `uint32`)
    - `__nv_usad` for unsigned 32-bit integers (`uint32`, `uint32`, `uint32`)

    The operation is commonly used in computer vision for block matching and
    template matching algorithms.

    All input tensors are broadcasted to a common shape before the operation.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def sad_kernel(x_ptr, y_ptr, acc_ptr, out_ptr, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK + tl.arange(0, BLOCK)
         x = tl.load(x_ptr + offsets)
         y = tl.load(y_ptr + offsets)
         acc = tl.load(acc_ptr + offsets)
         result = tl.extra.cuda.libdevice.sad(x, y, acc)
         tl.store(out_ptr + offsets, result)
```

---

### triton.language.extra.cuda.libdevice.saturatef

```python
saturatef(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.saturatef


.. autofunction:: saturatef

Clamp floating-point values to the range $[0.0, 1.0]$.

### Parameters
arg0 : tensor
    Input tensor of type `tl.float32`.

### Returns
result : tensor
    Output tensor of type `tl.float32`. Each element is clamped to the
    range $[0.0, 1.0]$.

### Notes
Calls the CUDA libdevice function `__nv_saturatef`. This operation is
equivalent to `tl.minimum(tl.maximum(arg0, 0.0), 1.0)`. The function
is pure and has no side effects.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
     y = tl.extra.cuda.libdevice.saturatef(x)
     tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.scalbn

```python
scalbn(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.scalbn


**`scalbn(arg0, arg1, _semantic=None)`**

    Compute `arg0 * 2**arg1` efficiently.

    Parameters
    ----------
    arg0 : tl.tensor
        Floating-point input tensor. Must be of type `fp32` or `fp64`.
    arg1 : tl.tensor
        Integer exponent tensor. Must be of type `int32`.
    _semantic : optional
        Internal parameter for Triton semantic analysis. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of the same floating-point type as `arg0` containing
        `arg0 * 2**arg1`.

    Notes
    -----
    This function maps to CUDA libdevice functions:

    - `__nv_scalbnf` for `fp32` inputs
    - `__nv_scalbn` for `fp64` inputs

    More efficient than explicit multiplication by `2**arg1` as it
    directly manipulates the floating-point exponent.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def scale_kernel(X, Y, SCALE_EXP, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(X + offsets)
         exp = tl.load(SCALE_EXP + offsets)
         y = tl.extra.cuda.libdevice.scalbn(x, exp)
         tl.store(Y + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.signbit

```python
signbit(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.signbit

**`signbit(arg0, _semantic=None)`**

    Test element-wise whether the sign bit is set (i.e., the value is negative).

    Parameters
    ----------
    arg0 : tensor
        Input tensor of floating-point type. Must be fp32 or fp64.

    Returns
    -------
    out : tensor
        Tensor of int32 values. Each element is 1 if the corresponding input
        element has the sign bit set (negative or negative zero), 0 otherwise.

    Notes
    -----
    This function maps to CUDA libdevice functions `__nv_signbitf` for fp32
    and `__nv_signbitd` for fp64. Unlike comparison with zero, `signbit`
    correctly handles signed zeros (returns 1 for -0.0) and NaN values
    (returns 1 for negative NaN).

    The function is pure (no side effects) and supports element-wise
    broadcasting.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, out_ptr, n):
         offsets = tl.arange(0, n)
         x = tl.load(x_ptr + offsets)
         sign = tl.extra.cuda.libdevice.signbit(x)
         tl.store(out_ptr + offsets, sign)

     # Usage
     import torch
     x = torch.tensor([-1.0, 0.0, -0.0, 1.0, float('nan')], device='cuda')
     # signbit returns: [1, 0, 1, 0, 0] (depends on NaN bit pattern)
```

---

### triton.language.extra.cuda.libdevice.sin

```python
sin(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.sin


.. autofunction:: sin

Compute the sine of each element in the input tensor.

### Parameters
arg0 : tl.tensor
    Input tensor. Must have dtype `fp32` or `fp64`.
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
result : tl.tensor
    Tensor containing the sine of each element in `arg0`. Has the same dtype
    as `arg0`.

### Notes
This function is an external wrapper around CUDA libdevice sine functions:

- `__nv_sinf` for `fp32` inputs
- `__nv_sin` for `fp64` inputs

The operation is pure (no side effects) and supports element-wise computation
on tensors. Input values are expected to be in radians.

For `fp32` inputs, the result has single-precision accuracy. For `fp64`
inputs, the result has double-precision accuracy.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def sine_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.sin(x)
     tl.store(y_ptr + offsets, y)

 # Compute sine of values in radians
 import torch
 x = torch.linspace(0, 2 * torch.pi, 1024, device='cuda')
 y = torch.empty_like(x)
 sine_kernel[(1024 // 256,)](x, y, BLOCK_SIZE=256)
```

---

### triton.language.extra.cuda.libdevice.sinh

```python
sinh(arg0, _semantic=None)
```

## sinh

Compute the hyperbolic sine of the input element-wise.

### Parameters
arg0 : tensor
    Input tensor. Supported dtypes are `fp32` and `fp64`.
_semantic : optional
    Internal parameter, do not set manually.

### Returns
tensor
    Output tensor of the same shape as `arg0`. Element type is `fp32`
    if input is `fp32`, `fp64` if input is `fp64`.

### Notes
This function calls CUDA libdevice functions `__nv_sinhf` for `fp32`
and `__nv_sinh` for `fp64`. The operation is pure (no side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
     y = tl.extra.cuda.libdevice.sinh(x)
     tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.sinpi

```python
sinpi(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.sinpi


**`sinpi(arg0, _semantic=None)`**

    Compute `sin(pi * x)` element-wise.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values.

    Returns
    -------
    tl.tensor
        Tensor containing `sin(pi * x)` for each element in `arg0`.
        Has the same dtype as the input (fp32 or fp64).

    Notes
    -----
    This function calls CUDA libdevice implementations:

    - `__nv_sinpif` for float32 inputs
    - `__nv_sinpi` for float64 inputs

    Using `sinpi(x)` is more accurate than `sin(pi * x)` for large
    values of `x`, as it avoids precision loss in the multiplication.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def sinpi_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = tl.extra.cuda.libdevice.sinpi(x)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.sqrt

```python
sqrt(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.sqrt


**`sqrt(arg0, _semantic=None)`**

    Compute the element-wise square root of a floating-point tensor.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values. Must be of type `fp32` or `fp64`.
    _semantic : optional
        Internal parameter for semantic propagation. Do not set manually.

    Returns
    -------
    tl.tensor
        Tensor containing the square root of each element in `arg0`. The dtype
        matches the input dtype (`fp32` or `fp64`).

    Notes
    -----
    This function wraps CUDA libdevice `__nv_sqrtf` (for `fp32`) and
    `__nv_sqrt` (for `fp64`). It is a pure elementwise operation with no
    side effects.

    The function is only available on CUDA targets. For portable code, consider
    using `triton.language.sqrt()` instead.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def sqrt_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK + tl.arange(0, BLOCK)
         mask = offset < n
         x = tl.load(x_ptr + offset, mask=mask)
         y = tl.extra.cuda.libdevice.sqrt(x)
         tl.store(y_ptr + offset, y, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.sqrt_rd

```python
sqrt_rd(arg0, _semantic=None)
```

## sqrt_rd

Compute square root with round-down rounding mode.

### Parameters
arg0 : tensor
    Input tensor of floating-point type. Must be `fp32` or `fp64`.

### Returns
tensor
    Element-wise square root of `arg0` with round-down rounding mode.
    Same dtype as input.

### Notes
This function calls the CUDA libdevice functions `__nv_fsqrt_rd` for
`fp32` and `__nv_dsqrt_rd` for `fp64`. The round-down rounding mode
rounds the result toward negative infinity.

This is a pure function with no side effects.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
     y = tl.extra.cuda.libdevice.sqrt_rd(x)
     tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.sqrt_rn

```python
sqrt_rn(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.sqrt_rn


**`sqrt_rn(arg0, _semantic=None)`**

    Compute the square root of `arg0` with round-to-nearest rounding mode.

    This function calls the CUDA libdevice functions `__nv_fsqrt_rn` for
    32-bit floats and `__nv_dsqrt_rn` for 64-bit floats.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor. Must be of type `fp32` or `fp64`.
    _semantic : optional
        Internal parameter for Triton semantic handling. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of the same dtype as `arg0` containing the square root
        of each element with round-to-nearest rounding.

    Notes
    -----
    This is a low-level libdevice function that provides precise control over
    rounding mode. For most use cases, prefer :py`tl.sqrt()` which provides
    a higher-level interface.

    The function is pure (no side effects) and supports both 32-bit and 64-bit
    floating-point types.

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
         # Compute square root with round-to-nearest
         y = tl.libdevice.sqrt_rn(x)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.sqrt_ru

```python
sqrt_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.sqrt_ru


.. autofunction:: sqrt_ru

Compute the square root of each element in `arg0` with round-up rounding mode.

### Parameters
arg0 : tensor
    Input tensor of floating-point values. Supported dtypes are `fp32` and `fp64`.
_semantic : optional
    Internal parameter used by Triton compiler. Do not set manually.

### Returns
tensor
    Tensor of the same dtype as `arg0` containing the square root of each element,
    rounded upward according to IEEE 754 round-up rounding mode.

### Notes
This function calls CUDA libdevice functions `__nv_fsqrt_ru` (for `fp32`) and
`__nv_dsqrt_ru` (for `fp64`). The "ru" suffix indicates round-up rounding mode,
which rounds results toward positive infinity.

This is a pure function with no side effects. Results may differ from standard
:py`triton.language.sqrt()` due to the explicit rounding mode.

Only available on CUDA targets with libdevice support.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def sqrt_ru_kernel(x_ptr, y_ptr, n):
     pid = tl.program_id(0)
     offset = pid * tl.constexpr(256) + tl.arange(0, 256)
     mask = offset < n
     x = tl.load(x_ptr + offset, mask=mask)
     y = tl.extra.cuda.libdevice.sqrt_ru(x)
     tl.store(y_ptr + offset, y, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.sqrt_rz

```python
sqrt_rz(arg0, _semantic=None)
```

sqrt_rz(arg0, _semantic=None)

Compute square root of `arg0` with round-towards-zero rounding mode.

### Parameters
arg0 : tensor
    Input tensor of floating-point type (fp32 or fp64).
_semantic : optional
    Internal semantic parameter, do not set manually.

### Returns
result : tensor
    Square root of input with round-towards-zero rounding. Same dtype as input.

### Notes
This function calls the CUDA libdevice functions `__nv_fsqrt_rz` for
32-bit floats and `__nv_dsqrt_rz` for 64-bit floats. The rounding mode
is round-towards-zero (rz), which truncates the result towards zero.

This is a pure function with no side effects.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(x_ptr, out_ptr, n):
     pid = tl.program_id(0)
     offset = pid * 256
     x = tl.load(x_ptr + offset + tl.arange(0, 256))
     result = tl.extra.cuda.libdevice.sqrt_rz(x)
     tl.store(out_ptr + offset + tl.arange(0, 256), result)
```

---

### triton.language.extra.cuda.libdevice.sub_rd

```python
sub_rd(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.sub_rd


.. autofunction:: sub_rd

Subtract two floating-point values with round-down (toward negative infinity) rounding mode.

### Parameters
arg0 : tensor
    First input tensor. Must be of floating-point type (`fp32` or `fp64`).
arg1 : tensor
    Second input tensor. Must match the dtype of `arg0`.
_semantic : optional
    Internal semantic argument. Do not set manually.

### Returns
tensor
    Element-wise difference of `arg0 - arg1` with round-down rounding mode.
    The dtype matches the input dtype (`fp32` or `fp64`).

### Notes
This function wraps CUDA libdevice intrinsics:

- `__nv_fsub_rd` for 32-bit floating-point (`fp32`)
- `__nv_dsub_rd` for 64-bit floating-point (`fp64`)

The "rd" suffix indicates round-down rounding mode (round toward negative
infinity). This differs from the default round-to-nearest-even mode used by
standard subtraction operations.

This function is marked as pure, meaning it has no side effects and the same
inputs will always produce the same outputs.

Only `fp32` and `fp64` dtypes are supported. Other floating-point types
(e.g., `fp16`, `bf16`, `fp8`) are not supported by this libdevice
function.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def sub_rd_kernel(x_ptr, y_ptr, out_ptr, n,
                   BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offs < n
     x = tl.load(x_ptr + offs, mask=mask)
     y = tl.load(y_ptr + offs, mask=mask)
     # Subtract with round-down rounding mode
     result = tl.extra.cuda.libdevice.sub_rd(x, y)
     tl.store(out_ptr + offs, result, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.sub_rn

```python
sub_rn(arg0, arg1, _semantic=None)
```

## sub_rn


**`sub_rn(arg0, arg1, _semantic=None)`**

    Subtract two values with round-to-nearest rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be floating-point type (fp32 or fp64).
    arg1 : tl.tensor
        Second input tensor. Must be floating-point type (fp32 or fp64).
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor containing the element-wise difference `arg0 - arg1` with
        round-to-nearest rounding.

    Notes
    -----
    This function maps to CUDA libdevice functions:

    - `__nv_fsub_rn` for fp32 inputs
    - `__nv_dsub_rn` for fp64 inputs

    The `rn` suffix indicates round-to-nearest rounding mode, which is the
    default IEEE 754 rounding mode. Both inputs must have the same dtype.

    This operation is pure (no side effects) and supports elementwise
    broadcasting.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(x_ptr + offsets)
         y = tl.load(y_ptr + offsets)
         result = tl.extra.cuda.libdevice.sub_rn(x, y)
         tl.store(out_ptr + offsets, result)
```

---

### triton.language.extra.cuda.libdevice.sub_ru

```python
sub_ru(arg0, arg1, _semantic=None)
```

## sub_ru


**`sub_ru(arg0, arg1, _semantic=None)`**

    Subtract two floating-point values with round-up rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be fp32 or fp64 dtype.
    arg1 : tl.tensor
        Second input tensor. Must be fp32 or fp64 dtype.
    _semantic : optional
        Internal parameter, do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of the same dtype as inputs containing `arg0 - arg1` computed
        with round-up rounding mode (round toward positive infinity).

    Notes
    -----
    This function maps to CUDA libdevice functions:

    - `__nv_fsub_ru` for fp32 inputs
    - `__nv_dsub_ru` for fp64 inputs

    The round-up rounding mode rounds the exact result toward positive infinity.
    This can produce different results than standard subtraction for values near
    rounding boundaries.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = tl.load(y_ptr + tl.arange(0, BLOCK_SIZE))
         result = tl.extra.cuda.libdevice.sub_ru(x, y)
         tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

---

### triton.language.extra.cuda.libdevice.sub_rz

```python
sub_rz(arg0, arg1, _semantic=None)
```

## triton.language.extra.cuda.libdevice.sub_rz


**`sub_rz(arg0, arg1, _semantic=None)`**

    Subtract two floating-point values with round-towards-zero rounding mode.

    Performs `arg0 - arg1` using IEEE floating-point subtraction with the
    round-towards-zero (rz) rounding mode. This is a CUDA libdevice intrinsic
    that maps to `__nv_fsub_rz` for float32 and `__nv_dsub_rz` for float64.

    Parameters
    ----------
    arg0 : tl.tensor
        First input tensor. Must be of type `tl.float32` or `tl.float64`.
    arg1 : tl.tensor
        Second input tensor. Must be of the same dtype as `arg0`.
    _semantic : optional
        Internal parameter for Triton semantics. Do not set manually.

    Returns
    -------
    result : tl.tensor
        Tensor of the same dtype as inputs containing `arg0 - arg1` computed
        with round-towards-zero rounding.

    Notes
    -----
    The round-towards-zero (rz) rounding mode truncates results toward zero,
    which differs from the default round-to-nearest-even mode. This can be
    useful for reproducibility or when specific rounding behavior is required.

    Both inputs must have the same floating-point precision (both fp32 or both
    fp64). Mixed precision is not supported.

    This function is pure (no side effects) and can be used in device code
    within `@triton.jit` decorated kernels.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel_sub_rz(X, Y, Z, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         x = tl.load(X + offs)
         y = tl.load(Y + offs)
         # Subtract with round-towards-zero
         z = tl.extra.cuda.libdevice.sub_rz(x, y)
         tl.store(Z + offs, z)
```

---

### triton.language.extra.cuda.libdevice.tan

```python
tan(arg0, _semantic=None)
```

## tan


**`tan(arg0, _semantic=None)`**

    Compute the tangent of each element in the input tensor.

    Parameters
    ----------
    arg0 : tensor
        Input tensor of floating-point values. Supported dtypes are `fp32` and
        `fp64`.
    _semantic : optional
        Internal semantic argument. Do not set manually.

    Returns
    -------
    out : tensor
        Tensor of the same shape and dtype as `arg0`, containing the tangent
        of each element.

    Notes
    -----
    This function is a wrapper around CUDA libdevice functions:

    - `__nv_tanf` for `fp32` inputs
    - `__nv_tan` for `fp64` inputs

    The operation is pure (no side effects). Input values where cosine is zero
    (e.g., $\pi/2 + k\pi$) will produce infinity or NaN.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def tan_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         x = tl.load(input_ptr + offsets)
         y = tl.extra.cuda.libdevice.tan(x)
         tl.store(output_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.tanh

```python
tanh(arg0, _semantic=None)
```

## tanh

Compute the hyperbolic tangent of the input element-wise.

### Parameters
arg0 : tensor
    Input tensor. Must have floating-point dtype (`fp32` or `fp64`).
_semantic :
    Internal parameter, do not set manually.

### Returns
tensor
    Output tensor of the same dtype as `arg0`, containing `tanh(arg0)`.

### Notes
This function is implemented via CUDA libdevice:

- `fp32` inputs call `__nv_tanhf`
- `fp64` inputs call `__nv_tanh`

The operation is pure (no side effects) and supports element-wise computation on tensors.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def tanh_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
     pid = tl.program_id(0)
     offset = pid * BLOCK + tl.arange(0, BLOCK)
     mask = offset < n
     x = tl.load(x_ptr + offset, mask=mask)
     y = tl.tanh(x)
     tl.store(y_ptr + offset, y, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.tgamma

```python
tgamma(arg0, _semantic=None)
```

## tgamma

Compute the gamma function element-wise.

.. math::
    \Gamma(x) = \int_0^\infty t^{x-1} e^{-t} dt

### Parameters
arg0 : tensor
    Input tensor of floating-point type (`fp32` or `fp64`).

### Returns
result : tensor
    Tensor of the same dtype as `arg0` containing $\Gamma(x)$ values.

### Notes
The gamma function extends the factorial function to real numbers. For positive
integers $n$, $\Gamma(n) = (n-1)!$.

This function is implemented using CUDA libdevice:
- `__nv_tgammaf` for `fp32` inputs
- `__nv_tgamma` for `fp64` inputs

The function is pure (no side effects) and supports automatic broadcasting.

### Examples
```python
 import torch
 import triton
 import triton.language as tl

 @triton.jit
 def gamma_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(axis=0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
     x = tl.load(x_ptr + offsets, mask=mask)
     out = tl.extra.cuda.libdevice.tgamma(x)
     tl.store(out_ptr + offsets, out, mask=mask)

 # Compute gamma function for input values
 x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device='cuda')
 out = torch.empty_like(x)
 gamma_kernel[(1,)](x, out, x.numel(), BLOCK_SIZE=4)
 # out = [1.0, 1.0, 2.0, 6.0]  # Γ(1)=0!=1, Γ(2)=1!=1, Γ(3)=2!=2, Γ(4)=3!=6
```

---

### triton.language.extra.cuda.libdevice.trunc

```python
trunc(arg0, _semantic=None)
```

## trunc


**`trunc(arg0, _semantic=None)`**

    Truncate a floating-point value towards zero.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of floating-point values. Must be fp32 or fp64.

    Returns
    -------
    result : tl.tensor
        Tensor of truncated values with the same dtype as `arg0`.

    Notes
    -----
    This function dispatches to CUDA libdevice: `__nv_trunc` for fp64 and
    `__nv_truncf` for fp32. Truncation rounds towards zero, discarding the
    fractional part (e.g., `trunc(3.7) = 3`, `trunc(-3.7) = -3`).

    The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         x = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE))
         y = libdevice.trunc(x)
         tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), y)
```

---

### triton.language.extra.cuda.libdevice.uint2double_rn

```python
uint2double_rn(arg0, _semantic=None)
```

## uint2double_rn


.. autofunction:: uint2double_rn

Convert unsigned 32-bit integer to double-precision float with round-to-nearest.

### Parameters
arg0 : tensor
    Input tensor of unsigned 32-bit integers (`uint32`).
_semantic : optional
    Internal semantic parameter (automatically set by Triton compiler).

### Returns
tensor
    Tensor of double-precision floating-point values (`fp64`).

### Notes
This function invokes the NVIDIA libdevice intrinsic `__nv_uint2double_rn`,
which performs conversion with round-to-nearest-even rounding mode.

The operation is pure (no side effects) and element-wise.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def convert_kernel(input_ptr, output_ptr, n):
     pid = tl.program_id(0)
     offsets = tl.arange(0, n)
     uint_vals = tl.load(input_ptr + offsets)
     double_vals = tl.extra.cuda.libdevice.uint2double_rn(uint_vals)
     tl.store(output_ptr + offsets, double_vals)
```

---

### triton.language.extra.cuda.libdevice.uint2float_rd

```python
uint2float_rd(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.uint2float_rd


**`uint2float_rd(arg0, _semantic=None)`**

    Convert unsigned 32-bit integer to 32-bit float with round-down rounding mode.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of unsigned 32-bit integers (`tl.uint32`).
    _semantic : optional
        Internal semantic argument (automatically provided by Triton compiler).

    Returns
    -------
    tl.tensor
        Tensor of 32-bit floating-point values (`tl.fp32`).

    Notes
    -----
    This function wraps the NVIDIA libdevice function `__nv_uint2float_rd`.
    The conversion uses round-down (toward negative infinity) rounding mode,
    which differs from the default round-to-nearest-even mode.

    This is a pure function with no side effects.

    The function is only available on CUDA targets with libdevice support.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
         uint_vals = tl.load(x_ptr + offsets)
         float_vals = libdevice.uint2float_rd(uint_vals)
         tl.store(y_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.uint2float_rn

```python
uint2float_rn(arg0, _semantic=None)
```

## uint2float_rn


**`uint2float_rn(arg0, _semantic=None)`**

   Convert unsigned 32-bit integer to 32-bit float using round-to-nearest rounding.

   Parameters
   ----------
   arg0 : tl.tensor
       Input tensor of unsigned 32-bit integers (`tl.uint32`).
   _semantic
       Internal parameter, do not set manually.

   Returns
   -------
   tl.tensor
       Tensor of 32-bit floating-point values (`tl.fp32`).

   Notes
   -----
   This function invokes the CUDA libdevice function `__nv_uint2float_rn`.
   The rounding mode is round-to-nearest-even (RN).
   This operation is pure (no side effects).
   Only `uint32` input type is supported.

   Example
   -------
```python
   import triton
   import triton.language as tl
   from triton.language.extra.cuda import libdevice

   @triton.jit
   def uint_to_float_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       uint_vals = tl.load(x_ptr + offsets)
       float_vals = libdevice.uint2float_rn(uint_vals)
       tl.store(y_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.uint2float_ru

```python
uint2float_ru(arg0, _semantic=None)
```

## uint2float_ru


**`uint2float_ru(arg0, _semantic=None)`**

   Convert unsigned 32-bit integer to 32-bit float using round-up rounding mode.

   Calls the CUDA libdevice function `__nv_uint2float_ru` which performs
   unsigned integer to float conversion with round-up (toward positive infinity)
   rounding semantics.

   Parameters
   ----------
   arg0 : tensor
       Input tensor of unsigned 32-bit integers (`uint32` dtype).

   _semantic : optional
       Internal semantic argument used by Triton compiler. Do not set manually.

   Returns
   -------
   tensor
       Tensor of 32-bit floating-point values (`fp32` dtype) with the same
       shape as the input.

   Notes
   -----
   This function is pure (no side effects) and compiles to a single GPU
   instruction using CUDA libdevice. The rounding mode is round-up (toward
   positive infinity), which differs from the default round-to-nearest-even
   mode used by standard type casting.

   This function is only available on CUDA targets. For other targets, use
   standard type casting via :py`triton.language.cast()`.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = tl.program_id(0)
       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
       uint_vals = tl.load(x_ptr + offsets)
       float_vals = tl.extra.cuda.libdevice.uint2float_ru(uint_vals)
       tl.store(y_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.uint2float_rz

```python
uint2float_rz(arg0, _semantic=None)
```

## uint2float_rz


.. autofunction:: uint2float_rz

Converts an unsigned 32-bit integer to a 32-bit floating-point value using round-towards-zero rounding mode.

### Parameters
arg0 : tl.tensor
    Input tensor of unsigned 32-bit integers (`tl.uint32`).

### Returns
tl.tensor
    Tensor of 32-bit floating-point values (`tl.fp32`).

### Notes
This function calls the CUDA libdevice function `__nv_uint2float_rz` which performs
unsigned integer to float conversion with round-towards-zero (RTZ) rounding mode.

The conversion is exact for all uint32 values that can be represented exactly in fp32.
For values requiring rounding, the result is rounded towards zero (truncation).

This function is pure (no side effects) and can be used in device code within
:py`@triton.jit <triton.jit>()` decorated kernels.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def convert_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     uint_vals = tl.load(input_ptr + offsets)
     float_vals = tl.extra.cuda.libdevice.uint2float_rz(uint_vals)
     tl.store(output_ptr + offsets, float_vals)

 # Usage
 import torch
 input_tensor = torch.tensor([0, 100, 1000, 2**32 - 1], dtype=torch.uint32, device='cuda')
 output_tensor = torch.empty_like(input_tensor, dtype=torch.float32)
 convert_kernel[(1,)](input_ptr, output_ptr, BLOCK_SIZE=4)
```

---

### triton.language.extra.cuda.libdevice.uint_as_float

```python
uint_as_float(arg0, _semantic=None)
```

## uint_as_float


**`uint_as_float(arg0, _semantic=None)`**

    Reinterpret the bit pattern of an unsigned 32-bit integer as a 32-bit float.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of uint32 values.
    _semantic : optional
        Internal parameter used by Triton compiler. Do not provide manually.

    Returns
    -------
    tl.tensor
        Tensor of fp32 values with the same bit pattern as the input.

    Notes
    -----
    This function performs a bitcast operation, reinterpreting the binary
    representation of uint32 values as IEEE 754 single-precision floating
    point numbers. No numerical conversion is performed - the bits are
    simply reinterpreted.

    This is a CUDA libdevice function that maps to `__nv_uint_as_float`
    in PTX. The operation is pure (no side effects).

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK_SIZE
         x = tl.load(x_ptr + offset).to(tl.uint32)
         y = tl.extra.cuda.libdevice.uint_as_float(x)
         tl.store(y_ptr + offset, y)

 See Also
 --------
 float_as_int : Reinterpret float32 bit pattern as uint32
 bitcast : General bitcast operation in Triton
```

---

### triton.language.extra.cuda.libdevice.ull2double_rd

```python
ull2double_rd(arg0, _semantic=None)
```

## ull2double_rd

Convert an unsigned 64-bit integer to double-precision float with round-down rounding.

### Parameters
arg0 : tensor
    Input tensor of unsigned 64-bit integers (`tl.uint64`).

### Returns
tensor
    Tensor of double-precision floating point values (`tl.fp64`).

### Notes
This function calls the CUDA libdevice function `__nv_ull2double_rd` which converts
unsigned 64-bit integers to double-precision floating point values using round-down
(round toward negative infinity) rounding mode. This differs from the default
round-to-nearest-even mode used by standard integer-to-float conversions.

This is an external function that requires CUDA libdevice support. The function is
pure (no side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(ptr_in, ptr_out, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     uint64_vals = tl.load(ptr_in + offsets)
     # Convert with round-down rounding mode
     float64_vals = tl.extra.cuda.libdevice.ull2double_rd(uint64_vals)
     tl.store(ptr_out + offsets, float64_vals)
```

---

### triton.language.extra.cuda.libdevice.ull2double_rn

```python
ull2double_rn(arg0, _semantic=None)
```

ull2double_rn(arg0, _semantic=None)
    Convert unsigned 64-bit integer to double-precision floating-point value.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of unsigned 64-bit integers (uint64).
    _semantic : optional
        Internal parameter, do not use directly.

    Returns
    -------
    result : tl.tensor
        Tensor of double-precision floating-point values (fp64).

    Notes
    -----
    This function uses the CUDA libdevice intrinsic `__nv_ull2double_rn` which
    performs conversion with round-to-nearest-even rounding mode.

    The operation is pure (no side effects) and can be used in device code.

    This function is only available on CUDA targets.

    Examples
    --------
```python
     import triton
     import triton.language as tl
     from triton.language.extra.cuda import libdevice

     @triton.jit
     def convert_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offset = pid * BLOCK
         uint_vals = tl.load(x_ptr + offset).to(tl.uint64)
         double_vals = libdevice.ull2double_rn(uint_vals)
         tl.store(y_ptr + offset, double_vals)
```

---

### triton.language.extra.cuda.libdevice.ull2double_ru

```python
ull2double_ru(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.ull2double_ru

**`triton.language.extra.cuda.libdevice.ull2double_ru(arg0, _semantic=None)`**

    Convert an unsigned 64-bit integer to double-precision float with round-upward rounding.

    Parameters
    ----------
    arg0 : tl.tensor
        Input tensor of unsigned 64-bit integers (`tl.uint64`).

    _semantic : optional
        Internal semantic argument (do not set manually).

    Returns
    -------
    tl.tensor
        Tensor of double-precision floating-point values (`tl.float64`).

    Notes
    -----
    This function calls the NVIDIA libdevice intrinsic `__nv_ull2double_ru`, which
    converts unsigned 64-bit integers to IEEE 754 double-precision floating-point
    values using round-toward-positive-infinity rounding mode (`ru`).

    The rounding mode affects conversion when the integer cannot be represented
    exactly in floating-point format. Round-upward means the result is rounded
    toward positive infinity.

    This is a pure function with no side effects.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         uint_vals = tl.load(ptr + offsets).to(tl.uint64)
         float_vals = tl.extra.cuda.libdevice.ull2double_ru(uint_vals)
         tl.store(ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.ull2double_rz

```python
ull2double_rz(arg0, _semantic=None)
```

## ull2double_rz


**`ull2double_rz(arg0, _semantic=None)`**

    Convert unsigned 64-bit integer to double-precision float with round-towards-zero.

    Parameters
    ----------
    arg0 : tl.uint64
        Unsigned 64-bit integer value to convert.

    Returns
    -------
    tl.fp64
        Double-precision floating-point value representing the input integer,
        rounded towards zero.

    Notes
    -----
    This function calls the CUDA libdevice intrinsic `__nv_ull2double_rz`.
    The rounding mode is round-towards-zero (rz), which truncates any fractional
    part when converting integers that cannot be exactly represented in fp64.

    This is a pure function with no side effects.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
         offsets = tl.arange(0, BLOCK_SIZE)
         uint_vals = tl.load(ptr + offsets).to(tl.uint64)
         float_vals = tl.extra.cuda.libdevice.ull2double_rz(uint_vals)
         tl.store(out_ptr + offsets, float_vals)
```

---

### triton.language.extra.cuda.libdevice.ull2float_rd

```python
ull2float_rd(arg0, _semantic=None)
```

## ull2float_rd


**`ull2float_rd(arg0, _semantic=None)`**

   Convert unsigned 64-bit integer to float32 with round-down rounding mode.

   Parameters
   ----------
   arg0 : tensor of uint64
       Input unsigned 64-bit integer tensor.
   _semantic : optional
       Internal semantic argument (do not pass explicitly).

   Returns
   -------
   tensor of fp32
       Float32 tensor with values converted from uint64 using round-down rounding.

   Notes
   -----
   This is an extern function that wraps the CUDA libdevice function `__nv_ull2float_rd`.
   The "rd" suffix indicates round-down (toward negative infinity) rounding mode for the
   integer-to-float conversion.

   This function is pure and elementwise, operating independently on each element of the
   input tensor.

   Examples
   --------
```python
   import triton
   import triton.language as tl

   @triton.jit
   def kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
       offsets = tl.arange(0, BLOCK_SIZE)
       uint64_vals = tl.load(input_ptr + offsets).to(tl.uint64)
       float32_vals = tl.extra.cuda.libdevice.ull2float_rd(uint64_vals)
       tl.store(output_ptr + offsets, float32_vals)
```

---

### triton.language.extra.cuda.libdevice.ull2float_rn

```python
ull2float_rn(arg0, _semantic=None)
```

## ull2float_rn


.. autofunction:: ull2float_rn

Convert an unsigned 64-bit integer to a 32-bit float with round-to-nearest rounding.

### Parameters
arg0 : tensor of uint64
    Input tensor containing unsigned 64-bit integers to convert.
_semantic : optional
    Internal Triton semantic argument (do not set manually).

### Returns
tensor of fp32
    Tensor containing the converted 32-bit floating-point values.

### Notes
This function calls the NVIDIA libdevice function `__nv_ull2float_rn`.
The `rn` suffix indicates round-to-nearest-even rounding mode.
Values that cannot be exactly represented in fp32 will be rounded.
Large uint64 values may lose precision when converted to fp32 due to
the smaller mantissa width (23 bits for fp32 vs 64 bits for uint64).

This function is only available on CUDA targets.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def convert_kernel(x_ptr, y_ptr, n):
     pid = tl.program_id(0)
     offsets = pid * 4 + tl.arange(0, 4)
     mask = offsets < n
     uint64_vals = tl.load(x_ptr + offsets, mask=mask)
     float32_vals = tl.extra.cuda.libdevice.ull2float_rn(uint64_vals)
     tl.store(y_ptr + offsets, float32_vals, mask=mask)
```

---

### triton.language.extra.cuda.libdevice.ull2float_ru

```python
ull2float_ru(arg0, _semantic=None)
```

## ull2float_ru


**`ull2float_ru(arg0, _semantic=None)`**

    Convert unsigned 64-bit integer to float32 with round-up rounding mode.

    Parameters
    ----------
    arg0 : tl.uint64
        Input unsigned 64-bit integer value or tensor.
    _semantic : optional
        Internal semantic argument, do not set manually.

    Returns
    -------
    tl.float32
        Float32 result of converting arg0 with round-up (toward +inf) rounding.

    Notes
    -----
    This is a CUDA libdevice function that wraps `__nv_ull2float_ru`.
    The conversion uses IEEE 754 round-up (toward positive infinity) rounding mode.
    Only available on CUDA targets.

    For large uint64 values that cannot be exactly represented in float32,
    the result is rounded up to the next representable float32 value.

    Examples
    --------
```python
     import triton
     import triton.language as tl

     @triton.jit
     def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
         pid = tl.program_id(0)
         offsets = pid * BLOCK + tl.arange(0, BLOCK)
         x = tl.load(x_ptr + offsets).to(tl.uint64)
         y = tl.extra.cuda.libdevice.ull2float_ru(x)
         tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.ull2float_rz

```python
ull2float_rz(arg0, _semantic=None)
```

## ull2float_rz

Convert unsigned 64-bit integer to 32-bit float with round-towards-zero rounding.

### Parameters
arg0 : tl.tensor
    Input tensor of unsigned 64-bit integers (`tl.uint64`).
_semantic : optional
    Internal parameter, do not set manually.

### Returns
tl.tensor
    Tensor of 32-bit floating-point values (`tl.float32`).

### Notes
Calls the CUDA libdevice function `__nv_ull2float_rz`. Uses round-towards-zero
(rz) rounding mode for the conversion, which truncates the value toward zero.
This is equivalent to the C cast `(float)(unsigned long long)` with rz rounding.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
     offsets = tl.arange(0, BLOCK)
     x = tl.load(in_ptr + offsets).to(tl.uint64)
     y = tl.extra.cuda.libdevice.ull2float_rz(x)
     tl.store(out_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.y0

```python
y0(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.y0

Compute the Bessel function of the second kind of order 0.

### Parameters
arg0 : tl.tensor
    Input tensor of floating-point values. Must be `fp32` or `fp64` dtype.

### Returns
result : tl.tensor
    Tensor of the same dtype as `arg0` containing the Y0 Bessel function
    values. For `fp32` input, returns `fp32`; for `fp64` input, returns
    `fp64`.

### Notes
This function is a wrapper around CUDA libdevice intrinsics. It dispatches to
`__nv_y0f` for `fp32` inputs and `__nv_y0` for `fp64` inputs.

The Bessel function of the second kind Y0(x) is singular at x=0 (approaches
negative infinity) and oscillates with decaying amplitude for x>0. The function
is undefined for x<0.

This is an external elementwise operation marked as pure (no side effects).

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def bessel_y0_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     y = tl.extra.cuda.libdevice.y0(x)
     tl.store(y_ptr + offsets, y)
```

---

### triton.language.extra.cuda.libdevice.y1

```python
y1(arg0, _semantic=None)
```

## triton.language.extra.cuda.libdevice.y1

Compute the Bessel function of the second kind of order 1.

### Parameters
arg0 : tl.tensor
    Input tensor. Must be of floating-point type (`fp32` or `fp64`).
_semantic :
    Internal parameter, do not use.

### Returns
tl.tensor
    Tensor of the same type as `arg0` containing the Y1 Bessel function
    values.

### Notes
This function is a wrapper around CUDA libdevice functions:

- `__nv_y1f` for 32-bit floating-point (`fp32`)
- `__nv_y1` for 64-bit floating-point (`fp64`)

The Bessel function of the second kind (Y1) is a solution to Bessel's
differential equation. It is singular at `x = 0`.

### Examples
```python
 import torch
 import triton
 import triton.language as tl

 @triton.jit
 def y1_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
     offsets = tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
     x = tl.load(x_ptr + offsets, mask=mask)
     y = tl.extra.cuda.libdevice.y1(x)
     tl.store(y_ptr + offsets, y, mask=mask)

 n_elements = 1024
 x = torch.randn(n_elements, device='cuda', dtype=torch.float32)
 y = torch.empty(n_elements, device='cuda', dtype=torch.float32)
 y1_kernel[(1,)](x, y, n_elements, BLOCK_SIZE=1024)
```

---

### triton.language.extra.cuda.libdevice.yn

```python
yn(arg0, arg1, _semantic=None)
```

yn
==

Compute the Bessel function of the second kind of order `n`.

### Parameters
arg0 : tl.int32
    The order of the Bessel function.
arg1 : tl.float32 or tl.float64
    The input value at which to evaluate the Bessel function.
_semantic : optional
    Internal parameter for Triton semantic handling. Do not set manually.

### Returns
tl.float32 or tl.float64
    The value of the Bessel function of the second kind `Y_n(arg1)`.
    Returns float32 if arg1 is float32, float64 if arg1 is float64.

### Notes
This function computes the Bessel function of the second kind (also known as
the Weber function or Neumann function). It is implemented using CUDA libdevice
functions: `__nv_ynf` for single precision and `__nv_yn` for
double precision.

The function is pure (no side effects) and supports elementwise operations on
tensors.

For large orders or arguments, numerical precision may degrade. The function
is undefined for negative integer orders in some implementations.

### Examples
```python
 import triton
 import triton.language as tl

 @triton.jit
 def bessel_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     x = tl.load(x_ptr + offsets)
     # Compute Y_2(x) - Bessel function of second kind, order 2
     y = tl.extra.cuda.libdevice.yn(2, x)
     tl.store(out_ptr + offsets, y)
```

---
