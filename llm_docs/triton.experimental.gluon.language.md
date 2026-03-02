# triton.experimental.gluon.language

Gluon language primitives — layouts, shared memory, barriers, TMA, warpgroup MMA, and hardware-specific ops (NVIDIA Ampere/Hopper/Blackwell, AMD CDNA3/CDNA4/RDNA3/RDNA4/GFX1250).

*222 APIs documented.*

---

## triton.experimental.gluon.language

### triton.experimental.gluon.language.AutoLayout

```python
AutoLayout() -> None
```

**`AutoLayout()`**

   Represents a distributed memory layout where the compiler automatically infers the distribution strategy.

   Notes
   -----
   Inherits from `DistributedLayout`. Unlike explicit layouts such as `BlockedLayout`,
   `AutoLayout` does not require specifying tiling parameters. It signals the backend to choose
   an optimal layout based on the target hardware and operation.

   The `rank` property is undefined for `AutoLayout`. Accessing it raises a `ValueError`.
   The mangled representation is `"AL"`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # Instantiate an automatic layout
   layout = ttgl.AutoLayout()

   # Query mangle string
   assert layout.mangle() == "AL"

   # Accessing rank raises an error
   try:
       _ = layout.rank
   except ValueError:
       pass
```

---

### triton.experimental.gluon.language.BlockedLayout

```python
BlockedLayout(size_per_thread: List[int], threads_per_warp: List[int], warps_per_cta: List[int], order: List[int], cga_layout: List[List[int]] = <factory>) -> None
```

**`BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order, cga_layout=None)`**

   Represents a blocked layout, partitioning a tensor across threads, warps, and CTAs.

   This layout distributes tensor elements hierarchically across the GPU execution model:
   elements are first partitioned per thread, then grouped into warps, and finally
   organized into cooperative thread arrays (CTAs). The `order` parameter controls
   which dimensions are partitioned first.

   Parameters
   ----------
   size_per_thread : List[int]
       Number of elements each thread holds per dimension. Determines the register
       footprint per thread.
   threads_per_warp : List[int]
       Number of threads per warp per dimension. Typically matches hardware warp
       size (e.g., 32 for NVIDIA GPUs).
   warps_per_cta : List[int]
       Number of warps per CTA per dimension. Controls how warps are distributed
       across the thread block.
   order : List[int]
       Dimension ordering for partitioning. Specifies which dimensions are
       partitioned first (e.g., [1, 0] partitions dimension 1 before dimension 0).
   cga_layout : List[List[int]], optional
       Bases describing how CTAs tile each dimension. Used for multi-CTA
       cooperation (default is empty list).

   Attributes
   ----------
   size_per_thread : List[int]
   threads_per_warp : List[int]
   warps_per_cta : List[int]
   order : List[int]
   cga_layout : List[List[int]]
   rank : int
       The tensor rank (number of dimensions), derived from `order`.

   Notes
   -----
   All list parameters must have the same length, equal to the tensor rank.
   The class automatically unwraps constexpr values during initialization.

   `BlockedLayout` is commonly used for defining register file layouts in
   Gluon kernels, particularly for matrix multiplication and other tiled
   operations where explicit control over data distribution is required.

   Examples
   --------
   Create a 2D blocked layout for a matrix tile:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # 2D layout: 4 elements per thread, 32 threads per warp, 4 warps per CTA
   layout = ttgl.BlockedLayout(
       size_per_thread=[4, 4],
       threads_per_warp=[1, 32],
       warps_per_cta=[4, 1],
       order=[1, 0]
   )

   print(layout.rank)  # Output: 2

Use with a kernel to specify register layout:

.. code-block:: python

   @gluon.jit
   def kernel(...):
       # Use layout for distributed memory operations
       pass
```

---

### triton.experimental.gluon.language.CoalescedLayout

```python
CoalescedLayout() -> None
```

## class triton.experimental.gluon.language.CoalescedLayout

**`CoalescedLayout()`**

   Represents a coalesced distributed memory layout in Gluon IR.

   CoalescedLayout provides a simple distribution strategy where tensor elements
   are distributed across threads in a coalesced access pattern. This layout is
   typically used when memory access patterns benefit from contiguous, sequential
   memory transactions across the thread hierarchy.

   Notes
   -----
   `CoalescedLayout` has no rank and cannot be queried for dimensionality.
   Attempting to access the `rank` property will raise a `ValueError`.

   This layout is converted to Gluon IR via `builder.get_coalesced_layout()`
   and is mangled as `"CL"` for kernel serialization.

   Examples
   --------
   Create a coalesced layout for use in kernel definitions:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel():
       layout = ttgl.CoalescedLayout()
       # Use layout in tensor operations
       # ...

See Also
--------
AutoLayout : Automatic layout selection
BlockedLayout : Blocked distribution across threads, warps, and CTAs
DistributedLayout : Base class for distributed memory layouts
```

---

### triton.experimental.gluon.language.DistributedLinearLayout

```python
DistributedLinearLayout(reg_bases: List[List[int]], lane_bases: List[List[int]], warp_bases: List[List[int]], block_bases: List[List[int]], shape: List[int]) -> None
```

**`triton.experimental.gluon.language.DistributedLinearLayout(reg_bases: List[List[int]], lane_bases: List[List[int]], warp_bases: List[List[int]], block_bases: List[List[int]], shape: List[int]) -> None`**

   Represents a linear distributed layout with explicit bases at register, lane, warp, and block levels.

   Parameters
   ----------
   reg_bases : list[list[int]]
       Bases for register-level distribution.
   lane_bases : list[list[int]]
       Bases for lane-level distribution.
   warp_bases : list[list[int]]
       Bases for warp-level distribution.
   block_bases : list[list[int]]
       Bases for block-level distribution.
   shape : list[int]
       The tensor global shape.

   Notes
   -----
   Inherits from `DistributedLayout`. This layout allows fine-grained control
   over data distribution across the hardware hierarchy. All basis lists must match
   the rank of `shape`. Validation ensures that each basis vector length equals
   the tensor rank.

   For theoretical background, refer to: https://arxiv.org/abs/2505.23819

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # Define bases for a rank-2 tensor
   reg_bases = [[1, 0], [0, 1]]
   lane_bases = [[2, 0], [0, 2]]
   warp_bases = [[4, 0], [0, 4]]
   block_bases = [[8, 0], [0, 8]]
   shape = [16, 16]

   layout = ttgl.DistributedLinearLayout(
       reg_bases, lane_bases, warp_bases, block_bases, shape
   )

   # Access the rank
   print(layout.rank)  # Output: 2
```

---

### triton.experimental.gluon.language.DotOperandLayout

```python
DotOperandLayout(operand_index: int, parent: triton.experimental.gluon.language._layouts.DistributedLayout, k_width: int) -> None
```

## class triton.experimental.gluon.language.DotOperandLayout

Represents a layout for a dot operand in matrix multiplication operations.

This layout wraps a parent distributed layout (typically an MMA layout) and specifies
which operand (LHS or RHS) of the dot operation it represents, along with the
packing width for elements.

### Parameters
operand_index : int
    Index identifying the dot operand: 0 for the left-hand side (LHS) matrix,
    1 for the right-hand side (RHS) matrix.
parent : DistributedLayout
    The parent distributed layout representing the underlying MMA operation.
    Typically an instance of `NVMMADistributedLayout`.
k_width : int
    Number of elements packed per 32 bits. Controls the element packing density
    for the operand.

### Attributes
operand_index : int
    The operand index (0 for LHS, 1 for RHS).
parent : DistributedLayout
    The parent MMA layout.
k_width : int
    Elements per 32-bit word.

### Notes
`DotOperandLayout` is used to describe how tensor operands are distributed
across threads and warps for tensor core operations. The `operand_index`
determines which dimension of the parent layout corresponds to the reduction
(K) dimension:

- For operand 0 (LHS), the K dimension is the last dimension (rank - 1)
- For operand 1 (RHS), the K dimension is the second-to-last dimension (rank - 2)

The `cga_layout` property derives from the parent's CGA layout by zeroing
out the K dimension basis vectors.

### Examples
Create a dot operand layout for the LHS matrix with an MMA parent:

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 # Create parent MMA layout
 mma_layout = ttgl.NVMMADistributedLayout(
     version=[1, 0],
     warps_per_cta=[4, 1],
     instr_shape=[16, 8, 16],
 )

 # Create LHS operand layout (operand_index=0)
 lhs_layout = ttgl.DotOperandLayout(
     operand_index=0,
     parent=mma_layout,
     k_width=8,
 )

 # Create RHS operand layout (operand_index=1)
 rhs_layout = ttgl.DotOperandLayout(
     operand_index=1,
     parent=mma_layout,
     k_width=8,
 )

 # Access layout properties
 print(lhs_layout.rank)        # Inherits rank from parent
 print(lhs_layout.operand_index)  # 0
 print(rhs_layout.operand_index)  # 1
```

---

### triton.experimental.gluon.language.NVMMADistributedLayout

```python
NVMMADistributedLayout(version: List[int], warps_per_cta: List[int], instr_shape: List[int], cga_layout: List[List[int]] = <factory>) -> None
```

## class triton.experimental.gluon.language.NVMMADistributedLayout

**`NVMMADistributedLayout(version, warps_per_cta, instr_shape, cga_layout=[])`**

   Represents a distributed layout for NVIDIA MMA (tensor core) operations.

   This layout defines how tensor data is partitioned across warps and CTAs for
   matrix multiply-accumulate instructions on NVIDIA GPUs. It is used to configure
   the distribution of operands and results for tensor core operations in Gluon
   kernels.

   Parameters
   ----------
   version : list of int
       Version identifier for the MMA instruction. Specifies the GPU architecture
       and instruction set version (e.g., `[8, 9]` for Hopper H100).
   warps_per_cta : list of int
       Number of warps per CTA (thread block) for each dimension. Determines how
       warps are distributed across the tensor dimensions.
   instr_shape : list of int
       Instruction shape for the MMA operation. Defines the tile size processed
       by a single tensor core instruction (e.g., `[16, 8, 16]` for FP16).
   cga_layout : list of list of int, optional
       Bases describing CTA tiling for cooperative group assembly (CGA). Default
       is empty list. Used for multi-CTA cooperation on Hopper and later
       architectures.

   Attributes
   ----------
   version : list of int
       The MMA instruction version.
   warps_per_cta : list of int
       The warp distribution per CTA.
   instr_shape : list of int
       The MMA instruction tile shape.
   cga_layout : list of list of int
       The CTA tiling bases.
   rank : int
       The rank (number of dimensions) of the layout, derived from
       `len(warps_per_cta)`.

   Notes
   -----
   This layout is specifically designed for NVIDIA tensor core operations and
   should be paired with compatible shared memory layouts (e.g.,
   `NVMMASharedLayout`) for optimal performance.

   The `version` parameter encodes the GPU architecture generation:
   - `[8, 0]`: Ampere (A100)
   - `[8, 9]`: Hopper (H100)
   - `[9, 0]`: Blackwell (B200)

   Examples
   --------
   Create an MMA layout for Hopper H100 with 4 warps per CTA:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # Hopper H100 MMA layout with 4 warps, 16x8x16 tile
   mma_layout = ttgl.NVMMADistributedLayout(
       version=[8, 9],
       warps_per_cta=[4, 1],
       instr_shape=[16, 8, 16],
   )

   # Use in a kernel with explicit layout control
   @gluon.jit
   def kernel(...):
       # Kernel implementation using mma_layout
       pass

Create an MMA layout with CGA for multi-CTA cooperation:

.. code-block:: python

   # H100 with 2 CTAs cooperating via CGA
   mma_layout = ttgl.NVMMADistributedLayout(
       version=[8, 9],
       warps_per_cta=[4, 1],
       instr_shape=[16, 8, 16],
       cga_layout=[[1, 0], [0, 2]],  # 2 CTAs in second dimension
   )

See Also
--------
:class:`NVMMASharedLayout` : Shared memory layout for NVIDIA MMA operations
:class:`BlockedLayout` : General blocked distributed layout
:class:`DistributedLinearLayout` : Linear distributed layout with explicit bases
```

---

### triton.experimental.gluon.language.NVMMASharedLayout

```python
NVMMASharedLayout(swizzle_byte_width: int, element_bitwidth: int, rank: int = 2, transposed: bool = False, fp4_padded: bool = False, cga_layout: List[List[int]] = <factory>) -> None
```

## class triton.experimental.gluon.language.NVMMASharedLayout

Shared memory layout optimized for NVIDIA tensor core (MMA) operations.

This layout configures shared memory with swizzling patterns compatible with
NVIDIA Hopper/Blackwell MMA instructions, supporting optional FP4 padding and
CTA tiling via CGA layouts.

### Parameters
swizzle_byte_width : int
    Width in bytes for memory swizzling. Must be one of `[0, 32, 64, 128]`.
    Larger values reduce bank conflicts but require alignment.
element_bitwidth : int
    Bitwidth of the element type. Must be one of `[8, 16, 32, 64]`.
rank : int, optional
    Rank of the tensor (default is 2).
transposed : bool, optional
    Whether the layout is transposed (default is `False`).
fp4_padded : bool, optional
    Whether FP4 padding is used (default is `False`). When enabled,
    requires `swizzle_byte_width=128` and `element_bitwidth=8`.
cga_layout : list of list of int, optional
    Bases describing CTA tiling for multi-CTA cooperation (default is empty list).
    Each inner list must have length equal to `rank`.

### Notes
- `element_bitwidth` must be in `[8, 16, 32, 64]`.
- `swizzle_byte_width` must be in `[0, 32, 64, 128]`.
- When `fp4_padded=True`, `swizzle_byte_width` must be 128 and
  `element_bitwidth` must be 8.
- If `cga_layout` is provided, each basis vector must have length equal to
  `rank`.
- Use `get_default_for()` to automatically select an optimal swizzle pattern
  for a given block shape and dtype.

### Examples
Create a basic NVMMASharedLayout for 16-bit elements with 64-byte swizzling:

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 layout = ttgl.NVMMASharedLayout(
     swizzle_byte_width=64,
     element_bitwidth=16,
     rank=2,
     transposed=False,
 )

```
Create a layout with FP4 padding (requires 128-byte swizzle and 8-bit elements):

```python
 layout_fp4 = ttgl.NVMMASharedLayout(
     swizzle_byte_width=128,
     element_bitwidth=8,
     rank=2,
     fp4_padded=True,
 )

```
Use the helper method to get an optimal layout for a given block shape:

```python
 block_shape = [128, 64]
 dtype = ttgl.float16
 layout_auto = ttgl.NVMMASharedLayout.get_default_for(
     block_shape=block_shape,
     dtype=dtype,
     transposed=False,
 )

```
Create a layout with CGA tiling for multi-CTA cooperation:

```python
 cga_layout = [[1, 0], [0, 1]]  # Identity tiling bases
 layout_cga = ttgl.NVMMASharedLayout(
     swizzle_byte_width=128,
     element_bitwidth=32,
     rank=2,
     cga_layout=cga_layout,
 )
```

---

### triton.experimental.gluon.language.PaddedSharedLayout

```python
PaddedSharedLayout(interval_padding_pairs: List[List[int]], offset_bases: List[List[int]], cga_layout: List[List[int]], shape: List[int]) -> None
```

## class PaddedSharedLayout


.. autoclass:: PaddedSharedLayout

   Represents a shared memory layout that combines padding and element reordering via
   linear transformation to avoid shared memory bank conflicts.

   Parameters
   ----------
   interval_padding_pairs : List[List[int]]
       List of [interval, padding] pairs. Both interval and padding values must be
       powers of 2. After every `interval` tensor elements, `padding` elements
       are inserted. If a position corresponds to multiple intervals, the padding
       amounts are summed.
   offset_bases : List[List[int]]
       Bases for shared memory offsets. Each basis is a list of length equal to
       the tensor rank, defining the linear remapping from 1-D shared memory offset
       to logical n-D tensor elements.
   cga_layout : List[List[int]], optional
       Bases for block-level shared memory offsets. Default is empty list.
   shape : List[int]
       The n-D logical shared memory shape.

   Notes
   -----
   Compared to `SwizzledSharedLayout`, this layout combines padding with
   element reordering through linear transformations (e.g., row permutation) to
   prevent shared memory bank conflicts.

   The encoding allows for a linear remapping from 1-D shared memory offsets to
   logical n-D tensor elements. The remapping is specified via linear bases that
   map offsets to tensor dimensions.

   All interval and padding values must be powers of 2. The tensor rank must be
   greater than 0.

   Examples
   --------
   Create a padded shared layout with custom interval-padding pairs:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # Define interval-padding pairs: [interval, padding]
   interval_padding_pairs = [[2, 1], [4, 2]]

   # Define offset bases for 2D tensor (rank=2)
   offset_bases = [[1, 0], [0, 1]]

   # Define shape
   shape = [8, 4]

   layout = ttgl.PaddedSharedLayout(
       interval_padding_pairs=interval_padding_pairs,
       offset_bases=offset_bases,
       cga_layout=[],
       shape=shape
   )

Use the convenience constructor for identity mapping:

.. code-block:: python

   # Create layout with identity linear mapping
   layout = ttgl.PaddedSharedLayout.with_identity_for(
       interval_padding_pairs=[[2, 2]],
       shape=[8],
       order=[0],
       cga_layout=[]
   )

.. rubric:: Methods

.. autosummary::

   ~PaddedSharedLayout.with_identity_for
   ~PaddedSharedLayout.verify
   ~PaddedSharedLayout._to_ir
   ~PaddedSharedLayout.mangle

.. automethod:: with_identity_for

   Returns a :class:`PaddedSharedLayout` with the given interval and padding
   pairs and an identity mapping as the linear component for the given shape
   and order.

   Parameters
   ----------
   interval_padding_pairs : List[List[int]]
       List of [interval, padding] pairs.
   shape : List[int]
       The tensor shape. All dimensions must be powers of 2.
   order : List[int]
       Dimension ordering for the identity mapping.
   cga_layout : List[List[int]], optional
       Bases for block-level shared memory offsets. Default is empty list.

   Returns
   -------
   PaddedSharedLayout
       A new padded shared layout instance with identity linear mapping.

   Notes
   -----
   This convenience method constructs offset bases automatically based on
   the shape and order, creating an identity linear transformation.

.. automethod:: verify

   Validates the layout configuration.

   Raises
   ------
   AssertionError
       If interval_padding_pairs is empty, intervals or paddings are not
       powers of 2, or shape rank is invalid.
```

---

### triton.experimental.gluon.language.SharedLinearLayout

```python
SharedLinearLayout(offset_bases: List[List[int]], block_bases: List[List[int]] = <factory>, alignment: int = 16) -> None
```

**`SharedLinearLayout(offset_bases: list[list[int]], block_bases: list[list[int]] = [], alignment: int = 16)`**

   Represents a shared memory layout defined via an explicit LinearLayout.

   Parameters
   ----------
   offset_bases : list[list[int]]
       Bases for shared memory offsets mapping to tensor dimensions. Each inner 
       list corresponds to a basis vector. Must not be empty. The rank of the 
       layout is inferred from the length of the first basis.
   block_bases : list[list[int]], optional
       Bases for block-level (CTA) shared memory offsets. Defaults to empty list.
   alignment : int, optional
       Alignment requirement in bytes. Must be a positive power of two. Defaults 
       to 16.

   Notes
   -----
   Defines a linear mapping from shared memory offsets to logical tensor indices. 
   All bases in `offset_bases` and `block_bases` must match the rank derived 
   from `offset_bases[0]`. Validation fails if `alignment` is not a positive 
   power of two.

   Examples
   --------
   Define a 2D shared memory layout with specific offset bases:

```python
   import triton.experimental.gluon.language as ttgl

   # Define bases for a 2D layout [dim0, dim1]
   offset_bases = [[1, 0], [0, 1]]
   layout = ttgl.SharedLinearLayout(offset_bases=offset_bases, alignment=32)
```

---

### triton.experimental.gluon.language.SliceLayout

```python
SliceLayout(dim: int, parent: triton.experimental.gluon.language._layouts.DistributedLayout) -> None
```

## class triton.experimental.gluon.language.SliceLayout

Represents a layout corresponding to slicing a distributed tensor along one dimension.

### Parameters
dim : int
    The dimension index to slice. Must satisfy `0 <= dim < parent.rank`.
parent : DistributedLayout
    The parent layout before slicing. Determines the rank and CGA layout of the
    sliced layout.

### Attributes
dim : int
    The dimension index that was sliced.
parent : DistributedLayout
    The original layout before slicing.
rank : int
    The rank of the sliced layout, equal to `parent.rank - 1`.
cga_layout : List[List[int]]
    The CGA layout with the sliced dimension removed from each basis vector.

### Notes
`SliceLayout` is used to represent the layout of a tensor after slicing one
dimension from a distributed layout. The resulting layout has rank one less than
the parent layout. The CGA layout bases have the sliced dimension component
removed from each basis vector.

The layout is immutable (frozen dataclass) and unwraps constexpr values during
initialization.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 # Create a blocked layout with rank 2
 parent_layout = ttgl.BlockedLayout(
     size_per_thread=[4, 8],
     threads_per_warp=[16, 1],
     warps_per_cta=[4, 1],
     order=[1, 0],
 )

 # Slice along dimension 0, resulting in rank 1 layout
 sliced_layout = ttgl.SliceLayout(dim=0, parent=parent_layout)

 print(sliced_layout.rank)  # Output: 1
 print(sliced_layout.dim)   # Output: 0
```

---

### triton.experimental.gluon.language.SwizzledSharedLayout

```python
SwizzledSharedLayout(vec: int, per_phase: int, max_phase: int, order: List[int], cga_layout: List[List[int]] = <factory>) -> None
```

## class triton.experimental.gluon.language.SwizzledSharedLayout

Represents a generic swizzled shared memory layout.

Swizzled layouts reorder shared memory accesses to avoid bank conflicts by
applying a swizzle function based on address bits. This layout provides
explicit control over swizzling parameters for fine-grained memory access
optimization.

### Parameters
vec : int
    Vector width for swizzling. Determines the granularity of vectorized
    memory operations.
per_phase : int
    Number of elements per swizzle phase. Controls how elements are
    distributed across swizzle phases.
max_phase : int
    Maximum number of swizzle phases. Defines the swizzle pattern period.
order : List[int]
    Dimension ordering for swizzling. Specifies which tensor dimensions
    participate in the swizzle computation.
cga_layout : List[List[int]], optional
    Bases describing CTA (Cooperative Thread Array) tiling. Each inner
    list represents a basis vector for block-level memory partitioning.
    Default is empty list (no CGA layout).

### Notes
This layout is hardware-agnostic and works across NVIDIA (Hopper/Blackwell)
and AMD (CDNA/RDNA) GPU architectures. The swizzle parameters must be
chosen to match the target hardware's shared memory bank structure.

All parameters are unwrapped from constexpr types during initialization
via `__post_init__`.

The layout implements `_to_ir` for Gluon IR code generation, `mangle`
for unique string representation, and `__hash__` for use in dictionaries.

### Examples
Create a basic swizzled shared layout:

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 # Simple 1D swizzle layout
 layout = ttgl.SwizzledSharedLayout(
     vec=8,
     per_phase=4,
     max_phase=8,
     order=[0]
 )

 # 2D layout with CGA tiling
 layout_2d = ttgl.SwizzledSharedLayout(
     vec=16,
     per_phase=8,
     max_phase=16,
     order=[0, 1],
     cga_layout=[[1, 0], [0, 1]]
 )

 # Use in a kernel with shared memory allocation
 @gluon.jit
 def kernel(X, Y, BLOCK_SIZE: tl.constexpr):
     sm_ptr = ttgl.alloc(
         shape=[BLOCK_SIZE],
         layout=layout,
         dtype=tl.float32
     )
     # ... kernel implementation

```
### See Also
NVMMASharedLayout : NVIDIA-specific shared memory layout for MMA operations
PaddedSharedLayout : Layout with padding to avoid bank conflicts
SharedLinearLayout : Explicit linear layout for shared memory

---

### triton.experimental.gluon.language.abs

```python
abs(x, _semantic=None)
```

## abs

Compute the element-wise absolute value of a tensor.

**`abs(x, _semantic=None)`**

   Computes the element-wise absolute value of `x`.

   Parameters
   ----------
   x : tensor
       The input tensor. Can be any dtype (floating, signed integer, unsigned integer, or fp8).
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually; automatically provided by the Gluon JIT.

   Returns
   -------
   tensor
       A tensor containing the absolute value of each element in `x`. Has the same shape and dtype as the input.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`, as `x.abs()` instead of `abs(x)`.

   For unsigned integer types, this operation is a no-op since values are already non-negative.

   For fp8e4b15 dtype, absolute value is computed via bitwise AND with 0x7F mask. For other floating types, uses FABS instruction. For signed integers, uses IABS instruction.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid * BLOCK_SIZE)
       y = ttgl.abs(x)  # or equivalently: y = x.abs()
       ttgl.store(y_ptr + pid * BLOCK_SIZE, y)
```

---

### triton.experimental.gluon.language.add

```python
add(x, y, sanitize_overflow: 'constexpr' = True, _semantic=None)
```

Element-wise addition of two tensors.

### Parameters
x : tensor or scalar
    Left hand side operand.
y : tensor or scalar
    Right hand side operand.
sanitize_overflow : constexpr, optional
    If True, enables overflow sanitization. Default is True.

### Returns
tensor
    Element-wise sum of `x` and `y`.

### Notes
This function must be called within a kernel decorated with `@gluon.jit`.
Standard broadcasting rules apply if `x` and `y` have different shapes.
The `_semantic` argument is internal and should not be provided by users.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def add_kernel(X, Y, Z, SIZE: ttgl.constexpr):
     offs = ttgl.arange(0, SIZE)
     x = ttgl.load(X + offs)
     y = ttgl.load(Y + offs)
     z = ttgl.add(x, y)
     ttgl.store(Z + offs, z)
```

---

### triton.experimental.gluon.language.allocate_shared_memory

```python
allocate_shared_memory(element_ty, shape, layout, value=None, _semantic=None) -> 'shared_memory_descriptor'
```

Allocate shared memory for a tensor with specified element type, shape, and layout.

Allocates a shared memory region accessible by all threads in a CTA. Returns a descriptor that can be used for subsequent shared memory operations (load, store, gather, scatter, etc.).

### Parameters
element_ty : dtype
    The element data type of the tensor (e.g., `ttgl.float16`, `ttgl.int32`).
shape : Sequence[int]
    The dimensions of the shared memory allocation.
layout : SharedLayout
    The shared memory layout specifying how data is organized in memory
    (e.g., `BlockedLayout`, `SwizzledLayout`).
value : tensor, optional
    Initial tensor value to copy into the allocated shared memory. If None,
    memory is allocated but not initialized.
_semantic : GluonSemantic, optional
    Internal semantic handler. Do not provide manually; this is set
    automatically by the `@gluon.jit` decorator.

### Returns
shared_memory_descriptor
    A descriptor handle for the allocated shared memory region. This descriptor
    provides methods for loading, storing, and transforming the shared memory
    view.

### Notes
This function must be called within a `@gluon.jit` decorated kernel. The
`_semantic` parameter is automatically injected by the JIT compiler and
should not be provided by users.

Shared memory allocated this way is visible to all threads in the CTA and
persists for the kernel's lifetime. Proper synchronization (e.g., via
`gluon.barrier()`) is required when coordinating access between threads.

The returned descriptor supports operations such as `load()`, `store()`,
`gather()`, `scatter()`, `slice()`, `reshape()`, and `permute()`.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language._layouts import BlockedLayout

 @gluon.jit
 def kernel(...):
     # Allocate 128-element float16 shared memory with blocked layout
     layout = BlockedLayout([128], [32], [4], [1], [0], [0])
     shared = ttgl.allocate_shared_memory(ttgl.float16, [128], layout)

     # Initialize with zeros
     zeros = ttgl.full([128], 0.0, ttgl.float16, layout)
     shared.store(zeros)

     # Load from shared memory with distributed layout
     dist_layout = ttgl.AutoLayout()
     data = shared.load(dist_layout)

     # Use data in computation
     ...
```

---

### triton.experimental.gluon.language.arange

```python
arange(start, end, layout=None, _semantic=None)
```

## arange


**`arange(start, end, layout=None)`**

   Generate a sequence tensor with values in the half-open interval `[start, end)`.

   Creates a 1D tensor containing sequential integer values from `start` (inclusive) to
   `end` (exclusive), distributed according to the specified layout.

   Parameters
   ----------
   start : int
       Inclusive start of the sequence.
   end : int
       Exclusive end of the sequence.
   layout : DistributedLayout, optional
       The layout of the output tensor. Defaults to `AutoLayout` if not specified.

   Returns
   -------
   tensor
       A 1D tensor containing sequential integer values from `start` to `end - 1`.

   Notes
   -----
   The `layout` parameter controls how the sequence values are distributed across
   threads and warps. When using Gluon kernels, explicit layout specification enables
   fine-grained control over memory access patterns and thread-level parallelism.

   This function must be called within a kernel decorated with `@gluon.jit()`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(X, stride, BLOCK_SIZE: tl.constexpr):
       # Generate sequence [0, BLOCK_SIZE) with default AutoLayout
       offsets = ttgl.arange(0, BLOCK_SIZE)

       # Generate sequence with explicit coalesced layout
       from triton.experimental.gluon.language._layouts import CoalescedLayout
       offsets = ttgl.arange(0, BLOCK_SIZE, layout=CoalescedLayout())

       # Use offsets for memory access
       pid = ttgl.program_id(0)
       mask = offsets < stride
       x = ttgl.load(X + pid * stride + offsets, mask=mask)
```

---

### triton.experimental.gluon.language.associative_scan

```python
associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None)
```

## associative_scan


.. autofunction:: associative_scan

Apply an associative scan operation along a specified axis of input tensors.

This function applies a combining function cumulatively to elements along an axis,
propagating a carry value through the sequence. It supports both forward and reverse
scan directions, and can operate on single tensors or tuples of tensors.

### Parameters
input : tensor or tuple of tensors
    The input tensor(s) to scan. When a tuple is provided, the scan is applied
    element-wise across all tensors simultaneously.
axis : int
    The dimension along which to perform the scan operation.
combine_fn : callable
    A JIT-compiled function (decorated with `@gluon.jit`) that combines two
    scalar tensor values. Must accept two arguments and return the combined result.
reverse : bool, optional
    If `True`, perform the scan in reverse order along the axis (default: `False`).

### Returns
tensor or tuple of tensors
    The scanned output tensor(s) with the same shape as the input. Each position
    contains the cumulative combination of all elements up to that position along
    the specified axis.

### Notes
The `combine_fn` must be associative for the scan to produce correct results.
Common operations include addition, multiplication, or custom reduction operations.

This function can also be called as a member method on tensor objects:
`x.associative_scan(axis, combine_fn, reverse=False)` instead of
`associative_scan(x, axis, combine_fn, reverse=False)`.

The scan operation is parallelized across warps and is optimized for GPU execution.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def add_fn(a, b):
     return a + b

 @gluon.jit
 def kernel(x_ptr, stride):
     # Load input tensor
     x = ttgl.load(x_ptr)
     
     # Perform cumulative sum along axis 0
     result = ttgl.associative_scan(x, axis=0, combine_fn=add_fn)
     
     # Store result
     ttgl.store(x_ptr, result)

 # Using as a member function
 @gluon.jit
 def kernel_member(x_ptr, stride):
     x = ttgl.load(x_ptr)
     result = x.associative_scan(axis=0, combine_fn=add_fn)
     ttgl.store(x_ptr, result)

 # Reverse scan
 @gluon.jit
 def kernel_reverse(x_ptr, stride):
     x = ttgl.load(x_ptr)
     result = ttgl.associative_scan(x, axis=0, combine_fn=add_fn, reverse=True)
     ttgl.store(x_ptr, result)

 # Scan on tuple of tensors
 @gluon.jit
 def kernel_tuple(x_ptr, y_ptr, stride):
     x = ttgl.load(x_ptr)
     y = ttgl.load(y_ptr)
     result = ttgl.associative_scan((x, y), axis=0, combine_fn=add_fn)
     ttgl.store(x_ptr, result[0])
     ttgl.store(y_ptr, result[1])
```

---

### triton.experimental.gluon.language.assume

```python
assume(cond, _semantic=None)
```

## assume

Allow the compiler to assume a condition is true for optimization purposes.

### Parameters
cond : tensor
    A boolean tensor representing the condition to assume is true.
_semantic : GluonSemantic, optional
    Internal semantic handler (automatically provided by the JIT compiler).

### Returns
None

### Notes
This function inserts a compiler hint that the given condition is always true at
this program point. This can enable downstream optimizations such as dead code
elimination, loop unrolling, or memory access simplification. Use with caution:
if the condition is false at runtime, behavior is undefined.

The `_semantic` parameter is automatically injected by the `@gluon.jit`
decorator and should not be provided by users.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, n):
     pid = ttgl.program_id(0)
     # Assume pid is always less than n
     ttgl.assume(pid < n)
     # Safe to load without bounds check
     value = ttgl.load(x_ptr + pid)
     ...
```

---

### triton.experimental.gluon.language.atomic_add

```python
atomic_add(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_add


**`atomic_add(pointer, val, mask=None, sem=None, scope=None, _semantic=None)`**

   Performs an atomic addition at the memory location specified by `pointer`.

   Returns the data stored at `pointer` before the atomic operation completes.

   Parameters
   ----------
   pointer : tensor
      Block of pointers with `dtype=triton.PointerDType` specifying the memory
      locations to operate on.
   val : tensor
      Block of values with `dtype=pointer.dtype.element_ty` to add atomically.
   mask : tensor, optional
      Boolean tensor specifying which elements to operate on. If `None`, all
      elements are processed.
   sem : str, optional
      Memory semantics for the operation. Acceptable values are `"acquire"`,
      `"release"`, `"acq_rel"`, and `"relaxed"`. Defaults to `"acq_rel"`.
   scope : str, optional
      Scope of threads observing the synchronizing effect. Acceptable values are
      `"gpu"` (default), `"cta"` (cooperative thread array), or `"sys"`
      (system-wide).

   Returns
   -------
   tensor
      Tensor containing the values stored at `pointer` before the atomic
      addition.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_add(...)` instead of `atomic_add(x, ...)`.

   The memory semantics (`sem`) control ordering constraints:

   - `"acquire"`: Subsequent memory operations cannot be reordered before this
     operation.
   - `"release"`: Prior memory operations cannot be reordered after this
     operation.
   - `"acq_rel"`: Combines both acquire and release semantics.
   - `"relaxed"`: No ordering constraints.

   The scope (`scope`) defines visibility:

   - `"gpu"`: Visible across the entire GPU device.
   - `"cta"`: Visible only within the cooperative thread array (thread block).
   - `"sys"`: Visible system-wide across all devices.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def atomic_add_kernel(ptr, val, n):
       pid = ttgl.program_id(0)
       offset = pid
       old_val = ttgl.atomic_add(ptr + offset, val)
       # old_val contains the value before addition

.. code-block:: python

   # Using member function syntax
   @gluon.jit
   def atomic_add_member_kernel(ptr, val, n):
       pid = ttgl.program_id(0)
       offset = pid
       old_val = (ptr + offset).atomic_add(val)

.. code-block:: python

   # With memory semantics and scope
   @gluon.jit
   def atomic_add_semantics_kernel(ptr, val, n):
       pid = ttgl.program_id(0)
       offset = pid
       old_val = ttgl.atomic_add(
           ptr + offset,
           val,
           sem="acq_rel",
           scope="gpu"
       )
```

---

### triton.experimental.gluon.language.atomic_and

```python
atomic_and(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_and


**`atomic_and(pointer, val, mask=None, sem=None, scope=None, _semantic=None)`**

   Performs an atomic logical AND at the memory location specified by `pointer`.

   Returns the data stored at `pointer` before the atomic operation.

   Parameters
   ----------
   pointer : tensor
       The memory locations to operate on. Block of `dtype=triton.PointerDType`.
   val : tensor
       The values with which to perform the atomic operation. Block of `dtype=pointer.dtype.element_ty`.
   mask : tensor, optional
       Boolean mask to control which elements are updated. If `None`, all elements are updated.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are `"acquire"`,
       `"release"`, `"acq_rel"` (stands for "ACQUIRE_RELEASE"), and `"relaxed"`.
       Defaults to `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the atomic operation.
       Acceptable values are `"gpu"` (default), `"cta"` (cooperative thread array, thread block),
       or `"sys"` (stands for "SYSTEM"). Defaults to `"gpu"`.

   Returns
   -------
   tensor
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_and(...)` instead of `atomic_and(x, ...)`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, val):
       # Atomic AND: x_ptr[0] = x_ptr[0] & val
       old_val = ttgl.atomic_and(x_ptr, val)
```

---

### triton.experimental.gluon.language.atomic_cas

```python
atomic_cas(pointer, cmp, val, sem=None, scope=None, _semantic=None)
```

## atomic_cas

Performs an atomic compare-and-swap operation at the specified memory location.

Atomically compares the value at `pointer` with `cmp`. If they are equal, stores `val` 
to `pointer`. Returns the value that was stored at `pointer` before the operation.

### Parameters
pointer : tensor
    Pointer to the memory location to operate on. Must be a block of pointer type 
    (`triton.PointerDType`).
cmp : tensor
    The expected value to compare against. Must have the same dtype as the element type 
    of `pointer`.
val : tensor
    The value to store if the comparison succeeds. Must have the same dtype as the element 
    type of `pointer`.
sem : str, optional
    Memory semantics for the operation. Acceptable values are `"acquire"`, `"release"`, 
    `"acq_rel"` (ACQUIRE_RELEASE), and `"relaxed"`. Defaults to `"acq_rel"`.
scope : str, optional
    Scope of threads that observe the synchronizing effect of the atomic operation. 
    Acceptable values are `"gpu"` (default), `"cta"` (cooperative thread array/thread 
    block), or `"sys"` (SYSTEM). Defaults to `"gpu"`.

### Returns
tensor
    The value stored at `pointer` before the atomic operation.

### Notes
This function can also be called as a member function on :py`tensor` objects, as 
`x.atomic_cas(...)` instead of `atomic_cas(x, ...)`.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def atomic_update_kernel(ptr, expected, new_val):
     # Perform atomic compare-and-swap
     old = ttgl.atomic_cas(ptr, expected, new_val)
     # old contains the value before the operation
```

---

### triton.experimental.gluon.language.atomic_max

```python
atomic_max(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_max


**`atomic_max(pointer, val, mask=None, sem=None, scope=None, _semantic=None)`**

   Performs an atomic maximum operation at the memory location specified by `pointer`.

   Returns the data stored at `pointer` before the atomic operation.

   Parameters
   ----------
   pointer : tensor
       The memory locations to operate on. Must be a block of `triton.PointerDType`.
   val : tensor
       The values with which to perform the atomic operation. Must be a block of
       `pointer.dtype.element_ty`.
   mask : tensor, optional
       Boolean mask to control which threads participate in the atomic operation.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are
       `"acquire"`, `"release"`, `"acq_rel"`, and `"relaxed"`. Defaults to
       `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the
       atomic operation. Acceptable values are `"gpu"` (default), `"cta"`
       (cooperative thread array), or `"sys"` (system).

   Returns
   -------
   tensor
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_max(...)` instead of `atomic_max(x, ...)`.

   The operation performs: `old = *pointer; *pointer = max(*pointer, val)`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, val, n: tl.constexpr):
       pid = ttgl.program_id(0)
       ptr = x_ptr + pid
       old = ttgl.atomic_max(ptr, val)
```

---

### triton.experimental.gluon.language.atomic_min

```python
atomic_min(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_min


**`atomic_min(pointer, val, mask=None, sem=None, scope=None, _semantic=None)`**

    Performs an atomic minimum operation at the specified memory location.

    Atomically compares the value at `pointer` with `val` and stores the
    minimum of the two. Returns the original value stored at `pointer` before
    the operation.

    Parameters
    ----------
    pointer : tensor
        Pointer to the memory location(s) to operate on. Must be a block of
        `triton.PointerDType`.
    val : tensor
        The value(s) to compare against. Must be a block with dtype matching
        `pointer.dtype.element_ty`.
    mask : tensor, optional
        Boolean mask to control which elements participate in the atomic
        operation. If `None`, all elements are processed.
    sem : str, optional
        Memory semantics for the operation. Acceptable values are `"acquire"`,
        `"release"`, `"acq_rel"`, and `"relaxed"`. Defaults to `"acq_rel"`.
    scope : str, optional
        Scope of threads that observe the synchronizing effect. Acceptable values
        are `"gpu"` (default), `"cta"`, or `"sys"`.

    Returns
    -------
    tensor
        The data stored at `pointer` before the atomic operation.

    Notes
    -----
    This function can also be called as a member function on :py`tensor`,
    as `x.atomic_min(...)` instead of `atomic_min(x, ...)`.

    The operation is performed element-wise for block tensors. Each thread
    operates on its corresponding element in the block.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(x_ptr, y_ptr, n: tl.constexpr):
         pid = ttgl.program_id(0)
         x = ttgl.load(x_ptr + pid)
         val = 5.0
         old_val = ttgl.atomic_min(x_ptr + pid, val)
         ttgl.store(y_ptr + pid, old_val)

     # Using member function syntax
     @gluon.jit
     def kernel_member(x_ptr, n: tl.constexpr):
         pid = ttgl.program_id(0)
         ptr = x_ptr + pid
         old_val = ptr.atomic_min(5.0)
```

---

### triton.experimental.gluon.language.atomic_or

```python
atomic_or(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_or


**`atomic_or(pointer, val, mask=None, sem=None, scope=None, _semantic=None)`**

   Performs an atomic logical OR at the memory location specified by `pointer`.

   Returns the data stored at `pointer` before the atomic operation.

   Parameters
   ----------
   pointer : tensor
       The memory locations to operate on. Must be a block of `triton.PointerDType`.
   val : tensor
       The values with which to perform the atomic operation. Must be a block of
       `pointer.dtype.element_ty`.
   mask : tensor, optional
       Boolean mask to control which elements are updated. If `None`, all elements
       are updated.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are
       `"acquire"`, `"release"`, `"acq_rel"` (stands for "ACQUIRE_RELEASE"),
       and `"relaxed"`. Defaults to `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the
       atomic operation. Acceptable values are `"gpu"` (default), `"cta"`
       (cooperative thread array, thread block), or `"sys"` (stands for "SYSTEM").
       Defaults to `"gpu"`.

   Returns
   -------
   tensor
       The data stored at `pointer` before the atomic operation.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_or(...)` instead of `atomic_or(x, ...)`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr, val):
       old_val = ttgl.atomic_or(ptr, val)
       # old_val contains the value before the OR operation
```

---

### triton.experimental.gluon.language.atomic_xchg

```python
atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_xchg


**`atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _semantic=None)`**

   Perform an atomic exchange at the specified memory location.

   Atomically exchanges the value at `pointer` with `val` and returns the
   original value stored at `pointer` before the operation.

   Parameters
   ----------
   pointer : tensor of pointer type
       The memory location(s) to operate on. Must be a block of pointer type.
   val : tensor
       The value(s) to exchange with the memory location. Must have the same
       element dtype as `pointer`.
   mask : tensor, optional
       Boolean tensor indicating which threads participate in the atomic
       operation. If None, all threads participate.
   sem : str, optional
       Memory semantics for the operation. Acceptable values are "acquire",
       "release", "acq_rel" (ACQUIRE_RELEASE), or "relaxed". Defaults to
       "acq_rel".
   scope : str, optional
       Scope of threads that observe the synchronizing effect. Acceptable
       values are "gpu" (default), "cta" (cooperative thread array), or "sys"
       (SYSTEM).

   Returns
   -------
   tensor
       The value stored at `pointer` before the atomic exchange operation.
       Has the same dtype and shape as `val`.

   Notes
   -----
   This function can be called as a standalone function or as a member function
   on a tensor object. When used as a member function, call as
   `x.atomic_xchg(...)` instead of `atomic_xchg(x, ...)`.

   The operation is atomic across all participating threads. Only threads where
   `mask` is True (or all threads if `mask` is None) will perform the
   exchange.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr, val):
       # Atomic exchange: swap memory value with val, return old value
       old_val = ttgl.atomic_xchg(ptr, val)
       # old_val now contains the value that was in memory before the swap
       # memory now contains val

   @gluon.jit
   def kernel_with_mask(ptr, val, mask):
       # Only threads where mask is True will perform the exchange
       old_val = ttgl.atomic_xchg(ptr, val, mask=mask)

   @gluon.jit
   def kernel_member_fn(tensor_ptr, val):
       # Can also be called as a member function
       old_val = tensor_ptr.atomic_xchg(val)
```

---

### triton.experimental.gluon.language.atomic_xor

```python
atomic_xor(pointer, val, mask=None, sem=None, scope=None, _semantic=None)
```

## atomic_xor


**`atomic_xor(pointer, val, mask=None, sem=None, scope=None, _semantic=None)`**

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
       Boolean tensor specifying which elements to operate on. If provided, only
       elements where `mask` is true will perform the atomic operation.
   sem : str, optional
       Specifies the memory semantics for the operation. Acceptable values are
       `"acquire"`, `"release"`, `"acq_rel"` (stands for "ACQUIRE_RELEASE"),
       and `"relaxed"`. Defaults to `"acq_rel"`.
   scope : str, optional
       Defines the scope of threads that observe the synchronizing effect of the
       atomic operation. Acceptable values are `"gpu"` (default), `"cta"`
       (cooperative thread array, thread block), or `"sys"` (stands for "SYSTEM").
   _semantic
       Internal parameter for semantic operations. Do not set manually.

   Returns
   -------
   tensor
       The data stored at `pointer` before the atomic operation was performed.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.atomic_xor(...)` instead of `atomic_xor(x, ...)`.

   The atomic XOR operation computes `*pointer = *pointer ^ val` atomically.
   This is useful for implementing lock-free data structures and synchronization
   primitives.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, val):
       # Atomic XOR on a single element
       old_val = ttgl.atomic_xor(x_ptr, val)

   @gluon.jit
   def kernel_block(x_ptr, val_ptr, mask):
       # Atomic XOR on a block of elements with masking
       vals = ttgl.load(val_ptr)
       old_vals = ttgl.atomic_xor(x_ptr, vals, mask=mask)
```

---

### triton.experimental.gluon.language.bank_conflicts

```python
bank_conflicts(distr_ty, shared_ty, _semantic=None) -> 'int'
```

## bank_conflicts

Count bank conflicts per wavefront for shared memory access instructions.

```python
 bank_conflicts(distr_ty, shared_ty, _semantic=None) -> int

```
Count the bank conflicts per wavefront of each instruction generated when reading/writing the distributed tensor from/to the shared memory descriptor using `ld.shared`/`st.shared` instructions.

A bank conflict of N is defined as the excess number of memory accesses that each wavefront needs to access the shared memory descriptor. When no load/store vectorization is used, this equals the number of excess memory accesses per instruction.

### Parameters
distr_ty : distributed_type
    The distributed tensor type describing the layout and shape of data in registers.
shared_ty : shared_memory_descriptor_type
    The shared memory descriptor type describing the layout and shape of data in shared memory.
_semantic : GluonSemantic, optional
    Internal semantic handler. Do not pass explicitly; this is managed by the `@gluon.jit` decorator.

### Returns
int
    The number of bank conflicts per wavefront for the given tensor and shared memory layout combination.

### Notes
This is a compile-time analysis function that operates on type descriptors, not runtime tensor values. It helps users evaluate the efficiency of different layout combinations for shared memory transfers.

Bank conflicts occur when multiple threads in a wavefront access the same shared memory bank simultaneously. Lower bank conflict counts indicate more efficient memory access patterns.

This function is typically used during kernel development to compare different layout configurations and optimize shared memory access patterns.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language._layouts import BlockedLayout, SwizzledSharedLayout

 @gluon.jit
 def kernel():
     # Define a distributed layout for registers
     distr_layout = BlockedLayout([1, 4], [16, 32], [1, 1], [0, 1])
     distr_ty = ttgl.distributed_type(ttgl.float16, [16, 32], distr_layout)

     # Define a shared memory layout
     shared_layout = SwizzledSharedLayout([1, 4], [16, 32], 4)
     shared_ty = ttgl.shared_memory_descriptor_type(
         ttgl.float16, [16, 32], shared_layout, [16, 32]
     )

     # Count bank conflicts for this layout combination
     conflicts = ttgl.bank_conflicts(distr_ty, shared_ty)
     ttgl.static_print("Bank conflicts:", conflicts)
```

---

### triton.experimental.gluon.language.barrier

```python
barrier(*, cluster: 'bool' = False, _semantic=None)
```

## gluon.language.barrier


**`barrier(*, cluster=False)`**

   Insert a barrier to synchronize threads within a CTA or across a CTA cluster.

   Parameters
   ----------
   cluster : bool, optional
       Whether to synchronize across the CTA cluster. Default is `False`, which
       synchronizes only within the current CTA. Set to `True` for cluster-wide
       synchronization when multiple CTAs are present.

   Returns
   -------
   None

   Notes
   -----
   The barrier behavior depends on the kernel configuration and the `cluster`
   parameter:

   - When `cluster=False` or the kernel has only one CTA (`num_ctas == 1`),
     inserts a CTA-level synchronization barrier (`debug_barrier`).
   - When `cluster=True` and multiple CTAs are present, inserts a cluster-wide
     barrier (`create_cluster_barrier`) that synchronizes across all CTAs in
     the cluster.

   All threads must reach the barrier before any can proceed. Barriers are
   essential for coordinating shared memory accesses and ensuring correct
   execution order in parallel kernels.

   Examples
   --------
   Synchronize threads within a CTA:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(...):
       # ... compute ...
       ttgl.barrier()  # CTA-level synchronization
       # ... continue ...

Synchronize across a CTA cluster:

.. code-block:: python

   @gluon.jit
   def kernel(...):
       # ... compute ...
       ttgl.barrier(cluster=True)  # Cluster-wide synchronization
       # ... continue ...
```

---

### triton.experimental.gluon.language.base_type

```python
base_type()
```

**`triton.experimental.gluon.language.base_type`**

   Abstract base class for type representations in the Gluon type system.

   This class defines the interface that all type objects must implement to
   support IR lowering, type mangling, and value reconstruction. Subclasses
   represent concrete types such as :py`dtype`, :py`pointer_type`,
   :py`block_type`, and :py`tuple_type`.

   .. rubric:: Methods

   .. py:method:: __eq__(other) -> bool

      Compare two types for equality.

      Parameters
      ----------
      other : base_type
         The type to compare against.

      Returns
      -------
      bool
         True if the types are equal, False otherwise.

      Notes
      -----
      Must be implemented by subclasses.

   .. py:method:: __ne__(other) -> bool

      Compare two types for inequality.

      Parameters
      ----------
      other : base_type
         The type to compare against.

      Returns
      -------
      bool
         True if the types are not equal, False otherwise.

      Notes
      -----
      Default implementation negates :py`__eq__()`.

   .. py:method:: _unflatten_ir(handles, cursor) -> Tuple[base_value, int]

      Reconstruct a frontend value from IR handles.

      Parameters
      ----------
      handles : List[ir.value]
         List of IR value handles to reconstruct from.
      cursor : int
         Index of the first handle relevant to this value.

      Returns
      -------
      Tuple[base_value, int]
         A tuple containing the reconstructed frontend value and the updated
         cursor position after consuming handles.

      Notes
      -----
      Must be implemented by subclasses. This method is used during IR
      deserialization to rebuild frontend values from low-level IR handles.

   .. py:method:: mangle() -> str

      Return a mangled string representation of the type.

      Returns
      -------
      str
         The mangled type name used for caching and identification.

      Notes
      -----
      Must be implemented by subclasses. Type mangling is used for kernel
      cache key generation and type identification.

   .. py:method:: _flatten_ir_types(builder, out) -> None

      Flatten this type into IR type handles.

      Parameters
      ----------
      builder : ir.builder
         The IR builder context.
      out : List[ir.type]
         Output list to append IR type handles to.

      Notes
      -----
      Must be implemented by subclasses. This method converts frontend type
      representations into low-level IR types during kernel compilation.

   .. rubric:: Examples

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # base_type is not instantiated directly - use concrete subclasses
   dtype = ttgl.dtype('fp32')
   ptr_type = ttgl.pointer_type(dtype)

   # Type comparison
   assert dtype == ttgl.dtype('fp32')
   assert dtype != ttgl.dtype('fp16')

   # Type mangling for cache keys
   mangled = dtype.mangle()  # Returns 'fp32'
```

---

### triton.experimental.gluon.language.base_value

```python
base_value()
```

### base_value

**`base_value()`**

   Base class for values that exist in the Triton IR (i.e., not constexprs).

   This abstract base class represents runtime values that are lowered to MLIR
   operations. All tensor-like objects in Triton inherit from this class.

   Attributes
   ----------
   type : base_type
       The type descriptor of the value in the Triton type system.

   Methods
   -------
   _set_name(builder, name)
       Assigns a debug name to the IR value.
       
       Parameters
       ----------
       builder : ir.builder
           The MLIR builder instance.
       name : str
           The debug name to assign.
   
   _flatten_ir(handles)
       Flattens the frontend value into a sequence of MLIR handles.
       
       Parameters
       ----------
       handles : list of ir.value
           Output list to append MLIR handles to.

   Notes
   -----
   This is an abstract base class and should not be instantiated directly.
   Common subclasses include:

   - :py`tensor`: N-dimensional arrays of values or pointers
   - :py`tuple`: Heterogeneous collections of values  
   - :py`tensor_descriptor_base`: Tensor descriptors for global memory
   - :py`constexpr`: Compile-time constant values

   Subclasses must implement `_set_name` and `_flatten_ir` to support
   proper IR lowering.

   Examples
   --------
   Users typically interact with subclasses rather than `base_value` directly:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offset = pid * BLOCK
       x = ttgl.load(x_ptr + offset)  # x is a tensor (base_value subclass)
       ttgl.store(y_ptr + offset, x)

See Also
--------
tensor, constexpr, tuple, tensor_descriptor_base
```

---

### triton.experimental.gluon.language.bfloat16

.. py:data:: triton.experimental.gluon.language.bfloat16

    16-bit brain floating point format.

    Notes
    -----
    The bfloat16 data type follows the IEEE 754 standard with 1 sign bit, 8
    exponent bits, and 7 mantissa bits. It offers the same dynamic range as
    float32 but with reduced precision, making it suitable for deep learning
    training and inference.

    This type is natively supported on NVIDIA Ampere (SM_80+), Hopper, and
    Blackwell GPUs, as well as AMD CDNA3 and RDNA3 architectures.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> @gluon.jit
    ... def cast_kernel(x_ptr, y_ptr, size, BLOCK_SIZE: ttgl.constexpr):
    ...     idx = ttgl.program_id(0) * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
    ...     mask = idx < size
    ...     fp32_val = ttgl.load(x_ptr + idx, mask=mask)
    ...     bf16_val = fp32_val.to(ttgl.bfloat16)
    ...     ttgl.store(y_ptr + idx, bf16_val, mask=mask)

---

### triton.experimental.gluon.language.block_type

```python
block_type(element_ty: 'dtype', shape: 'List')
```

## block_type

**`block_type(element_ty, shape)`**

   Represents a typed memory block with a specific element type and shape.

   `block_type` is used to describe the type of tensor blocks in Gluon kernels,
   specifying both the scalar element type and the multidimensional shape of the
   block. This is fundamental for defining tensor layouts, shared memory allocations,
   and TMA descriptor block shapes.

   Parameters
   ----------
   element_ty : dtype
       The scalar data type of elements in the block (e.g., `tl.float16`, `tl.int32`).
   shape : list or tuple of int
       The multidimensional shape of the block. Must be non-empty (0D blocks are forbidden).

   Attributes
   ----------
   element_ty : dtype
       The scalar element type of the block.
   shape : tuple of int
       The shape dimensions of the block.
   numel : int
       Total number of elements in the block (product of shape dimensions).
   scalar : dtype
       Alias for `element_ty`.
   nbytes : int
       Total size in bytes of the block (numel × element bitwidth / 8).

   Methods
   -------
   to_ir(builder)
       Converts the block type to an IR block type for the given builder.
   get_block_shapes()
       Returns the shape tuple of the block.
   with_element_ty(scalar_ty)
       Returns a new `block_type` with the same shape but different element type.
   is_block()
       Returns `True` (identifies this as a block type).
   mangle()
       Returns a mangled string representation for type encoding.

   Notes
   -----
   - `block_type` differs from `tensor` in that its shape is stored as a tuple
     of Python integers, not `constexpr` values.
   - 0D block types are forbidden and will raise `TypeError`.
   - The `numel` attribute is validated against `TRITON_MAX_TENSOR_NUMEL`.
   - Block types are used extensively in Gluon for explicit layout control, TMA
     descriptors, and warpgroup MMA operations.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # Create a 2D block type with 16x32 float16 elements
   block_ty = ttgl.block_type(ttgl.float16, [16, 32])
   print(block_ty)  # <(16, 32), fp16>

   # Access block properties
   print(block_ty.shape)      # (16, 32)
   print(block_ty.numel)      # 512
   print(block_ty.nbytes)     # 1024 (512 × 2 bytes)

   # Create a new block type with different element type
   int_block = block_ty.with_element_ty(ttgl.int32)
   print(int_block)           # <(16, 32), int32>
   print(int_block.nbytes)    # 2048 (512 × 4 bytes)

   # Use in tensor descriptor creation
   @gluon.jit
   def kernel(desc, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr):
       block_shape = [BLOCK_M, BLOCK_N]
       # Block types define the load/store granularity for TMA operations
       ...
```

---

### triton.experimental.gluon.language.broadcast

```python
broadcast(input, other, _semantic=None)
```

## broadcast

Broadcast two tensors to a common compatible shape.

### Parameters
input : Block
    The first input tensor to broadcast.
other : Block
    The second input tensor to broadcast.

### Returns
Block
    Tensor with shape compatible with both inputs after broadcasting.

### Notes
This function follows standard broadcasting rules to determine the output
shape. Both inputs must be broadcastable to a common shape. The `_semantic`
parameter is internal and should not be provided by users.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: ttgl.constexpr):
     pid = ttgl.program_id(0)
     x = ttgl.load(x_ptr + pid)
     y = ttgl.full([BLOCK_SIZE], 1.0, ttgl.float32)
     # Broadcast scalar x to match y's shape
     x_broadcasted = ttgl.broadcast(x, y)
     result = x_broadcasted + y
     ttgl.store(out_ptr + pid * BLOCK_SIZE, result)
```

---

### triton.experimental.gluon.language.cast

```python
cast(input, dtype: 'dtype', fp_downcast_rounding: 'Optional[str]' = None, bitcast: 'bool' = False, _semantic=None)
```

## cast

**`cast(input, dtype, fp_downcast_rounding=None, bitcast=False)`**

Casts a tensor to the given dtype.

### Parameters
input : tensor
    The input tensor to cast.
dtype : tl.dtype
    The target data type.
fp_downcast_rounding : str, optional
    The rounding mode for downcasting floating-point values. This parameter is
    only used when input is a floating-point tensor and dtype is a floating-point
    type with a smaller bitwidth. Supported values are `"rtne"` (round to
    nearest, ties to even) and `"rtz"` (round towards zero).
bitcast : bool, optional
    If True, the tensor is bitcasted to the given dtype, instead of being
    numerically casted. Default is False.

### Returns
tensor
    The casted tensor with the target dtype.

### Notes
This function can also be called as a member function on :py`tensor`,
as `x.cast(...)` instead of `cast(x, ...)`.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     x = ttgl.load(x_ptr)
     # Numeric cast to float16
     y = ttgl.cast(x, ttgl.float16)
     ttgl.store(y_ptr, y)

 @gluon.jit
 def kernel2(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     x = ttgl.load(x_ptr)
     # Bitcast to int32 (reinterpret bits without numeric conversion)
     y = ttgl.cast(x, ttgl.int32, bitcast=True)
     ttgl.store(y_ptr, y)

 @gluon.jit
 def kernel3(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     x = ttgl.load(x_ptr)
     # Downcast float32 to float16 with round-to-zero
     y = ttgl.cast(x, ttgl.float16, fp_downcast_rounding="rtz")
     ttgl.store(y_ptr, y)
```

---

### triton.experimental.gluon.language.cdiv

```python
cdiv(x, div)
```

**`ttgl.cdiv(x, div)`**

   Computes the ceiling division of `x` by `div`.

   Parameters
   ----------
   x : ttgl.Block
       The input number or tensor.
   div : ttgl.Block
       The divisor.

   Returns
   -------
   ttgl.Block
       The result of ceiling division.

   Notes
   -----
   This function computes `(x + div - 1) // div` element-wise.

   It can also be called as a member function on `ttgl.Block`,
   as `x.cdiv(div)` instead of `ttgl.cdiv(x, div)`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, div_ptr, out_ptr, BLOCK_SIZE: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offset = pid * BLOCK_SIZE
       x = ttgl.load(x_ptr + offset)
       div = ttgl.load(div_ptr + offset)
       # Functional style
       out = ttgl.cdiv(x, div)
       # Member style
       out = x.cdiv(div)
       ttgl.store(out_ptr + offset, out)
```

---

### triton.experimental.gluon.language.ceil

```python
ceil(x, _semantic=None)
```

## ceil


**`ceil(x, _semantic=None)`**

   Computes the element-wise ceiling of the input tensor.

   For each element `x_i`, returns the smallest integer greater than or equal to `x_i` as a floating-point value.

   Parameters
   ----------
   x : Block
      The input tensor. Must have floating-point dtype (`fp32` or `fp64`).
   _semantic : GluonSemantic, optional
      Internal semantic argument. Do not set manually.

   Returns
   -------
   tensor
      A tensor of the same shape and dtype as `x`, containing the ceiling values.

   Notes
   -----
   This function is only supported for `fp32` and `fp64` dtypes. Integer types are not supported.

   The ceiling operation rounds each element upward to the nearest integer value, returned as a float. For example, `ceil(3.2) = 4.0` and `ceil(-3.2) = -3.0`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def ceil_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = ttgl.program_id(0)
       offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offs)
       y = ttgl.ceil(x)
       ttgl.store(y_ptr + offs, y)
```

---

### triton.experimental.gluon.language.constexpr

```python
constexpr(value)
```

**`constexpr(value)`**

   Store a value that is known at compile-time.

   :py`constexpr` wraps Python values that can be resolved during kernel
   compilation. These values are used for kernel parameters marked as
   `constexpr`, loop bounds, tensor shapes, and other compile-time
   constants.

   Parameters
   ----------
   value : int, float, bool, or str
       The compile-time constant value to wrap. Nested :py`constexpr`
       instances are automatically unwrapped.

   Attributes
   ----------
   value
       The underlying Python value.
   type
       The :py`constexpr_type` describing this constant's type.

   Methods
   -------
   __index__()
       Returns the underlying value, enabling use in indexing contexts.
   __bool__()
       Returns the boolean value of the underlying constant.
   __iter__()
       Enables iteration if the underlying value is iterable.
   __call__(*args, **kwds)
       Calls the underlying value if it is callable.
   __getitem__(*args)
       Supports indexing into the underlying value.

   Notes
   -----
   :py`constexpr` supports all standard Python arithmetic and comparison
   operators (+, -, *, /, //, %, **, <<, >>, &, |, ^, <, <=, >, >=, ==, !=).
   Operations between :py`constexpr` and other values return a new
   :py`constexpr` instance.

   In Gluon kernels, :py`constexpr` values are typically passed via
   kernel parameters annotated with `: tl.constexpr` in the function
   signature. They enable compile-time specialization and loop unrolling.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, BLOCK_SIZE: ttgl.constexpr):
       # BLOCK_SIZE is a constexpr, usable for compile-time shapes
       offsets = ttgl.arange(0, BLOCK_SIZE)
       mask = offsets < BLOCK_SIZE
       value = ttgl.load(x_ptr + offsets, mask=mask)
       ttgl.store(x_ptr + offsets, value, mask=mask)

   # Launch with compile-time constant
   BLOCK_SIZE = 128
   kernel[(1,)](x_ptr, BLOCK_SIZE)

.. code-block:: python

   # Manual constexpr construction
   from triton.experimental.gluon.language import constexpr

   c = constexpr(42)
   print(c.value)  # 42
   print(c + 8)    # constexpr[50]
   print(c * 2)    # constexpr[84]

.. code-block:: python

   # constexpr in compile-time control flow
   @gluon.jit
   def kernel_with_branch(x_ptr, USE_FAST_PATH: ttgl.constexpr):
       if USE_FAST_PATH:
           # Compiled only when USE_FAST_PATH is True
           fast_compute(x_ptr)
       else:
           # Compiled only when USE_FAST_PATH is False
           slow_compute(x_ptr)
```

---

### triton.experimental.gluon.language.convert_layout

```python
convert_layout(value, layout, assert_trivial=False, _semantic=None)
```

## convert_layout

Convert a tensor to a different distributed layout.

```python
 convert_layout(value, layout, assert_trivial=False)

```
### Parameters
value : tensor
    The input tensor to convert.
layout : DistributedLayout
    The target distributed layout for the output tensor.
assert_trivial : bool, optional
    If True, asserts that the conversion is trivial (no data movement required).
    Default is False.

### Returns
tensor
    The input tensor converted to the specified layout.

### Notes
This operation may insert data movement instructions if the source and target
layouts differ. Setting `assert_trivial=True` will cause compilation to fail
if the conversion requires actual data movement rather than just a type change.

The `_semantic` parameter is internal and should not be provided by users;
it is automatically injected when using `@gluon.jit` decorated kernels.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language import BlockedLayout

 @gluon.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     pid = ttgl.program_id(0)
     offs = ttgl.arange(0, BLOCK_SIZE)
     mask = offs < BLOCK_SIZE
     x = ttgl.load(x_ptr + pid * BLOCK_SIZE + offs, mask=mask)
     
     # Convert from default layout to blocked layout
     blocked_layout = BlockedLayout([1], [BLOCK_SIZE], [32], [0])
     x_blocked = ttgl.convert_layout(x, blocked_layout)
     
     # Use converted tensor in computation
     y = x_blocked * 2.0
     ttgl.store(y_ptr + pid * BLOCK_SIZE + offs, y, mask=mask)
```

---

### triton.experimental.gluon.language.cos

```python
cos(x, _semantic=None)
```

## cos


**`cos(x, _semantic=None)`**

   Computes the element-wise cosine of `x`.

   Parameters
   ----------
   x : tensor
       The input values. Must be floating point (fp32 or fp64).
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually.

   Returns
   -------
   tensor
       A tensor containing the cosine of each element in `x`.

   Notes
   -----
   This function only supports fp32 and fp64 data types. Attempting to use
   other dtypes will raise an error.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offs)
       y = ttgl.cos(x)
       ttgl.store(y_ptr + offs, y)
```

---

### triton.experimental.gluon.language.device_assert

```python
device_assert(cond, msg='', mask=None, _semantic=None)
```

## device_assert


**`device_assert(cond, msg='', mask=None, _semantic=None)`**

   Assert a condition at runtime from the GPU device.

   This function inserts a device-side assertion that checks `cond` during kernel
   execution. If the condition is false, the message `msg` is printed. The assertion
   only has effect when the environment variable `TRITON_DEBUG` is set to a
   non-zero value.

   Parameters
   ----------
   cond : tensor
       Boolean tensor representing the condition to assert. Must evaluate to true
       for all active lanes.
   msg : str, optional
       Message to print if the assertion fails. Must be a string literal.
       Default is empty string.
   mask : tensor, optional
       Boolean tensor mask. If provided, assertion is only checked for lanes
       where mask is true.
   _semantic : GluonSemantic, optional
       Internal semantic object. Do not set manually.

   Returns
   -------
   None

   Notes
   -----
   Using the Python `assert` statement within a Gluon kernel is equivalent to
   calling this function, but requires the message argument to be provided as a
   string literal, e.g., `assert pid == 0, "pid != 0"`.

   Device assertions are only evaluated when `TRITON_DEBUG` environment
   variable is set to a non-zero value. In production builds (TRITON_DEBUG=0),
   assertions are elided.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(pid):
       # Assert that program ID is zero
       ttgl.device_assert(pid == 0, "pid must be 0")

       # Equivalent Python assert syntax
       assert pid == 0, "pid must be 0"

       # Conditional assertion with mask
       mask = pid < 8
       ttgl.device_assert(pid >= 0, "pid negative", mask=mask)
```

---

### triton.experimental.gluon.language.device_print

```python
device_print(prefix, *args, hex=False, _semantic=None)
```

## device_print

Print values at runtime from the GPU device.

**`device_print(prefix, *args, hex=False, _semantic=None)`**

   Print the values at runtime from the device. String formatting does not work
   for runtime values, so provide the values you want to print as arguments. The
   first value must be a string, all following values must be scalars or tensors.

   Calling the Python builtin `print` is equivalent to calling this function,
   and the argument requirements match this function (not the normal requirements
   for `print`).

   Parameters
   ----------
   prefix : str
       A prefix to print before the values. This is required to be a string
       literal and must contain only ASCII printable characters.
   *args : tensor or scalar
       The values to print. They can be any tensor or scalar.
   hex : bool, optional
       If True, print all values as hex instead of decimal (default: False).
   _semantic : GluonSemantic, optional
       Internal parameter for semantic operations. Do not set manually.

   Returns
   -------
   None

   Notes
   -----
   On CUDA, printfs are streamed through a buffer of limited size (on one host,
   the default was measured as 6912 KiB, but this may not be consistent across
   GPUs and CUDA versions). If you notice some printfs are being dropped, you
   can increase the buffer size by calling:

```python
    triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)

CUDA may raise an error if you try to change this value after running a
kernel that uses printfs. The value set here may only affect the current
device (so if you have multiple GPUs, you need to call it multiple times).

Examples
--------
.. code-block:: python

    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl

    @gluon.jit
    def kernel(pid):
        ttgl.device_print("pid", pid)
        print("pid", pid)  # equivalent to device_print
```

---

### triton.experimental.gluon.language.distributed_type

```python
distributed_type(element_ty: 'dtype', shape: 'List[int]', layout)
```

## distributed_type

**`distributed_type(element_ty, shape, layout)`**

   A distributed tensor type with explicit layout information for Gluon kernels.

   This type extends `block_type` to include layout metadata, enabling
   explicit control over how tensor data is distributed across GPU threads,
   warps, and memory hierarchies. Used internally by Gluon to represent typed
   tensors with known distribution patterns.

   Parameters
   ----------
   element_ty : dtype
       The scalar element data type (e.g., `ttgl.float16`, `ttgl.int32`).
   shape : List[int]
       The dimensions of the tensor block.
   layout : DistributedLayout
       The distribution layout describing how elements map to threads/warps.
       Must be an instance of `DistributedLayout`.

   Attributes
   ----------
   element_ty : dtype
       The scalar element type.
   shape : List[int]
       The tensor shape.
   layout : DistributedLayout
       The distribution layout.
   name : str
       String representation of the type.

   Methods
   -------
   to_ir(builder)
       Convert to LLVM IR type.
   mangle()
       Generate a mangled name string for type identification.
   with_element_ty(scalar_ty)
       Create a new distributed_type with a different element type.

   Notes
   -----
   - The layout must be a `DistributedLayout` instance.
   - For non-AutoLayout and non-CoalescedLayout layouts, the shape rank must
     match the layout rank.
   - This type is typically constructed internally by Gluon operations rather
     than directly by users.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>> from triton.experimental.gluon.language._layouts import BlockedLayout
   >>>
   >>> # Create a distributed type for a 128x64 float16 tensor
   >>> layout = BlockedLayout([1, 1], [128, 64], [4, 8], [1, 0])
   >>> dtype = ttgl.distributed_type(ttgl.float16, [128, 64], layout)
   >>> print(dtype.name)
   <[128, 64], float16, BlockedLayout(...)>

   See Also
   --------
   block_type, DistributedLayout, AutoLayout, CoalescedLayout

---

### triton.experimental.gluon.language.div_rn

```python
div_rn(x, y, _semantic=None)
```

## div_rn


**`div_rn(x, y, _semantic=None)`**

   Computes the element-wise precise division of `x` and `y`, rounding to nearest per the IEEE standard.

   Parameters
   ----------
   x : Block
      The dividend input values. Must be `fp32` dtype.
   y : Block
      The divisor input values. Must be `fp32` dtype.
   _semantic : GluonSemantic, optional
      Internal semantic argument. Do not set manually.

   Returns
   -------
   Block
      A tensor containing the element-wise division result `x / y` with `fp32` dtype.

   Notes
   -----
   This operation performs precise floating-point division with IEEE 754 rounding to nearest. Only `fp32` dtype is supported.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       pid = ttgl.program_id(0)
       offs = ttgl.arange(0, BLOCK_SIZE)
       mask = offs < BLOCK_SIZE
       x = ttgl.load(x_ptr + pid * BLOCK_SIZE + offs, mask=mask)
       y = ttgl.load(y_ptr + pid * BLOCK_SIZE + offs, mask=mask)
       out = ttgl.div_rn(x, y)
       ttgl.store(out_ptr + pid * BLOCK_SIZE + offs, out, mask=mask)
```

---

### triton.experimental.gluon.language.dot_fma

```python
dot_fma(a, b, acc, _semantic=None)
```

## dot_fma


**`dot_fma(a, b, acc, _semantic=None)`**

   Perform a fused multiply-add (FMA) matrix multiplication: $acc + a @ b$.

   This is a low-level operation that requires explicit layout control. Input tensors
   must have compatible `DotOperandLayout` layouts, and the accumulator must
   have a `BlockedLayout`. All layouts must share the same parent MMA layout.

   Parameters
   ----------
   a : tensor
       First operand tensor with `DotOperandLayout` (operand index 0).
       Shape must match `acc` shape plus a trailing K dimension.
   b : tensor
       Second operand tensor with `DotOperandLayout` (operand index 1).
       Shape must match `acc` shape plus a trailing K dimension.
   acc : tensor
       Accumulator tensor with `BlockedLayout`. Determines the output
       shape, dtype, and layout. Must be 2D (M x N) or 3D (batch x M x N).
   _semantic : GluonSemantic, optional
       Internal semantic handler. Automatically provided by the `@gluon.jit`
       decorator. Do not set manually.

   Returns
   -------
   tensor
       Result tensor with the same shape, dtype, and layout as `acc`.

   Raises
   ------
   AssertionError
       If input tensors do not meet layout requirements or shape constraints.

   Notes
   -----
   - This is a Gluon-specific primitive that gives explicit control over MMA
     instruction emission. For most use cases, prefer higher-level `ttgl.dot`.
   - Tensor `a` must have `operand_index == 0` and `b` must have
     `operand_index == 1` in their respective `DotOperandLayout`.
   - All three tensors must share the same parent MMA layout.
   - Large dot operations (product of M x N x K > 2^19) may have slow compile
     times and will emit a warning.
   - Supports both 2D matrix multiplication and 3D batched matrix multiplication.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon._layouts import BlockedLayout, DotOperandLayout

   @gluon.jit
   def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
       pid = ttgl.program_id(0)
       offs_m = pid * BLOCK_M
       offs_n = ttgl.arange(0, BLOCK_N)
       offs_k = ttgl.arange(0, BLOCK_K)

       # Define MMA layout for accumulator
       mma_layout = BlockedLayout([BLOCK_M, BLOCK_N], [1, 1])

       # Create dot operand layouts
       a_layout = DotOperandLayout(mma_layout, operand_index=0)
       b_layout = DotOperandLayout(mma_layout, operand_index=1)

       # Load tiles with appropriate layouts
       a = ttgl.load(a_ptr + offs_m * K + offs_k, layout=a_layout)
       b = ttgl.load(b_ptr + offs_k * N + offs_n, layout=b_layout)
       acc = ttgl.full([BLOCK_M, BLOCK_N], 0.0, dtype=ttgl.float16, layout=mma_layout)

       # Perform FMA dot operation
       c = ttgl.dot_fma(a, b, acc)

       # Store result
       ttgl.store(c_ptr + offs_m * N + offs_n, c)
```

---

### triton.experimental.gluon.language.dtype

```python
dtype(name)
```

**`dtype(name)`**

   Represents a data type in Triton IR.

   The `dtype` class encapsulates information about primitive data types
   including signed/unsigned integers, floating-point formats (including FP8
   variants), and void types. It provides methods to query type properties
   and convert to IR types.

   Parameters
   ----------
   name : str
       The name of the data type. Must be one of the supported type strings:
       `'int1'`, `'int8'`, `'int16'`, `'int32'`, `'int64'`,
       `'uint8'`, `'uint16'`, `'uint32'`, `'uint64'`,
       `'fp8e4b15'`, `'fp8e4nv'`, `'fp8e4b8'`, `'fp8e5'`,
       `'fp8e5b16'`, `'fp16'`, `'bf16'`, `'fp32'`, `'fp64'`,
       or `'void'`.

   Attributes
   ----------
   name : str
       The type name string.
   primitive_bitwidth : int
       The bit width of the primitive type.
   itemsize : int
       The size in bytes (primitive_bitwidth // 8).
   int_signedness : SIGNEDNESS
       For integer types, indicates signed or unsigned.
   int_bitwidth : int
       For integer types, the bit width.
   fp_mantissa_width : int
       For floating-point types, the mantissa width.
   exponent_bias : int
       For floating-point types, the exponent bias.

   Methods
   -------
   is_fp8()
       Returns True if this is an FP8 type.
   is_fp16(), is_bf16(), is_fp32(), is_fp64()
       Check for specific floating-point types.
   is_int8(), is_int16(), is_int32(), is_int64()
       Check for specific signed integer types.
   is_uint8(), is_uint16(), is_uint32(), is_uint64()
       Check for specific unsigned integer types.
   is_floating()
       Returns True if this is any floating-point type.
   is_standard_floating()
       Returns True if this is fp16, bf16, fp32, or fp64.
   is_int_signed(), is_int_unsigned(), is_int()
       Check integer type categories.
   is_bool()
       Returns True if this is int1 (boolean).
   kind()
       Returns KIND enum (BOOLEAN, INTEGRAL, or FLOATING).
   get_int_max_value(), get_int_min_value()
       Return max/min representable values for integer types.
   to_ir(builder)
       Convert to Triton IR type using the given builder.
   mangle()
       Return mangled type name for ABI compatibility.
   codegen_name()
       Return code generation friendly type name.

   Notes
   -----
   Predefined dtype instances are available as module-level constants:
   `int1`, `int8`, `int16`, `int32`, `int64`, `uint8`,
   `uint16`, `uint32`, `uint64`, `float8e5`, `float8e5b16`,
   `float8e4nv`, `float8e4b8`, `float8e4b15`, `float16`,
   `bfloat16`, `float32`, `float64`, and `void`.

   FP8 types may not be supported on all architectures. The `to_ir()`
   method will raise an error if the FP8 type is not supported by the
   target architecture.

   Examples
   --------
   >>> import triton.experimental.gluon.language as ttgl
   >>> dtype = ttgl.dtype('fp16')
   >>> dtype.name
   'fp16'
   >>> dtype.primitive_bitwidth
   16
   >>> dtype.itemsize
   2
   >>> dtype.is_floating()
   True
   >>> dtype.is_fp16()
   True
   >>> dtype.kind()
   <KIND.FLOATING: 2>

   Using predefined constants:

```python
    import triton.experimental.gluon.language as ttgl

    # Use predefined dtype constants
    dtype = ttgl.float16
    assert dtype.name == 'fp16'

    # Check type properties
    if dtype.is_floating():
        print(f"Mantissa width: {dtype.fp_mantissa_width}")

    # Convert to IR type in a kernel
    @gluon.jit
    def kernel(x_ptr, n):
        ir_type = ttgl.float32.to_ir(gluon.language.core.builder())
        # ... use ir_type in IR operations
```

---

### triton.experimental.gluon.language.erf

```python
erf(x, _semantic=None)
```

## erf


**`erf(x, _semantic=None)`**

   Computes the element-wise Gaussian error function of `x`.

   Parameters
   ----------
   x : tensor
       Input tensor. Must have floating point dtype (`fp32` or `fp64`).

   Returns
   -------
   tensor
       Output tensor of same shape and dtype as `x`, containing the error function values.

   Notes
   -----
   The error function is defined as:

   .. math::

      \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt

   This operation is only supported for `fp32` and `fp64` dtypes.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offs)
       y = ttgl.erf(x)
       ttgl.store(y_ptr + offs, y)
```

---

### triton.experimental.gluon.language.exp

```python
exp(x, _semantic=None)
```

## exp

Computes the element-wise natural exponential of the input tensor.

### Parameters
x : tensor
    The input tensor. Must have floating point dtype (`fp32` or `fp64`).
_semantic : GluonSemantic, optional
    Internal semantic argument. Do not set manually.

### Returns
tensor
    A tensor containing $e^x$ for each element in `x`. The output has
    the same shape and dtype as the input.

### Notes
This function only supports `fp32` and `fp64` data types. Attempts to use
other dtypes will raise an error at compile time.

The exponential is computed as $e^x$ where $e$ is Euler's number
(approximately 2.71828).

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def exp_kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     pid = ttgl.program_id(0)
     offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
     x = ttgl.load(x_ptr + offs)
     y = ttgl.exp(x)
     ttgl.store(y_ptr + offs, y)
```

---

### triton.experimental.gluon.language.exp2

```python
exp2(x, _semantic=None)
```

## exp2

Compute the element-wise exponential (base 2) of a tensor.

**`triton.experimental.gluon.language.exp2(x, _semantic=None)`**

   Computes $2^x$ element-wise for the input tensor.

   Parameters
   ----------
   x : tensor
       The input tensor. Must have floating-point dtype (`fp32` or `fp64`).
   _semantic : GluonSemantic, optional
       Internal parameter. Do not set manually.

   Returns
   -------
   tensor
       A tensor containing $2^x$ computed element-wise. Has the same shape
       and dtype as `x`.

   Notes
   -----
   This function is only supported for floating-point dtypes (`fp32`, `fp64`).
   It is decorated as a builtin and can be called as a tensor member function
   (e.g., `x.exp2()`).

   The operation is performed on the GPU and is optimized for Gluon kernels
   decorated with `@gluon.jit`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = ttgl.program_id(0)
       offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offs)
       y = ttgl.exp2(x)
       ttgl.store(y_ptr + offs, y)
```

---

### triton.experimental.gluon.language.expand_dims

```python
expand_dims(input, axis, _semantic=None)
```

## expand_dims


**`expand_dims(input, axis, _semantic=None)`**

Expand the shape of a tensor by inserting new length-1 dimensions.

Axis indices are with respect to the resulting tensor, so `result.shape[axis]` will be 1 for each specified axis.

### Parameters
input : tl.tensor
    The input tensor to expand.
axis : int or Sequence[int]
    The axis or axes along which to insert new dimensions. Indices refer to the output tensor shape.
_semantic : GluonSemantic, optional
    Internal semantic argument. Do not provide explicitly.

### Returns
tl.tensor
    A tensor with the same data as `input` but with additional length-1 dimensions inserted.

### Notes
This function can also be called as a member function on :py`tensor`, as `x.expand_dims(...)` instead of `expand_dims(x, ...)`.

Duplicate axes will raise a `ValueError`. Negative axis indices are supported and normalized relative to the output tensor rank.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, y_ptr, BLOCK: ttgl.constexpr):
     x = ttgl.load(x_ptr)
     # x shape: [BLOCK]
     y = ttgl.expand_dims(x, axis=0)
     # y shape: [1, BLOCK]
     ttgl.store(y_ptr, y)

 @gluon.jit
 def kernel_multi_axis(x_ptr, z_ptr, BLOCK: ttgl.constexpr):
     x = ttgl.load(x_ptr)
     # x shape: [BLOCK]
     z = ttgl.expand_dims(x, axis=[0, 2])
     # z shape: [1, BLOCK, 1]
     ttgl.store(z_ptr, z)
```

---

### triton.experimental.gluon.language.fdiv

```python
fdiv(x, y, ieee_rounding=False, _semantic=None)
```

## fdiv


.. autofunction:: fdiv

Computes the element-wise fast division of `x` and `y`.

### Parameters
x : Block
    The dividend input values.
y : Block
    The divisor input values.
ieee_rounding : bool, optional
    If True, use IEEE rounding mode. Default is False (fast division).

### Returns
Block
    A tensor containing the element-wise division result `x / y`.

### Notes
Fast division may have reduced precision compared to standard IEEE 754
division. Set `ieee_rounding=True` for IEEE-compliant behavior at
the cost of performance.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: ttgl.constexpr):
     pid = ttgl.program_id(0)
     offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
     x = ttgl.load(x_ptr + offs)
     y = ttgl.load(y_ptr + offs)
     # Fast division (default)
     result = ttgl.fdiv(x, y)
     # IEEE-compliant division
     result_ieee = ttgl.fdiv(x, y, ieee_rounding=True)
     ttgl.store(out_ptr + offs, result)
```

---

### triton.experimental.gluon.language.float16

.. py:data:: triton.experimental.gluon.language.float16

   16-bit floating-point data type instance.

   Notes
   -----
   Represents the IEEE 754 binary16 floating-point format (half precision).
   This instance is used to specify element types for tensors and memory
   operations in Gluon kernels. It is optimized for high-throughput compute
   on NVIDIA Hopper/Blackwell and AMD CDNA/RDNA architectures supporting FP16.

   Examples
   --------
   Import and inspect the dtype instance:

```python
   import triton.experimental.gluon.language as ttgl

   dtype = ttgl.float16
   print(dtype)  # <dtype: float16>
```

---

### triton.experimental.gluon.language.float32

.. py:data:: float32

    32-bit floating point data type.

    Represents IEEE 754 single-precision floating-point numbers. This instance is
    used to annotate element types in `@gluon.jit()` kernel signatures and
    tensor operations within the Gluon API.

    Notes
    -----
    Equivalent to IEEE 754 binary32 (`float` in C/CUDA). Supported on all
    Gluon-compatible hardware targets, including NVIDIA Hopper/Blackwell and
    AMD CDNA3/CDNA4/RDNA3/RDNA4/GFX1250 architectures.

    Examples
    --------
    Use `float32` to specify element types in kernel argument annotations:

```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def scale_kernel(
         x_ptr: ttgl.pointer_type[ttgl.float32],
         scale: ttgl.float32,
         size: ttgl.int32,
     ):
         # Kernel logic here
         pass
```

---

### triton.experimental.gluon.language.float64

.. py:data:: triton.experimental.gluon.language.float64

   64-bit floating-point data type.

   Represents IEEE 754 double-precision floating-point numbers. Use this type
   for high-precision arithmetic within Gluon kernels. Hardware support and
   performance characteristics vary across GPU architectures (e.g., NVIDIA
   Hopper/Blackwell, AMD CDNA).

   Notes
   -----
   Typically accessed via the alias `ttgl.float64` after importing:

```python
   import triton.experimental.gluon.language as ttgl

Examples
--------
Specify a kernel pointer argument as 64-bit float:

.. code-block:: python

   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def load_kernel(ptr: ttgl.pointer[ttgl.float64], size: ttgl.int32):
       # Implementation using float64 precision
       pass
```

---

### triton.experimental.gluon.language.float8e4b15

## float8e4b15

8-bit floating point data type with 4 exponent bits and bias 15.

`float8e4b15` is a predefined dtype instance representing the FP8
E4M3 format. It enables memory-efficient storage and compute for
deep learning workloads on supported hardware accelerators.

### Notes
Hardware support includes NVIDIA Hopper (H100), Blackwell (B200),
and AMD CDNA3 (MI300X). This dtype is distinct from
`float8e4m3` or `float8e5m2` depending on the backend
implementation details regarding NaN propagation and infinity
support.

### Examples
Specify the data type for a kernel argument or shared memory
allocation:

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def fp8_kernel(x_ptr: ttgl.pointer[ttgl.float8e4b15], size: int):
     # Load FP8 data from global memory
     data = ttgl.load(x_ptr)
     ...
```

---

### triton.experimental.gluon.language.float8e4b8

8-bit floating point type with 4 exponent bits and exponent bias of 8.

This dtype corresponds to the NVIDIA Hopper FP8 E4M3 format. Unlike IEEE 754 E4M3 (bias 7), this format uses an exponent bias of 8, sacrificing NaN representation for a larger dynamic range. It is optimized for mixed-precision compute on Tensor Cores.

### Notes
Hardware support requires NVIDIA Hopper (H100) or later architectures.
This type is suitable for mixed-precision matrix multiplications via Warpgroup MMA.
In Gluon, explicit layout control is required when using this dtype in shared memory or TMA operations.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr: ttgl.pointer[ttgl.float8e4b8], n: ttgl.int32):
     pid = gluon.program_id(0)
     x = ttgl.load(x_ptr + pid)
     ttgl.store(x_ptr + pid, x)
```

---

### triton.experimental.gluon.language.float8e4nv

.. py:data:: triton.experimental.gluon.language.float8e4nv

   NVIDIA FP8 (E4M3) data type instance.

   Represents an 8-bit floating point format with 4 exponent bits and 3
   mantissa bits, compliant with NVIDIA Hopper and Blackwell architecture
   specifications. Equivalent to `torch.float8_e4m3fn`.

   Notes
   -----
   Supported on Compute Capability 9.0+ (Hopper) and later. This format
   prioritizes precision over dynamic range compared to `float8e5`.
   Infinities are not represented; overflow results in NaN.

   Examples
   --------
   Specify tensor element types in Gluon kernel signatures:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def process_fp8(X: ttgl.pointer[ttgl.float8e4nv], N: int):
       # Explicit control over FP8 layout and memory
       pass
```

---

### triton.experimental.gluon.language.float8e5

`float8e5` is a data type instance representing the FP8 E5M2 floating-point format.

This type defines an 8-bit floating-point representation consisting of 1 sign bit,
5 exponent bits, and 2 mantissa bits. It is optimized for deep learning workloads
on NVIDIA Hopper (H100) and Blackwell GPU architectures.

### Notes
The `float8e5` (E5M2) format provides a wider dynamic range compared to
`float8e4` (E4M3), making it suitable for activations and gradients with
high variance. However, it offers lower mantissa precision (2 bits vs 3 bits).

Hardware support requires compute capability >= 9.0. When using this type,
ensure pointers are explicitly cast to `float8e5` to enable Tensor Core
accumulation paths where supported.

### Examples
The following example demonstrates loading data as `float8e5` and converting
it to `float32` for accumulation:

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def fp8_load_kernel(in_ptr, out_ptr, BLOCK_SIZE: ttgl.constexpr):
     offs = ttgl.arange(0, BLOCK_SIZE)
     mask = offs < BLOCK_SIZE

     # Load data explicitly as float8e5 (E5M2)
     x = ttgl.load(in_ptr + offs, mask=mask, dtype=ttgl.float8e5)

     # Convert to float32 for computation
     y = x.to(ttgl.float32)

     ttgl.store(out_ptr + offs, y, mask=mask)
```

---

### triton.experimental.gluon.language.float8e5b16

.. py:data:: triton.experimental.gluon.language.float8e5b16

   8-bit floating point data type with 5 exponent bits and bias 16 (E5M2).

   This dtype instance represents the NVIDIA FP8 E5M2 format, commonly used for
   high-performance matrix operations and memory bandwidth optimization on
   modern GPU architectures. It is available for use in Gluon kernel signatures,
   shared memory allocations, and explicit layout specifications.

   .. rubric:: Notes

   Hardware support is required for native FP8 operations. This type is
   primarily targeted at NVIDIA Hopper (H100) and Blackwell architectures,
   as well as AMD CDNA3/CDNA4 accelerators. On hardware lacking native FP8
   support, operations may emulate behavior via conversion to `float16`
   or `float32`.

   .. rubric:: Examples

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def fp8_kernel(
       x_ptr,
       y_ptr,
       n_elements,
   ):
       pid = gluon.program_id(0)
       x = gluon.load(x_ptr + pid, dtype=ttgl.float8e5b16)
       y = x * 2.0
       gluon.store(y_ptr + pid, y)

   # Allocate shared memory with explicit FP8 layout
   sm = gluon.alloc((128,), dtype=ttgl.float8e5b16, layout=gluon.swizzle_layout())
```

---

### triton.experimental.gluon.language.floor

```python
floor(x, _semantic=None)
```

## floor

Computes the element-wise floor of the input tensor.

**`floor(x, _semantic=None)`**

   Computes the element-wise floor of `x`.

   Parameters
   ----------
   x : Block
       The input tensor. Must be a floating-point type (fp32 or fp64).
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually.

   Returns
   -------
   Block
       A tensor containing the floor of each element in `x`.

   Notes
   -----
   This operation is only supported for floating-point dtypes (fp32, fp64).
   The floor function returns the largest integer less than or equal to each
   element.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
       pid = ttgl.program_id(0)
       offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offs)
       y = ttgl.floor(x)
       ttgl.store(y_ptr + offs, y)
```

---

### triton.experimental.gluon.language.fma

```python
fma(x, y, z, _semantic=None)
```

## fma


**`fma(x, y, z, _semantic=None)`**

   Computes the element-wise fused multiply-add of `x`, `y`, and `z`.

   Performs the operation `x * y + z` with a single rounding step, providing
   improved numerical precision compared to separate multiply and add operations.

   Parameters
   ----------
   x : Block
       First input tensor (multiplicand).
   y : Block
       Second input tensor (multiplicand).
   z : Block
       Third input tensor (addend).
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually.

   Returns
   -------
   Block
       Tensor containing the element-wise fused multiply-add result.

   Notes
   -----
   - All input tensors undergo type legalization to ensure compatible dtypes.
   - FMA operations provide better precision than separate multiply and add.
   - Hardware support varies by GPU architecture (NVIDIA Hopper/Blackwell, AMD CDNA3/CDNA4).
   - Must be called within a `@gluon.jit` decorated kernel.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, z_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       pid = ttgl.program_id(0)
       offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offs)
       y = ttgl.load(y_ptr + offs)
       z = ttgl.load(z_ptr + offs)
       result = ttgl.fma(x, y, z)
       ttgl.store(out_ptr + offs, result)
```

---

### triton.experimental.gluon.language.fp4_to_fp

```python
fp4_to_fp(src, elem_type, axis, _semantic=None)
```

## fp4_to_fp


**`fp4_to_fp(src, elem_type, axis, _semantic=None)`**

   Upcast a tensor from fp4 (e2m1) to another floating point type.

   Parameters
   ----------
   src : tensor
       Input tensor with fp4 (e2m1) elements to upcast.
   elem_type : dtype
       Target floating point data type (e.g., `float16`, `float32`, `bfloat16`).
   axis : int
       The axis along which the fp4 elements are packed. This determines how the
       upcast operation interprets the tensor layout.
   _semantic : GluonSemantic, optional
       Internal semantic handler. Do not pass explicitly; this is provided
       automatically when called from a `@gluon.jit` decorated function.

   Returns
   -------
   tensor
       A tensor with elements upcasted to the specified `elem_type`.

   Notes
   -----
   The fp4 format (e2m1) is a 4-bit floating point format with 1 sign bit,
   2 exponent bits, and 1 mantissa bit. This format is supported on NVIDIA
   Hopper and Blackwell architectures for efficient tensor core operations.

   The `axis` parameter is critical for correct interpretation of packed fp4
   values. Typically, fp4 values are packed in pairs along the specified axis,
   with two 4-bit values stored in each 8-bit byte.

   This operation is only valid when called from within a `@gluon.jit`
   decorated kernel function.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def upcast_kernel(src_ptr, dst_ptr, BLOCK_SIZE: tl.constexpr):
       # Load fp4 tensor from global memory
       src = ttgl.load(src_ptr + tl.arange(0, BLOCK_SIZE))
       
       # Upcast fp4 to float16 along axis 0
       fp16_tensor = ttgl.fp4_to_fp(src, ttgl.float16, axis=0)
       
       # Store upcasted results
       ttgl.store(dst_ptr + tl.arange(0, BLOCK_SIZE), fp16_tensor)
```

---

### triton.experimental.gluon.language.full

```python
full(shape, value, dtype, layout=None, _semantic=None)
```

## full

Create a tensor filled with a scalar value.

**`triton.experimental.gluon.language.full(shape, value, dtype, layout=None)`**

   Create a tensor filled with a scalar value, with specified shape, dtype, and layout.

   Parameters
   ----------
   shape : Sequence[int]
       The shape of the tensor.
   value : int or float
       The fill value for all elements in the tensor.
   dtype : dtype
       The data type for the tensor elements (e.g., `ttgl.float32`, `ttgl.int32`).
   layout : DistributedLayout, optional
       The layout of the output tensor. Defaults to `AutoLayout()` if not specified.

   Returns
   -------
   tensor
       A tensor where every element equals `value`.

   Notes
   -----
   This function must be called within a Gluon JIT kernel (decorated with `@gluon.jit`).
   The `_semantic` parameter is internal and should not be provided by users.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel():
       # Create a 128x128 tensor filled with 1.0 (float32)
       ones = ttgl.full((128, 128), 1.0, ttgl.float32)

       # Create a 64-element tensor filled with 0 (int32) with explicit layout
       from triton.experimental.gluon.language._layouts import BlockedLayout
       layout = BlockedLayout([1, 1], [64], [32], [1, 0])
       zeros = ttgl.full((64,), 0, ttgl.int32, layout=layout)
```

---

### triton.experimental.gluon.language.full_like

```python
full_like(input, value, shape=None, dtype=None, layout=None)
```

**`full_like(input, value, shape=None, dtype=None, layout=None)`**

    Create a tensor filled with a specific value, inheriting properties from a reference tensor.

    Parameters
    ----------
    input : tensor
        Reference tensor to infer default shape, dtype, and layout.
    value : int or float
        The fill value assigned to every element.
    shape : Sequence[int], optional
        Target shape of the output tensor. Defaults to `input.shape`.
    dtype : dtype, optional
        Target data type of the output tensor. Defaults to `input.dtype`.
    layout : DistributedLayout, optional
        Target distributed layout of the output tensor. Defaults to `input.layout`.

    Returns
    -------
    tensor
        A tensor where every element equals `value`.

    Notes
    -----
    This function is a JIT-compiled Gluon kernel primitive. It provides explicit control
    over memory layout via the `layout` argument, distinguishing it from standard
    Triton or NumPy equivalents.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(X, Y):
         ones = ttgl.full_like(X, 1.0)
         ttgl.store(Y, ones)
```

---

### triton.experimental.gluon.language.gather

```python
gather(src, index, axis, _semantic=None)
```

gather(src, index, axis, _semantic=None)

Gather elements from a tensor along a specified axis using an index tensor.

For each output position, the operation selects elements from `src` where the
coordinate at the gather axis is replaced by the corresponding value in `index`.

### Parameters
src : tensor
    The source tensor from which to gather elements.
index : tensor
    Tensor specifying which indices to gather along the axis. The output shape
    matches `index` shape, with the gather axis dimension from `src` replaced.
axis : int
    The axis along which to gather values.
_semantic : GluonSemantic, optional
    Internal parameter for semantic operations. Do not set manually.

### Returns
tensor
    A Gluon tensor containing the gathered elements.

### Notes
This function can also be called as a member function on :py`tensor`,
as `x.gather(...)` instead of `gather(x, ...)`.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(...):
     src = ...  # tensor of shape [M, N]
     index = ...  # tensor of shape [M, K]
     # Gather along axis 1: result[i, j] = src[i, index[i, j]]
     result = ttgl.gather(src, index, axis=1)  # shape [M, K]
```

---

### triton.experimental.gluon.language.histogram

```python
histogram(input, num_bins, mask=None, layout=None, _semantic=None, _generator=None)
```

## histogram

Compute a histogram of a 1D integer tensor.

### Parameters
input : tensor
    1D tensor of integer values.
num_bins : int
    Number of bins. Bins have width 1 and start at 0.
mask : tensor, optional
    Boolean mask to exclude elements when `False`.
layout : DistributedLayout, optional
    Destination layout of the output histogram.

### Returns
tensor
    1D int32 tensor of length `num_bins` with the requested layout.

### Notes
This operation is only available in Gluon kernels decorated with `@gluon.jit`.
The histogram counts occurrences of each integer value in the input tensor,
placing the count for value `i` in bin `i`. Values outside the range
`[0, num_bins)` are ignored.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def histogram_kernel(input_ptr, output_ptr, num_bins: ttgl.constexpr):
     input = ttgl.load(input_ptr)
     hist = ttgl.histogram(input, num_bins)
     ttgl.store(output_ptr, hist)
```

---

### triton.experimental.gluon.language.inline_asm_elementwise

```python
inline_asm_elementwise(asm: 'str', constraints: 'str', args: 'Sequence', dtype: 'Union[dtype, Sequence[dtype]]', is_pure: 'bool', pack: 'int', _semantic=None)
```

## inline_asm_elementwise


.. autofunction:: inline_asm_elementwise

Execute inline assembly over a tensor, applying the assembly function elementwise.

### Parameters
asm : str
    Assembly code to execute. Must match the target's assembly format (e.g., PTX for NVIDIA GPUs).
constraints : str
    Inline assembly constraints in `LLVM format <https://llvm.org/docs/LangRef.html#inline-asm-constraint-string>`_.
args : Sequence[tensor]
    Input tensors whose values are passed to the assembly block. Tensors are implicitly broadcasted to the same shape.
dtype : dtype or Sequence[dtype]
    Element type(s) of the returned tensor(s). Can be a single dtype or a tuple of dtypes for multiple outputs.
is_pure : bool
    If True, the compiler assumes the assembly block has no side-effects and may optimize accordingly.
pack : int
    Number of elements processed by one invocation of the inline assembly. Input elements smaller than 4 bytes are packed into 4-byte registers.

### Returns
tensor or tuple[tensor, ...]
    One tensor or a tuple of tensors with the specified dtypes.

### Notes
Each invocation of the inline assembly processes `pack` elements at a time. The exact set of inputs a block receives is unspecified.

This operation does not support empty `dtype` -- the assembly must return at least one tensor, even if unused. Work around this by returning a dummy tensor of arbitrary type.

### Examples
Example using PTX assembly to convert uint8 to int32, then to float32, and compute max with another float32 tensor:

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(A, B, C, D, BLOCK: ttgl.constexpr):
     a = ttgl.load(A + ttgl.arange(0, BLOCK))  # uint8 tensor
     b = ttgl.load(B + ttgl.arange(0, BLOCK))  # float32 tensor

     # For each (a,b) in zip(a,b), perform:
     # - Convert a to int32 (ai)
     # - Convert ai to float32
     # - Compute max(ai, b)
     # - Return ai and max
     # Process 4 elements at a time
     (c, d) = ttgl.inline_asm_elementwise(
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
             # 8 output registers: $0-$3=ai0-3, $4-$7=m0-3
             "=r,=r,=r,=r,=r,=r,=r,=r,"
             # 5 input registers: $8=packed_a, $9-$12=b0-3
             "r,r,r,r,r"),
         args=[a, b],
         dtype=(ttgl.int32, ttgl.float32),
         is_pure=True,
         pack=4,
     )
     ttgl.store(C + ttgl.arange(0, BLOCK), c)
     ttgl.store(D + ttgl.arange(0, BLOCK), d)
```

---

### triton.experimental.gluon.language.int1

.. py:data:: triton.experimental.gluon.language.int1

   1-bit signed integer data type.

   Represents a single-bit integer value. In Gluon kernels, this type is primarily used for predicates, boolean masks, or bit-packed memory layouts where minimal storage occupancy is required.

   Notes
   -----
   Operations involving `int1` are typically lowered to predicate registers or bit-wise logic instructions depending on the target architecture (e.g., NVIDIA Hopper/Blackwell, AMD CDNA3/RDNA3). While semantically equivalent to boolean values, `int1` exposes explicit integer representation for low-level memory control.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>> @gluon.jit
   ... def kernel(ptr, size):
   ...     mask = ttgl.full((size,), 1, dtype=ttgl.int1)
   ...     # Use mask for predicate logic
   ...     pass

---

### triton.experimental.gluon.language.int16

.. py:data:: int16
   :module: triton.experimental.gluon.language

   16-bit signed integer data type.

   Represents a 16-bit signed integer scalar or tensor element type within
   Gluon kernels. Use this dtype to specify precision for pointers, memory
   operations, and arithmetic.

   Examples
   --------
   Specify `int16` precision when loading or storing data:

```python
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl

    @gluon.jit
    def load_kernel(ptr, n):
        idx = ttgl.arange(0, n)
        mask = idx < n
        # Load 16-bit signed integers
        data = ttgl.load(ptr + idx, mask=mask, dtype=ttgl.int16)
        ttgl.store(ptr + idx, data, mask=mask)

Notes
-----
*   **Bit width**: 16
*   **Signedness**: Signed
*   **Range**: -32768 to 32767
*   **Alignment**: 2 bytes
```

---

### triton.experimental.gluon.language.int32

32-bit signed integer data type.

Represents a scalar 32-bit signed integer type for use within Gluon JIT kernels. This type ensures consistent precision across supported GPU architectures (NVIDIA Hopper/Blackwell, AMD CDNA/RDNA).

### Notes
Use this dtype for explicit type annotations, constant creation, or casting inside `@gluon.jit` functions. It corresponds to `i32` in underlying assembly (PTX/GCN).

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, n):
     pid = ttgl.program_id(0).to(ttgl.int32)
     val = ttgl.load(x_ptr + pid).to(ttgl.int32)
     ttgl.store(x_ptr + pid, val + 1)
```

---

### triton.experimental.gluon.language.int64

.. py:data:: triton.experimental.gluon.language.int64

   64-bit signed integer scalar type.

   Represents a 64-bit signed integer value within Gluon kernels. This type is used for type annotations in kernel signatures and arithmetic operations requiring 64-bit precision.

   Notes
   -----
   Operations involving `int64` are compiled to native 64-bit integer instructions on supported architectures (NVIDIA Hopper/Blackwell, AMD CDNA3/RDNA3+). Overflow behavior follows standard two's complement arithmetic.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x: ttgl.int64):
       return x + 1
```

---

### triton.experimental.gluon.language.int8

8-bit signed integer data type.

Represents the signed 8-bit integer type in Gluon kernels. This dtype is
used for type annotations, pointer definitions, and shared memory
allocations. It is supported across all Gluon backends (NVIDIA Hopper/Blackwell,
AMD CDNA/RDNA).

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(ptr: ttgl.pointer[ttgl.int8]):
     pass
```

---

### triton.experimental.gluon.language.join

```python
join(a, b, _semantic=None)
```

## join

Join two tensors along a new minor dimension.

Combines two input tensors by stacking them along a new innermost dimension of
size 2. The inputs are broadcasted to match shapes before joining. This
operation is the inverse of `split()`.

### Parameters
a : tensor
    The first input tensor.
b : tensor
    The second input tensor.

### Returns
tensor
    A tensor with shape equal to the broadcasted shape of `a` and `b`,
    plus a new minor dimension of size 2.

### Notes
The two input tensors are broadcasted to the same shape before joining. The
output tensor has one additional dimension compared to the inputs, with size 2
in that dimension.

To join more than two tensors, chain multiple calls to this function. This
reflects Triton's constraint that tensor dimensions must have power-of-two
sizes.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(X_ptr, Y_ptr, Z_ptr, BLOCK: tl.constexpr):
     a = ttgl.load(X_ptr + ttgl.arange(0, BLOCK))
     b = ttgl.load(Y_ptr + ttgl.arange(0, BLOCK))
     # Join two 1D tensors of shape (BLOCK,) into (BLOCK, 2)
     c = ttgl.join(a, b)
     ttgl.store(Z_ptr, c)

 # Given two scalars, returns a tensor of shape (2,)
 scalar_a = ttgl.full([], 1.0, ttgl.float32)
 scalar_b = ttgl.full([], 2.0, ttgl.float32)
 joined = ttgl.join(scalar_a, scalar_b)  # shape: (2,)

 # Given two tensors of shape (4, 8), produces (4, 8, 2)
 tensor_a = ttgl.full([4, 8], 1.0, ttgl.float32)
 tensor_b = ttgl.full([4, 8], 2.0, ttgl.float32)
 joined = ttgl.join(tensor_a, tensor_b)  # shape: (4, 8, 2)
```

---

### triton.experimental.gluon.language.load

```python
load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False, _semantic=None)
```

## load


**`load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False)`**

   Load data from memory at the location defined by `pointer`.

   Returns a tensor of data whose values are loaded from memory. The behavior depends on the type of `pointer`:

   1. **Single element pointer**: A scalar is loaded. `mask` and `other` must be scalars, `other` is implicitly typecast to `pointer.dtype.element_ty`, and `boundary_check` and `padding_option` must be empty.

   2. **N-dimensional tensor of pointers**: An N-dimensional tensor is loaded. `mask` and `other` are implicitly broadcast to `pointer.shape`, `other` is implicitly typecast to `pointer.dtype.element_ty`, and `boundary_check` and `padding_option` must be empty.

   3. **Block pointer** (defined by `make_block_ptr()`): A tensor is loaded. `mask` and `other` must be `None`, and `boundary_check` and `padding_option` can be specified to control out-of-bound access behavior.

   Parameters
   ----------
   pointer : `triton.PointerType` or block of `triton.PointerType`
       Pointer to the data to be loaded.
   mask : block of `triton.int1`, optional
       If `mask[idx]` is false, do not load the data at address `pointer[idx]`. Must be `None` with block pointers.
   other : block, optional
       If `mask[idx]` is false, return `other[idx]`.
   boundary_check : tuple of ints, optional
       Tuple of integers indicating the dimensions which should perform the boundary check.
   padding_option : str, optional
       Padding value to use while out of bounds. Should be one of `{"", "zero", "nan"}`. `""` means an undefined value.
   cache_modifier : str, optional
       Changes cache option in NVIDIA PTX. Should be one of `{"", ".ca", ".cg", ".cv"}`, where:

       - `.ca`: cache at all levels
       - `.cg`: cache at global level (cache in L2 and below, not L1)
       - `.cv`: don't cache and fetch again

       See `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
   eviction_policy : str, optional
       Changes eviction policy in NVIDIA PTX.
   volatile : bool, optional
       Changes volatile option in NVIDIA PTX.

   Returns
   -------
   tensor
       A tensor containing the loaded data.

   Notes
   -----
   For block pointers, `mask` and `other` must be `None`. Use `boundary_check` and `padding_option` to control out-of-bound access behavior instead.

   Examples
   --------
   Load a scalar from a single pointer:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr):
       value = ttgl.load(ptr)

Load a tensor with masking:

.. code-block:: python

   @gluon.jit
   def kernel(ptr, mask, other):
       data = ttgl.load(ptr, mask=mask, other=other)

Load from a block pointer with boundary checking:

.. code-block:: python

   @gluon.jit
   def kernel(block_ptr):
       data = ttgl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
```

---

### triton.experimental.gluon.language.log

```python
log(x, _semantic=None)
```

## log


**`log(x, _semantic=None)`**

   Computes the element-wise natural logarithm of `x`.

   Parameters
   ----------
   x : Block
      The input tensor. Must be floating point (fp32 or fp64).

   Returns
   -------
   Block
      A tensor containing the natural logarithm of each element in `x`.

   Notes
   -----
   This function is only supported for floating point dtypes (fp32, fp64).

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, n: tl.constexpr):
       pid = ttgl.program_id(0)
       offs = ttgl.arange(0, n)
       mask = offs < n
       x = ttgl.load(x_ptr + offs, mask=mask)
       y = ttgl.log(x)
       ttgl.store(y_ptr + offs, y, mask=mask)
```

---

### triton.experimental.gluon.language.log2

```python
log2(x, _semantic=None)
```

## log2

Compute the base-2 logarithm of each element in a tensor.

### Parameters
x : tensor
    Input tensor. Must have floating-point dtype (`float32` or `float64`).

### Returns
tensor
    Tensor containing the base-2 logarithm of each element in `x`. The output
    has the same shape and dtype as the input.

### Notes
This operation is only supported for `float32` and `float64` dtypes. Passing
other dtypes will result in a compilation error.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def log2_kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     pid = ttgl.program_id(0)
     offsets = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
     x = ttgl.load(x_ptr + offsets)
     y = ttgl.log2(x)
     ttgl.store(y_ptr + offsets, y)
```

---

### triton.experimental.gluon.language.map_elementwise

```python
map_elementwise(scalar_fn: 'Callable[..., Tuple[tensor, ...]]', *args: 'tensor', pack=1, _semantic=None, _generator=None)
```

## map_elementwise


**`map_elementwise(scalar_fn, *args, pack=1)`**

   Map a scalar function element-wise over input tensors.

   Applies `scalar_fn` to each element of the input tensors after implicit
   broadcasting. Enables per-element control flow that avoids computing both
   branches of conditional logic (unlike `where()` which eagerly evaluates
   both sides).

   Parameters
   ----------
   scalar_fn : Callable[..., Tuple[tensor, ...]]
      JIT-compiled function that operates on scalar values. Must accept scalar
      arguments and return a tensor or tuple of tensors.
   *args : tensor
      Input tensors to map over. All tensors are broadcasted to a common shape
      before applying the function.
   pack : int, optional
      Number of elements processed per function invocation. Default is 1.
      Increasing pack may improve throughput for lightweight operations.

   Returns
   -------
   tensor or Tuple[tensor, ...]
      Output tensor(s) with the same shape as the broadcasted inputs. Returns
      a single tensor if `scalar_fn` returns one tensor, otherwise returns
      a tuple of tensors.

   Notes
   -----
   Unlike `where()`, this function enables true control flow at the element
   level. Conditional branches inside `scalar_fn` are only executed for
   elements that satisfy the condition, avoiding unnecessary computation.

   The `pack` parameter allows processing multiple elements in a single
   function call, which can reduce launch overhead for simple operations.
   All packed elements must produce consistent types.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def selu_scalar(x, alpha):
       if x > 0:
           return x
       else:
           return alpha * (ttgl.exp(x) - 1)

   @gluon.jit
   def selu(x, alpha):
       return ttgl.map_elementwise(selu_scalar, x, alpha)

.. code-block:: python

   # Example with pack > 1 for improved throughput
   @gluon.jit
   def relu_scalar(x):
       if x > 0:
           return x
       else:
           return 0.0

   @gluon.jit
   def relu_packed(x):
       return ttgl.map_elementwise(relu_scalar, x, pack=4)
```

---

### triton.experimental.gluon.language.max

```python
max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
```

## ttgl.max


**`max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)`**

   Returns the maximum value of all elements in the input tensor along the specified axis.

   The reduction operation is associative and commutative. For floating-point inputs with bitwidth less than 32, values are promoted to float32 before reduction. For integer inputs with bitwidth less than 32, values are promoted to int32.

   Parameters
   ----------
   input : Tensor
       The input tensor containing values to reduce.
   axis : int, optional
       The dimension along which the reduction should be performed. If None, reduces all dimensions.
   return_indices : bool, optional
       If True, returns both the maximum value and the index corresponding to the maximum value. Default is False.
   return_indices_tie_break_left : bool, optional
       If True, in case of a tie (multiple elements have the same maximum value), returns the left-most index for values that aren't NaN. Only used when return_indices is True. Default is True.
   keep_dims : bool, optional
       If True, keeps the reduced dimensions with length 1 in the output shape. Default is False.

   Returns
   -------
   output : Tensor
       The maximum values. If return_indices is True, returns a tuple of (max_values, max_indices).

   Notes
   -----
   This function can also be called as a member function on :py`tensor`, as `x.max(...)` instead of `max(x, ...)`.

   bfloat16 inputs are automatically promoted to float32 before reduction. For other floating-point types with bitwidth less than 32, values are promoted to float32. For integer types with bitwidth less than 32, values are promoted to int32.

   When return_indices is True, the tie-breaking behavior is controlled by return_indices_tie_break_left. When True, the left-most index is returned in case of ties. When False, tie-breaking is implementation-dependent and may be faster.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, output_ptr, n: ttgl.constexpr):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid * n)
       
       # Compute maximum along axis 0
       max_val = ttgl.max(x, axis=0)
       ttgl.store(output_ptr + pid, max_val)

   # Compute maximum with indices
   @gluon.jit
   def kernel_with_indices(x_ptr, output_ptr, indices_ptr, n: ttgl.constexpr):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid * n)
       
       max_val, max_idx = ttgl.max(x, axis=0, return_indices=True)
       ttgl.store(output_ptr + pid, max_val)
       ttgl.store(indices_ptr + pid, max_idx)

   # Keep reduced dimensions
   @gluon.jit
   def kernel_keep_dims(x_ptr, output_ptr, n: ttgl.constexpr):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid * n)
       
       max_val = ttgl.max(x, axis=0, keep_dims=True)
       ttgl.store(output_ptr + pid * n, max_val)

   # Member function syntax
   @gluon.jit
   def kernel_member_fn(x_ptr, output_ptr, n: ttgl.constexpr):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid * n)
       
       max_val = x.max(axis=0)
       ttgl.store(output_ptr + pid, max_val)
```

---

### triton.experimental.gluon.language.max_constancy

```python
max_constancy(input, values, _semantic=None)
```

## max_constancy


**`max_constancy(input, values, _semantic=None)`**

   Inform the compiler that consecutive groups of values in `input` are constant.

   This optimization hint indicates that the input tensor contains repeated values in
   contiguous blocks, enabling the compiler to generate more efficient code.

   Parameters
   ----------
   input : tensor
       The input tensor whose constancy pattern is being described.
   values : list of constexpr
       List of constexpr integers specifying the group sizes. Each element indicates
       that consecutive groups of that size contain equal values.
   _semantic : GluonSemantic, optional
       Internal semantic handler. Do not pass explicitly; this is managed by the
       `@gluon.jit` decorator.

   Returns
   -------
   tensor
       The input tensor annotated with constancy information.

   Notes
   -----
   For each group size `d` in `values`, the compiler assumes that every
   consecutive block of `d` elements in `input` contains identical values.

   For example, if `values` is `[4]`, then the input should have the
   pattern `[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ...]` where each group of 4
   consecutive elements is constant.

   All elements in `values` must be compile-time constants (`constexpr`).
   Passing non-constexpr values will raise a `TypeError`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, n):
       pid = ttgl.program_id(0)
       offsets = ttgl.arange(0, n)
       x = ttgl.load(x_ptr + offsets)

       # Inform compiler that every 4 consecutive values are constant
       x_const = ttgl.max_constancy(x, [ttgl.constexpr(4)])

       # Use x_const for subsequent operations
       y = x_const * 2.0
       ttgl.store(x_ptr + offsets, y)
```

---

### triton.experimental.gluon.language.max_contiguous

```python
max_contiguous(input, values, _semantic=None)
```

## max_contiguous

**`max_contiguous(input, values, _semantic=None)`**

   Inform the compiler that the first N elements in `input` are contiguous in memory.

   Parameters
   ----------
   input : tensor
       The input tensor to annotate with contiguity information.
   values : constexpr or list of constexpr
       Compile-time constant(s) specifying the number of contiguous elements.
       Each value must be a `constexpr[int]`.
   _semantic : GluonSemantic, optional
       Internal semantic handler. Do not set manually; provided automatically
       when using `@gluon.jit`.

   Returns
   -------
   tensor
       The input tensor with contiguity hints applied for compiler optimizations.

   Notes
   -----
   This function provides memory access pattern hints to the compiler, enabling
   optimizations for contiguous memory operations. The hints are used during
   code generation to improve load/store efficiency.

   All values must be compile-time constants. Passing runtime values will raise
   a `TypeError`. Multiple contiguity hints can be provided as a list.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.language import constexpr

   @gluon.jit
   def kernel(x_ptr, BLOCK: constexpr):
       x = ttgl.load(x_ptr)
       x = ttgl.max_contiguous(x, [BLOCK])
       # Compiler can now optimize for contiguous access
       # ... rest of kernel
```

---

### triton.experimental.gluon.language.maximum

```python
maximum(x, y, propagate_nan: 'constexpr' = <PROPAGATE_NAN.NONE: 0>, _semantic=None)
```

## maximum


**`maximum(x, y, propagate_nan=PropagateNan.NONE)`**

   Computes the element-wise maximum of two tensors.

   Parameters
   ----------
   x : Block
       The first input tensor.
   y : Block
       The second input tensor.
   propagate_nan : PropagateNan, optional
       Whether to propagate NaN values. Defaults to `PropagateNan.NONE`.

   Returns
   -------
   Block
       A tensor containing the element-wise maximum values.

   Notes
   -----
   bfloat16 inputs are promoted to float32 before computing the maximum.

   The behavior when encountering NaN values is controlled by the
   `propagate_nan` parameter. See `triton.language.PropagateNan`
   for available options.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       pid = ttgl.program_id(0)
       offsets = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offsets)
       y = ttgl.load(y_ptr + offsets)
       maximum = ttgl.maximum(x, y)
       ttgl.store(out_ptr + offsets, maximum)
```

---

### triton.experimental.gluon.language.min

```python
min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)
```

## min


**`min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)`**

   Returns the minimum value of elements in the input tensor along the specified axis.

   Parameters
   ----------
   input : tensor
       The input tensor to reduce.
   axis : int, optional
       The dimension along which to compute the minimum. If `None`, reduces all dimensions.
   return_indices : bool, optional
       If `True`, also return the index of the minimum value. Default is `False`.
   return_indices_tie_break_left : bool, optional
       If `True` and there are multiple minimum values, return the left-most index.
       Default is `True`.
   keep_dims : bool, optional
       If `True`, the reduced axis is retained with length 1. Default is `False`.

   Returns
   -------
   min_vals : tensor
       The minimum values. If `return_indices` is `True`, returns a tuple of
       `(min_vals, min_indices)`.

   Notes
   -----
   - For floating-point types with bitwidth less than 32, values are promoted to
     `float32` before reduction.
   - For integer types with bitwidth less than 32, values are promoted to `int32`
     before reduction.
   - `bfloat16` is always promoted to `float32`.
   - The reduction operation is associative and commutative.
   - This function can be called as a member function: `x.min(...)` instead of
     `min(x, ...)`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(X, Y, BLOCK_SIZE: ttgl.constexpr):
       x = ttgl.load(X + ttgl.arange(0, BLOCK_SIZE))
       min_val = ttgl.min(x, axis=0)
       ttgl.store(Y, min_val)

   # Get minimum value and its index
   @gluon.jit
   def kernel_with_index(X, Y, Z, BLOCK_SIZE: ttgl.constexpr):
       x = ttgl.load(X + ttgl.arange(0, BLOCK_SIZE))
       min_val, min_idx = ttgl.min(x, axis=0, return_indices=True)
       ttgl.store(Y, min_val)
       ttgl.store(Z, min_idx)

   # Reduce along specific axis with keep_dims
   @gluon.jit
   def kernel_2d(X, Y, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr):
       x = ttgl.load(X + ttgl.arange(0, BLOCK_M)[:, None] * BLOCK_N + ttgl.arange(0, BLOCK_N)[None, :])
       min_val = ttgl.min(x, axis=1, keep_dims=True)  # Shape: [BLOCK_M, 1]
       ttgl.store(Y, min_val)
```

---

### triton.experimental.gluon.language.minimum

```python
minimum(x, y, propagate_nan: 'constexpr' = <PROPAGATE_NAN.NONE: 0>, _semantic=None)
```

## minimum


**`minimum(x, y, propagate_nan=PropagateNan.NONE)`**

   Computes the element-wise minimum of two tensors.

   Parameters
   ----------
   x : Block
       The first input tensor.
   y : Block
       The second input tensor.
   propagate_nan : PropagateNan, optional
       Whether to propagate NaN values. Defaults to `PropagateNan.NONE`.

   Returns
   -------
   Block
       A tensor containing the element-wise minimum values.

   Notes
   -----
   If either input is `bfloat16`, it is promoted to `float32` before
   computing the minimum. NaN propagation behavior is controlled by the
   `propagate_nan` parameter.

   .. seealso:: `triton.experimental.gluon.language.PropagateNan`

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offsets = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offsets)
       y = ttgl.load(y_ptr + offsets)
       minimum = ttgl.minimum(x, y)
       ttgl.store(out_ptr + offsets, minimum)
```

---

### triton.experimental.gluon.language.mul

```python
mul(x, y, sanitize_overflow: 'constexpr' = True, _semantic=None)
```

### mul

**`mul(x, y, sanitize_overflow=True)`**

   Element-wise multiplication of two tensors.

   Parameters
   ----------
   x : tensor
       First input tensor.
   y : tensor
       Second input tensor.
   sanitize_overflow : constexpr, optional
       If True (default), enables overflow sanitization for integer operations.

   Returns
   -------
   tensor
       Element-wise product of `x` and `y`.

   Notes
   -----
   Both inputs must have compatible shapes for broadcasting. The `sanitize_overflow`
   parameter controls whether integer overflow checks are inserted.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: ttgl.constexpr):
       x = ttgl.load(x_ptr)
       y = ttgl.load(y_ptr)
       z = ttgl.mul(x, y)
       ttgl.store(out_ptr, z)
```

---

### triton.experimental.gluon.language.multiple_of

```python
multiple_of(input, values, _semantic=None)
```

## multiple_of

**`multiple_of(input, values, _semantic=None)`**

   Inform the compiler that all values in `input` are multiples of specified constants.

   This hint enables the compiler to generate more efficient code by leveraging
   alignment guarantees.

   Parameters
   ----------
   input : tensor
       The input tensor whose values are guaranteed to be multiples of `values`.
   values : list of constexpr[int]
       Compile-time constant divisors. Each element must be a `constexpr`
       with an integer value.
   _semantic : GluonSemantic, optional
       Internal semantic handler. Do not pass explicitly; required for JIT compilation.

   Returns
   -------
   tensor
       The input tensor with alignment metadata attached.

   Notes
   -----
   This is a compiler hint that does not modify the tensor values at runtime.
   All elements in `values` must be `constexpr` integers. Passing
   non-constexpr values will raise a `TypeError`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, BLOCK: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offs = ttgl.arange(0, BLOCK)
       # Inform compiler that offsets are multiples of 4
       offs = ttgl.multiple_of(offs, [ttgl.constexpr(4)])
       x = ttgl.load(x_ptr + offs)
       # ... rest of kernel
```

---

### triton.experimental.gluon.language.num_ctas

```python
num_ctas(_semantic=None)
```

**`num_ctas()`**

   Returns the number of CTAs (Cooperative Thread Arrays) in the current kernel.

   Returns
   -------
   int
      The number of CTAs configured for the current kernel.

   Notes
   -----
   This function is only available within Gluon JIT-compiled kernels (decorated with
   `@gluon.jit <triton.experimental.gluon.jit>()`). The `_semantic` parameter
   is internal and should not be provided by users.

   CTAs are the basic execution unit in GPU kernels, analogous to thread blocks in
   CUDA. Multi-CTA kernels enable cluster-level execution on NVIDIA Hopper and
   later architectures.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel():
       num_ctas = ttgl.num_ctas()
       cta_id = ttgl.program_id(0)
       # Use num_ctas for cluster-aware indexing
       if cta_id < num_ctas:
           # Process data for this CTA
           pass
```

---

### triton.experimental.gluon.language.num_programs

```python
num_programs(axis, _semantic=None)
```

## num_programs


**`num_programs(axis, _semantic=None)`**

   Returns the number of program instances launched along the given axis.

   Parameters
   ----------
   axis : int
       The axis of the 3D launch grid. Must be 0, 1, or 2.
   _semantic : GluonSemantic, optional
       Internal semantic argument. Do not pass directly.

   Returns
   -------
   int
       The number of programs along the specified axis.

   Notes
   -----
   This function is typically used within Gluon kernels decorated with
   `@gluon.jit`. The `_semantic` parameter is automatically provided
   by the JIT compiler and should not be passed by users.

   Valid axis values correspond to the grid dimensions:
   
   - `axis=0`: X dimension
   - `axis=1`: Y dimension
   - `axis=2`: Z dimension

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, N):
       # Get the number of programs along the X axis
       num_prog_x = ttgl.num_programs(0)
       
       # Get the number of programs along the Y axis
       num_prog_y = ttgl.num_programs(1)
       
       # Program ID within the grid
       pid = ttgl.program_id(0)
       
       # Use num_programs for bounds checking
       if pid < num_prog_x:
           # Process data
           pass
```

---

### triton.experimental.gluon.language.num_warps

```python
num_warps(_semantic=None, _generator=None)
```

**`num_warps()`**

    Returns the number of warps that execute the current context.

    Returns
    -------
    int
        The number of warps allocated for the current kernel context. This value
        includes warps in warp-specialized regions.

    Notes
    -----
    This function queries the warp count configured for the current execution
    context. When called inside a :py`warp_specialize()` region, it returns
    the number of warps assigned to that specific partition rather than the
    total kernel warp count.

    Examples
    --------
    Query the number of warps within a Gluon kernel:

```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(...):
         num_w = ttgl.num_warps()
         # Use num_w for compile-time or runtime logic
```

---

### triton.experimental.gluon.language.permute

```python
permute(input, *dims, _semantic=None)
```

## permute


**`permute(input, *dims, _semantic=None)`**

   Permutes the dimensions of a tensor.

   Parameters
   ----------
   input : Block
       The input tensor to permute.
   *dims : int or tuple of int
       The desired ordering of dimensions. For example, `(2, 1, 0)`
       reverses the order of dimensions in a 3D tensor. Can be passed
       as individual arguments or as a tuple.

   Returns
   -------
   Block
       A tensor with dimensions permuted according to `dims`.

   Notes
   -----
   The `dims` argument can be passed as a tuple or as individual
   parameters. These are equivalent::

       permute(x, (2, 1, 0))
       permute(x, 2, 1, 0)

   :py`trans()` is equivalent to this function, except when `dims`
   is empty, it tries to swap the last two axes.

   This function can also be called as a member function on
   :py`tensor`, as `x.permute(...)` instead of
   `permute(x, ...)`.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>> @gluon.jit
   ... def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
   ...     x = ttgl.load(x_ptr)
   ...     y = ttgl.permute(x, 2, 1, 0)  # Reverse dimensions
   ...     ttgl.store(y_ptr, y)

---

### triton.experimental.gluon.language.pointer_type

```python
pointer_type(element_ty: 'dtype', address_space: 'int' = 1, const: 'bool' = False)
```

## pointer_type


**`pointer_type(element_ty, address_space=1, const=False)`**

   Represents a pointer type in Triton IR with explicit element type, address
   space, and constness qualifiers.

   Pointer types are used to define memory access patterns in Gluon kernels,
   enabling explicit control over shared memory, global memory, and other
   address spaces.

   Parameters
   ----------
   element_ty : dtype
       The element type that the pointer points to (e.g., `ttgl.int32`,
       `ttgl.float16`).
   address_space : int, optional
       The address space number. Default is 1 (global memory). Address space 0
       typically represents shared memory.
   const : bool, optional
       Whether the pointer points to constant data. Constant pointers cannot
       be used with store operations. Default is False.

   Attributes
   ----------
   element_ty : dtype
       The scalar element type of the pointer.
   address_space : int
       The address space number.
   const : bool
       Whether the pointer is constant.
   name : str
       Human-readable string representation (e.g., `pointer<int32>` or
       `const_pointer<float16>`).

   Methods
   -------
   to_ir(builder)
       Convert to MLIR pointer type for code generation.
   is_ptr()
       Returns True (identifies this type as a pointer).
   is_const()
       Returns True if the pointer is constant, False otherwise.
   mangle()
       Returns the mangled name for type encoding in kernel signatures.
   scalar
       Property returning the pointer type itself (pointer types are scalar).

   See Also
   --------
   dtype : Base type class for all Triton types.
   block_type : Type for blocked/tensor types.

   Examples
   --------
   Create a pointer to int32 in global memory:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr: ttgl.pointer_type(ttgl.int32)):
       # Kernel implementation
       pass

Create a constant pointer (read-only):

.. code-block:: python

   const_ptr_ty = ttgl.pointer_type(ttgl.float16, const=True)
   # const_ptr_ty.is_const() returns True

Create a shared memory pointer (address space 0):

.. code-block:: python

   shared_ptr_ty = ttgl.pointer_type(ttgl.int32, address_space=0)

Check pointer type properties:

.. code-block:: python

   ptr_ty = ttgl.pointer_type(ttgl.int64)
   print(ptr_ty.is_ptr())      # True
   print(ptr_ty.is_const())    # False
   print(ptr_ty.element_ty)    # int64
   print(ptr_ty.address_space) # 1
```

---

### triton.experimental.gluon.language.program_id

```python
program_id(axis, _semantic=None)
```

## program_id


**`program_id(axis, _semantic=None)`**

   Returns the ID of the current program instance along the given axis.

   Parameters
   ----------
   axis : int
       The axis of the 3D launch grid. Must be 0, 1, or 2.

   Returns
   -------
   int
       The program ID along the specified axis.

   Notes
   -----
   This function is only valid inside Gluon JIT kernels decorated with
   `@gluon.jit`. The `_semantic` parameter is internal and should not
   be provided by users.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, N):
       pid_x = ttgl.program_id(0)
       pid_y = ttgl.program_id(1)
       pid_z = ttgl.program_id(2)
       # Use program IDs to compute memory offsets
       offset = pid_x + pid_y * N + pid_z * N * N
       # ... rest of kernel
```

---

### triton.experimental.gluon.language.ravel

```python
ravel(x, can_reorder=False)
```

## ravel

Returns a contiguous flattened view of a tensor.

```python
 ravel(x, can_reorder=False)

```
### Parameters

x : Block
    The input tensor to flatten.

can_reorder : bool, optional
    If `True`, allows the compiler to reorder elements for efficiency.
    Default is `False`.

### Returns

Block
    A 1-D tensor containing all elements of `x` in contiguous memory.

### Notes

This function can also be called as a member function on :py`tensor`,
as `x.ravel(...)` instead of `ravel(x, ...)`.

The output tensor has shape `(x.numel,)` where `numel` is the total
number of elements in the input tensor.

### Examples

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     x = ttgl.load(x_ptr + ttgl.arange(0, BLOCK_SIZE))
     # Reshape to 2D
     x_2d = ttgl.reshape(x, (BLOCK_SIZE // 4, 4))
     # Flatten back to 1D
     x_flat = ttgl.ravel(x_2d)
     ttgl.store(y_ptr + ttgl.arange(0, BLOCK_SIZE), x_flat)

 # Using member function syntax
 @gluon.jit
 def kernel_member(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     x = ttgl.load(x_ptr + ttgl.arange(0, BLOCK_SIZE))
     x_2d = ttgl.reshape(x, (BLOCK_SIZE // 4, 4))
     x_flat = x_2d.ravel()  # Equivalent to ttgl.ravel(x_2d)
     ttgl.store(y_ptr + ttgl.arange(0, BLOCK_SIZE), x_flat)
```

---

### triton.experimental.gluon.language.reduce

```python
reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None)
```

## reduce


**`reduce(input, axis, combine_fn, keep_dims=False)`**

   Applies a reduction operation to all elements in input tensors along the specified axis.

   Parameters
   ----------
   input : tensor or tuple of tensors
       The input tensor(s) to reduce.
   axis : int or None
       The dimension along which to perform the reduction. If `None`, reduces across all dimensions.
   combine_fn : callable
       A JIT-compiled function (marked with `@triton.jit`) that combines two scalar tensors. Must be associative and commutative.
   keep_dims : bool, optional
       If `True`, the reduced dimensions are retained with length 1. Default is `False`.

   Returns
   -------
   tensor or tuple of tensors
       The reduced tensor(s). If `input` is a tuple, returns a tuple of reduced tensors.

   Notes
   -----
   This function can also be called as a member function on :py`tensor` objects,
   as `x.reduce(...)` instead of `reduce(x, ...)`.

   The `combine_fn` must be marked with `@triton.jit` and should take two scalar tensors
   as arguments, returning their combination. Common operations include addition, multiplication,
   maximum, or minimum.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>>
   >>> @gluon.jit
   >>> def add_fn(a, b):
   ...     return a + b
   >>>
   >>> @gluon.jit
   >>> def kernel():
   ...     x = ttgl.full([4, 8], 1.0, ttgl.float32)
   ...     # Reduce along axis 1 (sum across columns)
   ...     y = ttgl.reduce(x, axis=1, combine_fn=add_fn)
   ...     # y has shape [4]
   >>>
   >>> # Using keep_dims to preserve reduced dimension
   >>> @gluon.jit
   >>> def kernel_keep_dims():
   ...     x = ttgl.full([4, 8], 1.0, ttgl.float32)
   ...     y = ttgl.reduce(x, axis=1, combine_fn=add_fn, keep_dims=True)
   ...     # y has shape [4, 1]
   >>>
   >>> # Reduce across all dimensions
   >>> @gluon.jit
   >>> def kernel_all_dims():
   ...     x = ttgl.full([4, 8], 1.0, ttgl.float32)
   ...     y = ttgl.reduce(x, axis=None, combine_fn=add_fn)
   ...     # y is a scalar (shape [])
   >>>
   >>> # Using as member function
   >>> @gluon.jit
   >>> def kernel_member():
   ...     x = ttgl.full([4, 8], 1.0, ttgl.float32)
   ...     y = x.reduce(axis=0, combine_fn=add_fn)

---

### triton.experimental.gluon.language.reduce_or

```python
reduce_or(input, axis, keep_dims=False)
```

## reduce_or

Compute the bitwise OR reduction of all elements in the input tensor along the
provided axis.

### Parameters
input : tensor
    The input tensor containing integer values.
axis : int
    The dimension along which the reduction should be performed. If None,
    reduce all dimensions.
keep_dims : bool, optional
    If True, keep the reduced dimensions with length 1. Default is False.

### Returns
tensor
    The tensor containing the bitwise OR reduction result.

### Notes
The reduction operation is associative and commutative. This function only
supports integer types.

This function can also be called as a member function on tensor,
as `x.reduce_or(...)` instead of `reduce_or(x, ...)`.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     pid = ttgl.program_id(0)
     x = ttgl.load(x_ptr + pid * BLOCK_SIZE)
     result = ttgl.reduce_or(x, axis=0)
     ttgl.store(y_ptr + pid, result)
```

---

### triton.experimental.gluon.language.reshape

```python
reshape(input, *shape, can_reorder=False, _semantic=None, _generator=None)
```

## reshape


**`reshape(input, *shape, can_reorder=False)`**

   Returns a tensor with the same number of elements as `input` but with the
   provided shape.

   Parameters
   ----------
   input : Block
       The input tensor to reshape.
   *shape : int or tuple of int
       The new shape. Can be passed as a tuple or as individual integer
       parameters.
   can_reorder : bool, optional
       If True, allows reordering of elements during reshape. Default is False.

   Returns
   -------
   tensor
       A tensor with the same number of elements as `input` but with the
       new shape.

   Notes
   -----
   The `shape` argument can be passed as a tuple or as individual parameters:

```python
   # These are equivalent
   reshape(x, (32, 32))
   reshape(x, 32, 32)

This function can also be called as a member function on :py:class:`tensor`,
as ``x.reshape(...)`` instead of ``reshape(x, ...)``.

Examples
--------
Reshape a 1D tensor to 2D:

.. code-block:: python

   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       x = ttgl.load(x_ptr)
       # Reshape from [BLOCK_SIZE] to [BLOCK_SIZE // 2, 2]
       y = ttgl.reshape(x, BLOCK_SIZE // 2, 2)
       ttgl.store(out_ptr, y)
```

---

### triton.experimental.gluon.language.rsqrt

```python
rsqrt(x, _semantic=None)
```

## rsqrt


**`rsqrt(x, _semantic=None)`**

   Computes the element-wise inverse square root of the input tensor.

   Parameters
   ----------
   x : tensor
       The input tensor. Must have floating point dtype (`fp32` or `fp64`).
   _semantic : GluonSemantic, optional
       Internal semantic argument. Do not set manually.

   Returns
   -------
   tensor
       A tensor containing the inverse square root of each element in `x`.
       Equivalent to `1 / sqrt(x)`.

   Notes
   -----
   This function is only supported for `fp32` and `fp64` data types.
   Calling with other dtypes will raise an error.

   The function can be called as a module function or as a tensor method:
   `ttgl.rsqrt(x)` or `x.rsqrt()`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offs)
       y = ttgl.rsqrt(x)
       ttgl.store(y_ptr + offs, y)
```

---

### triton.experimental.gluon.language.set_auto_layout

```python
set_auto_layout(value, layout, _semantic=None)
```

## set_auto_layout


**`set_auto_layout(value, layout, _semantic=None)`**

    Convert a tensor with `AutoLayout` to a concrete `DistributedLayout`.

    This operation materializes the automatic layout inference into an explicit
    distributed layout, enabling further layout-aware operations.

    Parameters
    ----------
    value : tensor
        The input tensor with `AutoLayout`.
    layout : DistributedLayout
        The target concrete layout to apply to the tensor.
    _semantic : GluonSemantic, optional
        Internal parameter for semantic context. Do not set explicitly.

    Returns
    -------
    tensor
        The tensor with the new concrete layout applied.

    Notes
    -----
    This function is typically used internally by Gluon to resolve automatic
    layout inference. Users should prefer `convert_layout()` for explicit
    layout conversions in kernel code.

    The `_semantic` parameter is automatically provided by the `@gluon.jit`
    decorator and should not be set by users.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(x_ptr, y_ptr, BLOCK: ttgl.constexpr):
         # Load tensor with AutoLayout
         x = ttgl.load(x_ptr)
         
         # Convert to concrete layout
         layout = ttgl.BlockedLayout(...)
         x_concrete = ttgl.set_auto_layout(x, layout)
         
         # Now x_concrete has explicit layout for further operations
         y = x_concrete * 2.0
         ttgl.store(y_ptr, y)
```

---

### triton.experimental.gluon.language.shared_memory_descriptor

```python
shared_memory_descriptor(handle, element_ty, shape, layout, alloc_shape)
```

## shared_memory_descriptor


**`shared_memory_descriptor(handle, element_ty, shape, layout, alloc_shape)`**

   Represents a handle to a shared memory allocation in Gluon IR.

   Parameters
   ----------
   handle : ir.value
       The underlying IR value representing the shared memory allocation.
   element_ty : dtype
       The data type of elements stored in shared memory.
   shape : Sequence[int]
       The logical shape of the shared memory view.
   layout : SharedLayout
       The shared memory layout specifying how data is organized in memory.
   alloc_shape : Sequence[int]
       The physical allocation shape, which may differ from the logical shape.

   Attributes
   ----------
   dtype : dtype
       The element data type of the shared memory.
   shape : List[int]
       The logical shape of the shared memory view.
   rank : int
       The number of dimensions (length of shape).
   numel : int
       The total number of elements (product of shape dimensions).
   layout : SharedLayout
       The shared memory layout.

   Notes
   -----
   This class provides a low-level interface for managing shared memory in Gluon
   kernels. It supports various memory operations including load, store, gather,
   scatter, and view transformations (slice, index, permute, reshape).

   The descriptor separates the logical view (shape) from the physical allocation
   (alloc_shape), enabling efficient memory reuse and subview operations without
   additional allocations.

   Methods prefixed with underscore (e.g., `_reinterpret`, `_keep_alive`) are
   internal APIs and may change in future versions.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language import SharedLayout

   @gluon.jit
   def kernel(...):
       # Allocate shared memory
       layout = ttgl.SharedLayout(...)
       smem = ttgl.allocate_shared_memory(ttgl.float16, [128, 64], layout)

       # Store data to shared memory
       data = ttgl.full([128, 64], 0.0, ttgl.float16, layout)
       smem.store(data)

       # Load from shared memory with different layout
       dist_layout = ttgl.DistributedLayout(...)
       loaded = smem.load(dist_layout)

       # Create a subview by slicing
       subview = smem.slice(0, 64, dim=0)

       # Gather elements along an axis
       indices = ttgl.arange(0, 64, layout)
       gathered = smem.gather(indices, axis=0)
```

---

### triton.experimental.gluon.language.shared_memory_descriptor_type

```python
shared_memory_descriptor_type(element_ty, shape, layout, alloc_shape)
```

## shared_memory_descriptor_type


**`shared_memory_descriptor_type(element_ty, shape, layout, alloc_shape)`**

   Type descriptor for shared memory allocations in Gluon IR.

   Represents the type of a shared memory descriptor, specifying the element
   data type, logical shape, shared memory layout, and physical allocation
   shape. This type is used to statically describe shared memory regions for
   explicit GPU memory management.

   Parameters
   ----------
   element_ty : dtype
       The element data type stored in shared memory (e.g., `float16`,
       `int32`).
   shape : Sequence[int]
       The logical dimensions of the shared memory region.
   layout : SharedLayout
       The shared memory layout describing how elements are distributed
       across memory banks. Must be an instance of `SharedLayout`.
   alloc_shape : Sequence[int]
       The physical allocation shape, which may differ from the logical
       shape to accommodate padding or swizzling.

   Attributes
   ----------
   element_ty : dtype
       The element data type.
   shape : Tuple[int, ...]
       The logical shape of the shared memory region.
   layout : SharedLayout
       The shared memory layout.
   alloc_shape : Tuple[int, ...]
       The physical allocation shape.

   Notes
   -----
   The `layout` parameter must be a `SharedLayout` instance (e.g.,
   `BlockedLayout`, `CoalescedLayout`). The `alloc_shape` may be
   larger than `shape` to accommodate hardware requirements such as
   bank conflict avoidance or TMA swizzling.

   This type is typically constructed via `allocate_shared_memory` and
   used to type-check shared memory operations.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>> from triton.experimental.gluon.language._layouts import BlockedLayout
   >>>
   >>> @gluon.jit
   >>> def kernel():
   ...     layout = BlockedLayout([1, 32], [128, 128], [4, 1], [1, 1])
   ...     smem_ty = ttgl.shared_memory_descriptor_type(
   ...         element_ty=ttgl.float16,
   ...         shape=[128, 128],
   ...         layout=layout,
   ...         alloc_shape=[128, 128]
   ...     )
   ...     # Use smem_ty for type annotations or static checks

   See Also
   --------
   allocate_shared_memory : Allocate shared memory with given type.
   shared_memory_descriptor : Runtime handle to shared memory allocation.

---

### triton.experimental.gluon.language.sin

```python
sin(x, _semantic=None)
```

## sin


Computes the element-wise sine of a tensor.

**`sin(x, _semantic=None)`**

   Computes the sine of each element in the input tensor `x`.

   Parameters
   ----------
   x : Block
       The input tensor. Must have floating-point dtype (`fp32` or `fp64`).
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually.

   Returns
   -------
   Block
       A tensor containing the element-wise sine values. Has the same shape
       and dtype as `x`.

   Notes
   -----
   This function is only supported for `fp32` and `fp64` dtypes. Calling
   with other dtypes will raise an error.

   The operation is computed as $\sin(x)$ for each element.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, n: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offs = ttgl.arange(0, n)
       mask = offs < n
       x = ttgl.load(x_ptr + offs, mask=mask)
       y = ttgl.sin(x)
       ttgl.store(y_ptr + offs, y, mask=mask)
```

---

### triton.experimental.gluon.language.split

```python
split(a, _semantic=None, _generator=None) -> 'tuple[tensor, tensor]'
```

## split

**`split(a, _semantic=None, _generator=None)`**

Split a tensor in two along its last dimension.

### Parameters
a : tensor
    The tensor to split. The last dimension must have size 2.

### Returns
tuple[tensor, tensor]
    A tuple containing two tensors. If the input has rank 1, returns two scalars.

### Notes
The last dimension of the input tensor must have size 2. For example, given a
tensor of shape `(4, 8, 2)`, produces two tensors of shape `(4, 8)`. Given
a tensor of shape `(2,)`, returns two scalars.

To split into more than two pieces, use multiple calls to this function
(possibly combined with `reshape()`). This reflects the constraint in
Triton that tensors must have power-of-two sizes.

`split()` is the inverse of `join()`.

This function can also be called as a member function on `tensor`,
as `x.split()` instead of `split(x)`.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel():
     # Split a (4, 8, 2) tensor into two (4, 8) tensors
     x = ttgl.full((4, 8, 2), 1.0, ttgl.float32)
     left, right = ttgl.split(x)

     # Split a (2,) tensor into two scalars
     y = ttgl.full((2,), 2.0, ttgl.float32)
     a, b = ttgl.split(y)

     # Can also be called as a member function
     c, d = x.split()
```

---

### triton.experimental.gluon.language.sqrt

```python
sqrt(x, _semantic=None)
```

## sqrt

Computes the element-wise fast square root of a tensor.

### Parameters
x : Block
    The input tensor values. Must be floating point (fp32 or fp64).

### Returns
tensor
    A tensor containing the element-wise square root of `x`.

### Notes
This operation uses a fast square root approximation. Only floating point
dtypes (fp32, fp64) are supported. The result has the same dtype and shape
as the input.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
     pid = ttgl.program_id(0)
     offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
     x = ttgl.load(x_ptr + offs)
     y = ttgl.sqrt(x)
     ttgl.store(y_ptr + offs, y)
```

---

### triton.experimental.gluon.language.sqrt_rn

```python
sqrt_rn(x, _semantic=None)
```

## sqrt_rn


**`sqrt_rn(x, _semantic=None)`**

   Computes the element-wise precise square root (rounding to nearest per IEEE 754 standard) of `x`.

   Parameters
   ----------
   x : tensor
       The input tensor. Must have `fp32` dtype.
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not pass explicitly; this is handled automatically by the `@gluon.jit` decorator.

   Returns
   -------
   tensor
       A tensor containing the precise square root of each element in `x`. Same shape and dtype as input.

   Notes
   -----
   This operation uses the precise square root instruction (`sqrt.rn`) which rounds to nearest according to the IEEE 754 floating-point standard. Only `fp32` dtype is supported.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK_SIZE: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offsets = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(x_ptr + offsets)
       y = ttgl.sqrt_rn(x)
       ttgl.store(y_ptr + offsets, y)
```

---

### triton.experimental.gluon.language.static_assert

```python
static_assert(cond, msg='', _semantic=None)
```

## static_assert

Assert a condition at compile time.

### Parameters
cond : constexpr
    The condition to assert. Must be a compile-time constant expression.
msg : str, optional
    Error message to display if the assertion fails. Default is empty string.

### Returns
None

### Notes
This assertion is evaluated at compile time during kernel compilation, not at
runtime. It does not require the `TRITON_DEBUG` environment variable to
be set, unlike `device_assert()` which performs runtime assertions that
can be conditionally enabled.

Use `static_assert()` to validate compile-time constants such as block
sizes, tile dimensions, or other kernel configuration parameters. If the
condition evaluates to false, compilation will fail with an error message.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 BLOCK_SIZE: tl.constexpr = 1024

 @gluon.jit
 def kernel(...):
     ttgl.static_assert(BLOCK_SIZE == 1024)
     ttgl.static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be multiple of 32")
```

---

### triton.experimental.gluon.language.static_print

```python
static_print(*values, sep: 'str' = ' ', end: 'str' = '\n', file=None, flush=False, _semantic=None)
```

## static_print

Print values at compile time during kernel compilation.

```python
 triton.experimental.gluon.language.static_print(*values, sep=' ', end='\n', file=None, flush=False)

```
### Parameters

*values : tuple
    Values to print at compile time. Can be strings, constexpr values, or other
    compile-time constants.

sep : str, optional
    String inserted between values. Default is a space.

end : str, optional
    String appended after the last value. Default is a newline.

file : object, optional
    Ignored. Present for API compatibility with builtin `print`.

flush : bool, optional
    Ignored. Present for API compatibility with builtin `print`.

### Returns

None

### Notes

This function emits output during kernel compilation, not during kernel execution.
It is useful for debugging compile-time values such as `constexpr` constants,
block sizes, or other static parameters.

Calling the Python builtin `print` inside a Gluon kernel is not equivalent to
`static_print`. The builtin `print` maps to `device_print`, which
executes at runtime and has different argument requirements.

Use `static_print` for inspecting compile-time constants and `device_print`
for runtime tensor values.

### Examples

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(BLOCK_SIZE: ttgl.constexpr):
     ttgl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
     ttgl.static_print("Compilation complete", sep=" - ")
```

---

### triton.experimental.gluon.language.static_range

```python
static_range(arg1, arg2=None, step=None)
```

## static_range

**`static_range(arg1, arg2=None, step=None)`**

   Compile-time range iterator for loop unrolling in JIT-compiled kernels.

   `static_range` provides Python `range`-like semantics for use within
   `@gluon.jit` functions. All arguments must be compile-time constants
   (`constexpr`). The compiler uses this iterator to aggressively unroll
   loops, enabling better optimization opportunities.

   Parameters
   ----------
   arg1 : constexpr
      If `arg2` is None, this is the end value (start defaults to 0).
      If `arg2` is provided, this is the start value.
   arg2 : constexpr, optional
      The end value of the range (exclusive). Defaults to None.
   step : constexpr, optional
      The step value between iterations. Defaults to 1.

   Returns
   -------
   static_range
      An iterator object for use in for-loops within JIT-compiled functions.

   Notes
   -----
   All arguments must be `constexpr` values (compile-time constants).
   Runtime values will cause an assertion error.

   The iterator cannot be used outside of `@gluon.jit` decorated functions.
   Calling `__iter__` or `__next__` at runtime raises a RuntimeError.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(...):
       # Iterate from 0 to 9
       for i in ttgl.static_range(10):
           ...

       # Iterate from 2 to 8
       for i in ttgl.static_range(2, 8):
           ...

       # Iterate from 0 to 10 with step 2
       for i in ttgl.static_range(0, 10, 2):
           ...
```

---

### triton.experimental.gluon.language.store

```python
store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='', _semantic=None)
```

## store


**`store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='', _semantic=None)`**

   Store a tensor of data into memory locations defined by `pointer`.

   Parameters
   ----------
   pointer : PointerType or block of PointerType
       The memory location where the elements of `value` are stored. Can be a
       single element pointer, an N-dimensional tensor of pointers, or a block
       pointer defined by `make_block_ptr()`.
   value : Block
       The tensor of elements to be stored. Implicitly broadcast to
       `pointer.shape` and typecast to `pointer.dtype.element_ty`.
   mask : Block of int1, optional
       If `mask[idx]` is false, do not store `value[idx]` at `pointer[idx]`.
       Must be scalar if `pointer` is a single element pointer. Must be None
       if `pointer` is a block pointer.
   boundary_check : tuple of ints, optional
       Tuple of integers indicating the dimensions which should perform boundary
       checks. Only valid when `pointer` is a block pointer.
   cache_modifier : str, optional
       Changes cache option in NVIDIA PTX. Should be one of `""`, `".wb"`,
       `".cg"`, `".cs"`, `".wt"`, where `".wb"` stands for cache
       write-back all coherent levels, `".cg"` stands for cache global,
       `".cs"` stands for cache streaming, `".wt"` stands for cache
       write-through. See `cache operator
       <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_
       for more details.
   eviction_policy : str, optional
       Changes eviction policy in NVIDIA PTX. Should be one of `""`,
       `"evict_first"`, `"evict_last"`.

   Returns
   -------
   None

   Notes
   -----
   The behavior depends on the type of `pointer`:

   (1) If `pointer` is a single element pointer, a scalar is stored. In this
       case:

       - `mask` must also be scalar
       - `boundary_check` must be empty

   (2) If `pointer` is an N-dimensional tensor of pointers, an N-dimensional
       block is stored. In this case:

       - `mask` is implicitly broadcast to `pointer.shape`
       - `boundary_check` must be empty

   (3) If `pointer` is a block pointer defined by `make_block_ptr()`, a
       block of data is stored. In this case:

       - `mask` must be None
       - `boundary_check` can be specified to control out-of-bound access

   This function can also be called as a member function on `tensor`,
   as `x.store(...)` instead of `store(x, ...)`.

   Examples
   --------
   Store a scalar value to a single memory location:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr, val):
       ttgl.store(ptr, val)

Store a tensor to multiple memory locations with a mask:

.. code-block:: python

   @gluon.jit
   def kernel(ptrs, vals, mask):
       ttgl.store(ptrs, vals, mask=mask)

Store a block using a block pointer with boundary checking:

.. code-block:: python

   @gluon.jit
   def kernel(base_ptr, vals):
       block_ptr = ttgl.make_block_ptr(
           base_ptr,
           shape=(128, 128),
           strides=(128, 1),
           offsets=(0, 0),
           block_shape=(32, 32),
           order=(1, 0)
       )
       ttgl.store(block_ptr, vals, boundary_check=(0, 1))
```

---

### triton.experimental.gluon.language.sub

```python
sub(x, y, sanitize_overflow: 'constexpr' = True, _semantic=None)
```

**`sub(x, y, sanitize_overflow: 'constexpr' = True, _semantic=None)`**

   Element-wise subtraction of two arguments.

   Computes $x - y$ element-wise. Supports broadcasting according to standard
   Triton rules. This operation is available within Gluon JIT kernels.

   Parameters
   ----------
   x : tensor or scalar
      The left-hand operand.
   y : tensor or scalar
      The right-hand operand.
   sanitize_overflow : constexpr, optional
      If True (default), sanitizes overflow. Must be a compile-time constant.
   _semantic : GluonSemantic, optional
      Internal argument injected by the JIT compiler; do not set manually.

   Returns
   -------
   tensor
      The result of subtracting $y$ from $x$.

   Notes
   -----
   Broadcasting rules apply if `x` and `y` have different shapes.
   Overflow sanitization ensures defined behavior for integer subtraction
   when overflow occurs, depending on the backend implementation.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def sub_kernel(X_PTR, Y_PTR, OUT_PTR, BLOCK_SIZE: ttgl.constexpr):
       pid = ttgl.program_id(0)
       offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
       x = ttgl.load(X_PTR + offs)
       y = ttgl.load(Y_PTR + offs)
       # Element-wise subtraction
       out = ttgl.sub(x, y)
       ttgl.store(OUT_PTR + offs, out)
```

---

### triton.experimental.gluon.language.sum

```python
sum(input, axis=None, keep_dims=False, dtype: 'core.constexpr' = None)
```

## sum


**`sum(input, axis=None, keep_dims=False, dtype=None)`**

   Returns the sum of all elements in the `input` tensor along the provided `axis`.

   The reduction operation is associative and commutative.

   Parameters
   ----------
   input : Tensor
       The input tensor values to sum.
   axis : int, optional
       The dimension along which the reduction should be performed. If `None`,
       reduces all dimensions.
   keep_dims : bool, optional
       If `True`, keeps the reduced dimensions with length 1. Default is `False`.
   dtype : tl.dtype, optional
       The desired data type of the returned tensor. If specified, the input tensor is
       casted to `dtype` before the operation is performed. This is useful for
       preventing data overflows. If not specified, integer and bool dtypes are upcasted
       to `tl.int32` and float dtypes are upcasted to at least `tl.float32`.

   Returns
   -------
   Tensor
       A tensor containing the sum of the input elements along the specified axis.

   Notes
   -----
   This function can also be called as a member function on :py`tensor`,
   as `x.sum(...)` instead of `sum(x, ...)`.

   Default dtype promotion rules:
   - Signed integers with bitwidth < 32 are promoted to `tl.int32`
   - Unsigned integers with bitwidth < 32 are promoted to `tl.uint32`
   - Float types are promoted to at least `tl.float32`

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, out_ptr, N: ttgl.constexpr):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid * N)
       # Sum all elements in the tensor
       total = ttgl.sum(x)
       ttgl.store(out_ptr + pid, total)

   # Sum along a specific axis
   @gluon.jit
   def kernel_axis(x_ptr, out_ptr, M: ttgl.constexpr, N: ttgl.constexpr):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid * M * N)
       # Sum along axis 0, keeping dimensions
       result = ttgl.sum(x, axis=0, keep_dims=True)
       ttgl.store(out_ptr + pid * N, result)

   # Specify output dtype to prevent overflow
   @gluon.jit
   def kernel_dtype(x_ptr, out_ptr, N: ttgl.constexpr):
       pid = ttgl.program_id(0)
       x = ttgl.load(x_ptr + pid * N)
       # Cast to int32 before summing to avoid overflow
       total = ttgl.sum(x, dtype=ttl.int32)
       ttgl.store(out_ptr + pid, total)
```

---

### triton.experimental.gluon.language.tensor

```python
tensor(handle, type: 'dtype')
```

## class tensor


.. autoclass:: tensor

   Represents an N-dimensional array of values or pointers.

   `tensor` is the fundamental data structure in Gluon programs. Most functions
   in :py`triton.experimental.gluon.language` operate on and return tensors.

   Most named member functions are duplicates of free functions in
   :py`triton.experimental.gluon.language`. For example,
   `triton.experimental.gluon.language.sqrt(x)` is equivalent to `x.sqrt()`.

   `tensor` defines magic/dunder methods for arithmetic and comparison operations,
   enabling expressions like `x + y`, `x << 2`, `x == y`, etc.

   Parameters
   ----------
   handle : ir.value
       Internal IR handle representing the tensor value.
   type : dtype
       Tensor type, which may be a scalar dtype or block_type with shape information.

   Attributes
   ----------
   handle : ir.value
       Low-level IR handle for the tensor.
   shape : tuple[constexpr]
       Block shape of the tensor. Empty tuple for scalars.
   numel : constexpr
       Total number of elements in the tensor.
   type : dtype
       Tensor type (can be block_type).
   dtype : dtype
       Scalar element type of the tensor.

   Notes
   -----
   The `__init__` method is not called by user code. Tensors are typically created
   through Gluon operations like :py`arange()`, :py`full()`, :py`load()`,
   or :py`make_block_ptr()`.

   Member functions listed in the class body are type stubs. Actual implementations
   are added via the `_tensor_member_fn` decorator and forward to corresponding
   free functions.

   Supported operations include:

   - Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
   - Bitwise: `&`, `|`, `^`, `<<`, `>>`, `~`
   - Comparison: `<`, `<=`, `>`, `>=`, `==`, `!=`
   - Unary: `-`, `~`, `not`
   - Indexing: `__getitem__` (limited to expand_dims operations)

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, BLOCK: ttgl.constexpr):
       # Load tensor from global memory
       x = ttgl.load(x_ptr + ttgl.arange(0, BLOCK))

       # Arithmetic operations
       y = x * 2.0 + 1.0

       # Element-wise functions
       z = ttgl.sqrt(y)

       # Reduction
       sum_z = ttgl.sum(z, axis=0)

       # Store result
       ttgl.store(y_ptr + ttgl.arange(0, BLOCK), z)

.. code-block:: python

   # Tensor properties
   @gluon.jit
   def inspect_tensor(x: ttgl.tensor):
       # Access dtype and shape
       dtype = x.dtype  # e.g., ttgl.float32
       shape = x.shape  # e.g., (16, 32)
       numel = x.numel  # e.g., 512

       # Type casting
       x_fp16 = x.to(ttgl.float16)

       # Reshaping
       x_reshaped = x.reshape(32, 16)
```

---

### triton.experimental.gluon.language.to_linear_layout

```python
to_linear_layout(layout, shape, _semantic=None)
```

## to_linear_layout

**`to_linear_layout(layout, shape, _semantic=None)`**

    Convert a distributed layout to a linear layout representation.

    Parameters
    ----------
    layout : DistributedLayout
        The source layout to convert.
    shape : Sequence[int]
        The tensor shape associated with the layout.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Do not provide this argument directly.

    Returns
    -------
    LinearLayout
        The linear layout representation of the input layout.

    Notes
    -----
    This function is typically used internally by Gluon operations for low-level
    layout manipulation. Users should generally work with high-level layout
    objects rather than linear layouts directly.

    The linear layout representation provides explicit control over memory
    access patterns and is useful for advanced optimization scenarios.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> @gluon.jit
    ... def kernel():
    ...     layout = ttgl.AutoLayout()
    ...     shape = [128, 64]
    ...     linear = ttgl.to_linear_layout(layout, shape)
    ...     # Use linear layout for low-level operations
    ...     pass

---

### triton.experimental.gluon.language.to_tensor

```python
to_tensor(x, _semantic=None)
```

## to_tensor


**`to_tensor(x, _semantic=None)`**

    Convert a value to a Gluon tensor.

    Parameters
    ----------
    x : scalar or tensor
        The value to convert. Can be a Python scalar (int, float) or an existing tensor.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Automatically provided by the JIT compiler.

    Returns
    -------
    tensor
        A Gluon tensor containing the converted value.

    Notes
    -----
    This function is typically used internally by Gluon operations to ensure operands are
    in tensor form. Users generally do not need to call this directly within JIT-compiled
    kernels, as type conversion happens automatically.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(x_ptr):
         # Convert scalar to tensor
         scalar_val = 1.0
         tensor_val = ttgl.to_tensor(scalar_val)
```

---

### triton.experimental.gluon.language.tuple

```python
tuple(args: 'Sequence', type: 'Optional[tuple_type]' = None)
```

**`tuple(args, type=None)`**

   Container for multiple IR values in Triton Gluon kernels.

   Represents a heterogeneous collection of values that exist in the Triton IR,
   allowing kernels to return or manipulate multiple values as a single unit.
   Unlike Python tuples, Gluon tuples are first-class IR values with explicit
   typing.

   Parameters
   ----------
   args : Sequence
       Sequence of values to pack into the tuple. Each element must be a
       Triton IR value (e.g., `tensor`, `constexpr`, or another
       `tuple`).
   type : tuple_type, optional
       Explicit tuple type specification. If provided, must be a
       `tuple_type` instance. If `None`, the type is inferred from
       `args`.

   Attributes
   ----------
   values : list
       The underlying IR values stored in the tuple.
   type : tuple_type
       The static type of the tuple, including element types and optional
       field names.

   Methods
   -------
   __getitem__(idx)
       Access elements by integer index or slice. Returns a single value for
       integer indices, or a new `tuple` for slices.
   __getattr__(name)
       Access elements by field name if the tuple was created with named
       fields. Raises :exc:`AttributeError` if the field does not exist.
   __len__()
       Return the number of elements in the tuple.
   __iter__()
       Iterate over the tuple elements.

   Notes
   -----
   Gluon tuples differ from Python tuples in several ways:

   - Elements are Triton IR values, not Python objects.
   - Tuples can have named fields for structured access.
   - Tuple operations (concatenation, multiplication) are available but
     operate on IR values.
   - Tuples are flattened into IR handles during code generation.

   Named field access requires the tuple to be constructed with a
   `tuple_type` that specifies field names.

   Examples
   --------
   Create a tuple from multiple tensor values:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(...):
       a = ttgl.load(ptr1)
       b = ttgl.load(ptr2)
       t = ttgl.tuple([a, b])
       # Access by index
       first = t[0]
       second = t[1]

Create a tuple with named fields:

.. code-block:: python

   @gluon.jit
   def kernel(...):
       from triton.language import tuple_type
       ty = tuple_type([ttl.float32, ttl.float32], fields=["x", "y"])
       t = ttgl.tuple([val_x, val_y], type=ty)
       # Access by field name
       x = t.x
       y = t.y

Return multiple values from a kernel helper function:

.. code-block:: python

   @gluon.jit
   def compute_pair(a, b):
       sum_val = a + b
       prod_val = a * b
       return ttgl.tuple([sum_val, prod_val])

   @gluon.jit
   def kernel(...):
       result = compute_pair(x, y)
       s = result[0]
       p = result[1]

Concatenate tuples:

.. code-block:: python

   @gluon.jit
   def kernel(...):
       t1 = ttgl.tuple([a, b])
       t2 = ttgl.tuple([c, d])
       t3 = t1 + t2  # Four elements: a, b, c, d
```

---

### triton.experimental.gluon.language.tuple_type

```python
tuple_type(types, fields=None)
```

## tuple_type


**`tuple_type(types, fields=None)`**

   Composite type representing an ordered collection of element types.

   `tuple_type` defines a structured type composed of multiple constituent
   types, optionally with named fields. It is used to represent tuple values
   in Triton IR, enabling structured data passing between kernel operations
   and low-level hardware features like TMA descriptors and warpgroup MMA.

   Parameters
   ----------
   types : list of dtype
       List of element types comprising the tuple. Each element must be a
       valid Triton type (e.g., `tl.int32`, `tl.float16`, `pointer_type`).
   fields : list of str, optional
       Optional field names for each element. If provided, `len(fields)`
       must equal `len(types)`. Enables attribute-style access via
       `tuple.field_name`.

   Attributes
   ----------
   types : list of dtype
       The constituent types of the tuple.
   fields : list of str or None
       Field names if provided at construction, otherwise `None`.
   name : str
       String representation of the tuple type (cached property).

   Methods
   -------
   __getitem__(index)
       Access element type by integer index.
   __iter__()
       Iterate over constituent types.
   __eq__(other)
       Compare equality with another `tuple_type`.
   mangle()
       Return mangled type name for kernel compilation.
   _flatten_ir_types(builder, out)
       Flatten types into IR type list.
   _unflatten_ir(handles, cursor)
       Reconstruct tuple value from IR handles.

   Notes
   -----
   `tuple_type` is a low-level type constructor used primarily in Gluon
   kernels for explicit memory layout control. It supports both positional
   tuples (anonymous fields) and named tuples (struct-like access).

   The mangled name format is `T<type1_mangle>_<type2_mangle>...T` for
   kernel compilation and caching.

   Tuple types are immutable after construction. Equality requires matching
   types, fields, and order.

   Examples
   --------
   Create a simple tuple type with two elements:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # Positional tuple: (int32, float16)
   simple_tuple = ttgl.tuple_type([ttlgl.int32, ttlgl.float16])
   print(simple_tuple.name)  # [int32,float16]

Create a named tuple type for structured access:

.. code-block:: python

   # Named tuple: {x: int32, y: float16}
   named_tuple = ttgl.tuple_type([ttlgl.int32, ttlgl.float16], fields=['x', 'y'])
   print(named_tuple.name)  # [x:int32,y:float16]

Access element types by index:

.. code-block:: python

   first_type = simple_tuple[0]  # int32
   second_type = simple_tuple[1]  # float16

Use in kernel type annotations:

.. code-block:: python

   @gluon.jit
   def kernel(ptr: ttgl.pointer_type, desc: ttgl.tuple_type([...])):
       # Kernel implementation
       pass

See Also
--------
tensor_descriptor_type : Tensor descriptor with shape and stride tuples
aggregate : User-defined aggregate types with field annotations
```

---

### triton.experimental.gluon.language.uint16

.. py:data:: ttgl.uint16

   16-bit unsigned integer data type.

   Represents an unsigned 16-bit integer scalar type in Gluon. This instance is used to specify element types for tensors, shared memory, and kernel arguments within device code.

   Examples
   --------
   Specify `uint16` as the element type for a pointer argument:

```python
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl

    @gluon.jit
    def kernel(ptr: ttgl.pointer[ttgl.uint16], size: int):
        pid = ttgl.program_id(0)
        mask = pid < size
        val = ttgl.load(ptr + pid, mask=mask)
        # val dtype is ttgl.uint16

Notes
-----
Corresponds to ``uint16`` in CUDA and ``uint16_t`` in C++. Valid values range from 0 to 65535. Arithmetic operations follow standard modular semantics.
```

---

### triton.experimental.gluon.language.uint32

.. py:data:: triton.experimental.gluon.language.uint32

    32-bit unsigned integer data type.

    This instance represents the unsigned 32-bit integer type within the Gluon type system. It is used to define tensor element types, kernel argument signatures, and shared memory allocations in `@gluon.jit` decorated functions.

    Notes
    -----
    `uint32` adheres to standard IEEE 754 integer semantics for unsigned operations. It is commonly required for indexing, pointer arithmetic, and interacting with hardware features such as TMA descriptors or barrier indices on NVIDIA Hopper/Blackwell and AMD CDNA/RDNA architectures.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> @gluon.jit
    ... def kernel(ptr: ttgl.pointer_type[ttgl.uint32]):
    ...     val = ttgl.load(ptr)
    ...     # val has type uint32
    ...     pass

---

### triton.experimental.gluon.language.uint64

.. py:data:: triton.experimental.gluon.language.uint64

   64-bit unsigned integer data type.

   Represents an unsigned 64-bit integer type within Gluon kernels. This dtype instance is used to annotate scalar values or tensor elements requiring 64-bit precision without sign bits.

   Notes
   -----
   Supported across all Gluon backends (NVIDIA Hopper/Blackwell, AMD CDNA3/RDNA3). Commonly used for global pointer offsets, large counters, or bit-wise operations requiring 64-bit width. Use `ttgl.int64` for signed arithmetic or `ttgl.uint32` to reduce register pressure when 64-bit range is unnecessary.

   Examples
   --------
   Define a kernel accepting a 64-bit unsigned integer argument:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def process_count(count: ttgl.uint64):
       # Perform 64-bit unsigned arithmetic
       next_count = count + 1
       pass
```

---

### triton.experimental.gluon.language.uint8

.. py:data:: triton.experimental.gluon.language.uint8

   8-bit unsigned integer data type.

   Represents an unsigned 8-bit integer type for use in Gluon kernels. This dtype is used to specify the element type of tensors and blocks within kernel definitions.

   Notes
   -----
   Values range from 0 to 255. Overflow behavior follows standard unsigned integer wrapping.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr):
       val = ttgl.load(ptr).to(ttgl.uint8)
       ttgl.store(ptr, val)
```

---

### triton.experimental.gluon.language.umulhi

```python
umulhi(x, y, _semantic=None)
```

## umulhi


**`umulhi(x, y, _semantic=None)`**

   Computes the element-wise most significant N bits of the 2N-bit product of `x` and `y`.

   For integer types with N bits, this returns the upper N bits of the full 2N-bit multiplication result. This is useful for high-precision multiplication where the full product exceeds the operand width.

   Parameters
   ----------
   x : Block
       The first input tensor. Must be int32, int64, uint32, or uint64.
   y : Block
       The second input tensor. Must have the same dtype as `x`.
   _semantic : GluonSemantic, optional
       Internal semantic context (automatically provided by JIT compiler).

   Returns
   -------
   tensor
       A tensor containing the most significant N bits of the 2N-bit product. Has the same dtype and shape as the inputs.

   Notes
   -----
   Supported dtypes: int32, int64, uint32, uint64. Both inputs must have matching dtypes.

   For unsigned 32-bit inputs, this computes bits [32:63] of the 64-bit product. For signed inputs, the result preserves the sign extension behavior of the full multiplication.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>> @gluon.jit
   ... def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
   ...     pid = ttgl.program_id(0)
   ...     offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
   ...     x = ttgl.load(x_ptr + offs)
   ...     y = ttgl.load(y_ptr + offs)
   ...     hi = ttgl.umulhi(x, y)
   ...     ttgl.store(out_ptr + offs, hi)

---

### triton.experimental.gluon.language.void

.. py:data:: triton.experimental.gluon.language.void

    Represents the void data type.

    Used to specify the absence of a return value or an opaque memory type within
    Gluon kernels. This instance is a singleton in the Gluon type system and aligns
    with Triton's intermediate representation for non-returning functions.

    Notes
    -----
    Accessible via the `ttgl` namespace alias (`ttgl.void`). Typically used
    internally by the compiler to represent kernels without return values. Explicit
    usage is rare in user code but available for type introspection.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> ttgl.void
    void

---

### triton.experimental.gluon.language.warp_specialize

```python
warp_specialize(functions_and_args, worker_num_warps, worker_num_regs=None, _semantic=None, _generator=None)
```

## warp_specialize

Create a warp-specialized execution region that partitions work across warps.

Forks the current execution into a "default partition" and arbitrary number of
"worker partitions". The default partition executes in the same `num_warps` as
the parent region and may accept tensor arguments and return tensors. Worker
partitions execute in additional warps that sit idle during parent region
execution.

### Parameters
functions_and_args : List[Tuple[Callable, Any]]
    List of functions and arguments for each partition. The first tuple specifies
    the default partition function and its arguments.
worker_num_warps : List[int]
    Number of warps allocated to each worker partition.
worker_num_regs : List[int], optional
    Number of registers for each worker partition. If provided, used by the
    backend for dynamic register reallocation.

### Returns
Tuple[Any, ...]
    Results returned from the default partition.

### Notes
Recursive calls to `warp_specialize` are not supported. Worker partitions
execute concurrently with the default partition, enabling overlapping of
compute and memory operations across warp groups.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def worker_kernel(x, y):
     # Worker partition logic
     return x + y

 @gluon.jit
 def default_kernel(x):
     # Default partition logic
     return x * 2

 @gluon.jit
 def kernel(ptrs):
     # Warp-specialize: default runs in main warps, worker runs in additional warps
     result = ttgl.warp_specialize(
         functions_and_args=[
             (default_kernel, [ptrs]),
             (worker_kernel, [ptrs, ptrs]),
         ],
         worker_num_warps=[4],  # 4 warps for worker partition
     )
     return result
```

---

### triton.experimental.gluon.language.where

```python
where(condition, x, y, _semantic=None)
```

## ttgl.where


**`where(condition, x, y, _semantic=None)`**

   Select elements from `x` or `y` based on `condition`.

   Parameters
   ----------
   condition : tensor
       Boolean tensor. When True (nonzero), yield `x`, otherwise yield `y`.
   x : tensor
       Values selected at indices where `condition` is True.
   y : tensor
       Values selected at indices where `condition` is False.

   Returns
   -------
   tensor
       A tensor with elements from `x` or `y` selected by `condition`.

   Notes
   -----
   Both `x` and `y` are always evaluated regardless of the value of
   `condition`. If you want to avoid unintended memory operations, use the
   `mask` arguments in `ttgl.load()` and `ttgl.store()` instead.

   The shapes of `x` and `y` are both broadcast to the shape of
   `condition`. `x` and `y` must have the same data type.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
       pid = ttgl.program_id(0)
       offs = ttgl.arange(0, BLOCK_SIZE)
       mask = offs < BLOCK_SIZE
       x = ttgl.load(x_ptr + offs, mask=mask)
       y = ttgl.load(y_ptr + offs, mask=mask)
       # Select x where offs is even, y where offs is odd
       condition = (offs % 2) == 0
       result = ttgl.where(condition, x, y)
       ttgl.store(out_ptr + offs, result, mask=mask)
```

---

### triton.experimental.gluon.language.xor_sum

```python
xor_sum(input, axis=None, keep_dims=False)
```

## xor_sum

Compute the XOR sum of tensor elements along specified axes.

```python
 xor_sum(input, axis=None, keep_dims=False)

```
Returns the bitwise XOR sum of all elements in the `input` tensor along the
provided `axis`. The reduction operation is associative and commutative.

### Parameters
input : Tensor
    The input tensor. Must have integer dtype.
axis : int, optional
    The dimension along which the reduction should be performed. If `None`,
    reduce across all dimensions.
keep_dims : bool, optional
    If `True`, keep the reduced dimensions with length 1. Default is
    `False`.

### Returns
Tensor
    A tensor containing the XOR sum of the input elements. The output dtype
    matches the input dtype.

### Notes
This function only supports integer tensors. Calling `xor_sum` on
floating-point tensors will raise a compilation error.

The function can be called as a member function on :py`tensor`, as
`x.xor_sum(...)` instead of `xor_sum(x, ...)`.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(x_ptr, out_ptr, BLOCK_SIZE: ttgl.constexpr):
     x = ttgl.load(x_ptr + ttgl.arange(0, BLOCK_SIZE))
     # Compute XOR sum across all elements
     result = ttgl.xor_sum(x)
     ttgl.store(out_ptr, result)

 # Using keep_dims to preserve shape
 @gluon.jit
 def kernel_2d(x_ptr, out_ptr, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr):
     x = ttgl.load(x_ptr + ttgl.arange(0, BLOCK_M)[:, None] * BLOCK_N + ttgl.arange(0, BLOCK_N)[None, :])
     # Reduce along axis 1, keeping the dimension
     result = ttgl.xor_sum(x, axis=1, keep_dims=True)
     ttgl.store(out_ptr, result)

 # Using member function syntax
 @gluon.jit
 def kernel_member(x_ptr, out_ptr, BLOCK_SIZE: ttgl.constexpr):
     x = ttgl.load(x_ptr + ttgl.arange(0, BLOCK_SIZE))
     result = x.xor_sum()
     ttgl.store(out_ptr, result)
```

---

### triton.experimental.gluon.language.zeros

```python
zeros(shape, dtype, layout=None)
```

**`triton.experimental.gluon.language.zeros(shape, dtype, layout=None)`**

   Create a tensor filled with zeros.

   Parameters
   ----------
   shape : Sequence[int]
       The shape of the tensor.
   dtype : dtype
       The data type for the tensor.
   layout : DistributedLayout, optional
       The distributed layout of the tensor. Defaults to automatic layout selection.

   Returns
   -------
   tensor
       A tensor where every element is zero.

   Examples
   --------
```python
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl

    @gluon.jit
    def kernel():
        z = ttgl.zeros((128, 128), ttgl.float16)
```

---

### triton.experimental.gluon.language.zeros_like

```python
zeros_like(input, shape=None, dtype=None, layout=None)
```

**`zeros_like(input, shape=None, dtype=None, layout=None)`**

   Create a tensor with the same properties as a given tensor, filled with zeros.

   Parameters
   ----------
   input : ttgl.tensor
       Reference tensor to infer default shape, dtype, and layout.
   shape : Sequence[int], optional
       Target shape. Defaults to `input.shape`.
   dtype : ttgl.dtype, optional
       Target data type. Defaults to `input.dtype`.
   layout : ttgl.DistributedLayout, optional
       Target layout. Defaults to `input.layout`.

   Returns
   -------
   ttgl.tensor
       A tensor where every element is zero.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(x):
       # Create a zero tensor matching x's properties
       z = ttgl.zeros_like(x)
```

---


## triton.experimental.gluon.language.amd

### triton.experimental.gluon.language.amd.AMDMFMALayout

```python
AMDMFMALayout(version: 'int', instr_shape: 'List[int]', transposed: 'bool', warps_per_cta: 'List[int]', element_bitwidth: 'Optional[int]' = None, tiles_per_warp: 'Optional[List[int]]' = None, cga_layout: 'List[List[int]]' = <factory>) -> None
```

**`AMDMFMALayout(version: 'int', instr_shape: 'List[int]', transposed: 'bool', warps_per_cta: 'List[int]', element_bitwidth: 'Optional[int]' = None, tiles_per_warp: 'Optional[List[int]]' = None, cga_layout: 'List[List[int]]' = <factory>)`**

   Represents a layout for AMD MFMA (matrix core) operations.

   Parameters
   ----------
   version : int
       The GPU architecture version. Supported values are 1 to 4.
   instr_shape : List[int]
       The shape in the form of `(M, N, K)` of the matrix. Valid `(M, N)` pairs are `[32, 32]`, `[16, 16]`, `[64, 4]`, and `[4, 64]`.
   transposed : bool
       Indicates whether the result tensor is transposed. If `True`, each thread holds consecutive elements in the same row instead of column, optimizing chained dot products and global writes.
   warps_per_cta : List[int]
       The warp layout in the block. Determines the rank of the layout.
   element_bitwidth : Optional[int], optional
       Bit width of the output element type. Supported values are 32 and 64. Defaults to 32.
   tiles_per_warp : Optional[List[int]], optional
       The tile layout within a warp. Defaults to unit tile layout (i.e., single tile on all dimensions).
   cga_layout : List[List[int]], optional
       Bases describing CTA tiling. Defaults to an empty list.

   Notes
   -----
   Current supported `version` values correspond to the following architectures:

   - 1: gfx908
   - 2: gfx90a
   - 3: gfx942
   - 4: gfx950

   The `rank` property is derived from the length of `warps_per_cta`.

   Examples
   --------
```python
   import triton.experimental.gluon.language as ttgl

   # Create a layout for gfx942 (version 3) with 16x16x16 matrix shape
   layout = ttgl.amd.AMDMFMALayout(
       version=3,
       instr_shape=[16, 16, 16],
       transposed=False,
       warps_per_cta=[4, 2],
       element_bitwidth=32
   )
```

---

### triton.experimental.gluon.language.amd.AMDWMMALayout

```python
AMDWMMALayout(version: 'int', transposed: 'bool', warp_bases: 'List[List[int]]', reg_bases: 'Optional[List[List[int]]]' = None, instr_shape: 'Optional[List[int]]' = None, cga_layout: 'List[List[int]]' = <factory>, rank: 'Optional[int]' = None) -> None
```

## class triton.experimental.gluon.language.amd.AMDWMMALayout


.. autoclass:: AMDWMMALayout

   Represents a layout for AMD WMMA (Wave Matrix Multiply-Accumulate) matrix core operations.

   Parameters
   ----------
   version : int
       GPU architecture version indicator. Supported values are 1-3.
   transposed : bool
       Whether the result tensor is transposed.
   warp_bases : List[List[int]]
       Warp bases for CTA (Cooperative Thread Array) layout. Each inner list represents a basis vector.
   reg_bases : List[List[int]], optional
       Repetition (register) bases for CTA layout. Defaults to empty list if not provided.
   instr_shape : List[int], optional
       Instruction shape in (M, N, K) format. Defaults to `[16, 16, 16]` if not provided.
   cga_layout : List[List[int]], optional
       Bases describing CTA tiling. Defaults to empty list.
   rank : int, optional
       Rank of warp and register bases. Defaults to 2 if not provided.

   Notes
   -----
   Current supported versions:

   - **1**: RDNA3 architecture (e.g., gfx1100, gfx1101)
   - **2**: RDNA4 architecture (e.g., gfx1200, gfx1201)
   - **3**: gfx1250 architecture

   The layout defines how matrix fragments are distributed across warps and registers for WMMA operations on AMD GPUs. The `warp_bases` and `reg_bases` parameters control the distribution pattern, while `instr_shape` specifies the matrix tile dimensions.

   Examples
   --------
   Create a WMMA layout for RDNA3 (version 1) with default instruction shape:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.amd import AMDWMMALayout

   # Create a 2D WMMA layout for RDNA3
   layout = AMDWMMALayout(
       version=1,
       transposed=False,
       warp_bases=[[1, 0], [0, 1]],
       reg_bases=[[2, 0], [0, 2]],
       instr_shape=[16, 16, 16],
       rank=2
   )

Create a transposed layout for RDNA4 (version 2):

.. code-block:: python

   layout = AMDWMMALayout(
       version=2,
       transposed=True,
       warp_bases=[[1, 0], [0, 1]],
       rank=2
   )
```

---

### triton.experimental.gluon.language.amd.warp_pipeline_stage

```python
warp_pipeline_stage(label=None, *, priority: 'int | None' = None, **_internal)
```

**`warp_pipeline_stage(label=None, *, priority=None, **_internal)`**

    Marks a warp-pipeline stage inside a Gluon kernel.

    Parameters
    ----------
    label : str, optional
        Name of the pipeline stage (e.g., "load", "compute"). Used for identification
        and diagnostics without affecting program semantics.
    priority : int, optional
        Relative scheduling priority of the warp. Valid values range from 0 (lowest)
        to 3 (highest). Corresponds to the operand of `s_setprio`. If unspecified,
        priority resets to zero when any other stage in the loop uses explicit
        priority.

    Notes
    -----
    This class is used as a context manager. Each block semantically defines a distinct
    stage of a warp pipeline. All operations inside the block belong to the same
    pipeline cluster and are intended to execute as a unit relative to other stages.

    Priority is a performance hint to the hardware scheduler. Its effect may vary
    depending on the dynamic interaction of instruction streams across different warps.
    It should be used judiciously, only when explicit scheduling guidance is beneficial.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(...):
         for k in ttgl.range(0, K, 1):
             # Stage 0: prefetch tiles
             with ttgl.amd.warp_pipeline_stage("load", priority=3):
                 a = ttgl.amd.buffer_load(a_ptr, offs_a)
                 b = ttgl.amd.buffer_load(b_ptr, offs_b)

             # Stage 1: prepare MFMA operands
             with ttgl.amd.warp_pipeline_stage("prep"):
                 a_tile = a.load(layout=...)
                 b_tile = b.load(layout=...)

             # Stage 2: compute
             with ttgl.amd.warp_pipeline_stage("compute", priority=0):
                 acc = ttgl.amd.mfma(a_tile, b_tile, acc)
                 offs_a += strideA
                 offs_b += strideB
```

---


## triton.experimental.gluon.language.amd.cdna3

### triton.experimental.gluon.language.amd.cdna3.buffer_atomic_add

```python
buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_add


.. autofunction:: buffer_atomic_add

Perform an atomic add operation on buffer memory for AMD CDNA3 GPUs.

This function atomically adds a value to elements in buffer memory at specified offsets.
It is optimized for AMD CDNA3 architecture and uses buffer memory operations.

### Parameters
ptr : pointer
    Pointer to the buffer memory region.
offsets : tensor
    Tensor of offsets specifying which locations to update.
value : tensor
    Tensor of values to add at each offset location.
mask : tensor, optional
    Boolean tensor controlling which offsets are active. If provided, only
    offsets where mask is True will be updated.
sem : str, optional
    Memory semantics for the atomic operation. Common options include
    `'acquire'`, `'release'`, `'acq_rel'`, or `'relaxed'`.
    Defaults to relaxed semantics.
scope : str, optional
    Memory scope for the atomic operation. Options include `'cta'`,
    `'cluster'`, or `'gpu'`. Defaults to CTA scope.
_semantic : GluonSemantic
    Internal semantic handler. Do not set manually.

### Returns
tensor
    Tensor containing the values at each location before the atomic add
    was performed (original values).

### Notes
This operation is only available on AMD CDNA3 GPUs. The buffer memory
must be properly allocated and accessible by the GPU.

The atomic operation ensures that concurrent accesses from multiple
threads are handled correctly without race conditions.

For best performance, ensure offsets are coalesced and aligned.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.amd import cdna3

 @gluon.jit
 def kernel(ptr, offsets, value):
     # Perform atomic add on buffer memory
     old_values = cdna3.buffer_atomic_add(ptr, offsets, value)
     # old_values contains the values before the add
```

---

### triton.experimental.gluon.language.amd.cdna3.buffer_atomic_and

```python
buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_and


**`buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)`**

    Perform an atomic bitwise AND operation on AMD CDNA3 buffer memory.

    Executes `memory = memory & value` atomically using AMD CDNA3 buffer
    atomic instructions. Returns the original value from memory before the
    operation.

    Parameters
    ----------
    ptr : pointer
        Pointer to the buffer memory location.
    offsets : tensor
        Offsets into the buffer memory.
    value : tensor
        Value to AND with the memory location.
    mask : tensor, optional
        Boolean mask to control which elements participate in the atomic
        operation.
    sem : str, optional
        Memory semantics. One of `'acquire'`, `'release'`, `'acq_rel'`,
        or `'relaxed'`.
    scope : str, optional
        Memory scope. One of `'cta'`, `'cluster'`, or `'gpu'`.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Do not set manually.

    Returns
    -------
    tensor
        The original value at the memory location before the atomic operation.

    Notes
    -----
    This function is specific to AMD CDNA3 architecture and requires buffer
    memory operations. The atomic operation is performed element-wise for
    tensor inputs.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl
     from triton.experimental.gluon.language.amd import cdna3

     @gluon.jit
     def kernel(ptr, offsets, value):
         old_value = cdna3.buffer_atomic_and(ptr, offsets, value)
```

---

### triton.experimental.gluon.language.amd.cdna3.buffer_atomic_max

```python
buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_max


**`buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)`**

    Perform an atomic maximum operation on AMD CDNA3 GPU buffer memory.

    Atomically computes `max(*ptr[offsets], value)` and stores the result back to
    memory. This operation is specific to AMD CDNA3 architecture and uses buffer
    atomic instructions.

    Parameters
    ----------
    ptr : pointer
        Pointer to the buffer memory location(s) to operate on.
    offsets : tensor
        Integer tensor specifying byte offsets from the base pointer.
    value : tensor
        Value(s) to compare atomically with memory contents.
    mask : tensor, optional
        Boolean tensor indicating which offsets to operate on. If None, all
        offsets are processed.
    sem : str, optional
        Memory semantics for the atomic operation (e.g., `"acquire"`,
        `"release"`, `"acq_rel"`). Defaults to hardware default.
    scope : str, optional
        Scope of the atomic operation (e.g., `"cta"`, `"cluster"`, `"gpu"`).
        Defaults to CTA scope.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Must be provided by the JIT compiler.

    Returns
    -------
    tensor
        The original values loaded from memory before the atomic operation.

    Notes
    -----
    This function is only available on AMD CDNA3 GPUs (MI300 series). It requires
    the kernel to be decorated with `@gluon.jit` rather than `@triton.jit`.

    The operation is equivalent to:
    `old_value = atomicMax(ptr[offset], value)`

    Buffer atomics on CDNA3 provide improved performance over regular global
    memory atomics for certain access patterns.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl
     from triton.experimental.gluon.language.amd import cdna3

     @gluon.jit
     def kernel(ptr, offsets, value):
         # Perform atomic max on CDNA3 buffer memory
         old_vals = cdna3.buffer_atomic_max(ptr, offsets, value)
         return old_vals
```

---

### triton.experimental.gluon.language.amd.cdna3.buffer_atomic_min

```python
buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_min


.. autofunction:: buffer_atomic_min

Perform an atomic minimum operation on AMD CDNA3 buffer memory.

```python
 buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)

```
Atomically computes `min(*ptr[off], value)` for each offset and stores the result back to memory. Returns the original values at the memory locations before the update.

### Parameters
ptr : pointer_type
    Pointer to the global memory buffer.
offsets : tensor
    Tensor of offsets specifying which locations to access.
value : tensor
    Tensor of values to compare against memory contents.
mask : tensor, optional
    Boolean tensor controlling which offsets are active. If `None`, all offsets are active.
sem : str, optional
    Memory semantics for the atomic operation (e.g., `"acquire"`, `"release"`, `"acq_rel"`).
scope : str, optional
    Memory scope for the atomic operation (e.g., `"cta"`, `"cluster"`, `"gpu"`).
_semantic : GluonSemantic
    Internal semantic handler. Do not pass directly; set via `@gluon.jit` decorator.

### Returns
tensor
    Tensor containing the original values from memory before the atomic minimum update.

### Notes
This operation is specific to AMD CDNA3 architecture and uses the buffer atomic instruction
set. The atomic minimum compares each `value` with the corresponding memory location and
stores the smaller of the two.

The operation is only valid within kernels decorated with `@gluon.jit`. The `_semantic`
parameter is automatically provided by the JIT compiler and should not be passed manually.

Memory semantics (`sem`) and scope (`scope`) control synchronization behavior across
threads and memory hierarchies. Consult AMD CDNA3 documentation for valid options.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def atomic_min_kernel(ptr, offsets, values, n_elements, BLOCK_SIZE: tl.constexpr):
     pid = ttgl.program_id(0)
     offs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
     mask = offs < n_elements
     old_vals = ttgl.amd.cdna3.buffer_atomic_min(ptr, offs, values, mask=mask)
```

---

### triton.experimental.gluon.language.amd.cdna3.buffer_atomic_or

```python
buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_or


**`buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)`**

    Perform an atomic bitwise OR operation on buffer memory for AMD CDNA3 GPUs.

    This function atomically reads a value from buffer memory, performs a bitwise OR
    with the provided value, and writes the result back. Returns the original value
    from memory before the operation.

    Parameters
    ----------
    ptr : tensor
        Pointer tensor to the buffer memory location(s).
    offsets : tensor
        Tensor of byte offsets from the base pointer.
    value : tensor
        Tensor of values to OR with the memory contents.
    mask : tensor, optional
        Boolean tensor controlling which elements are updated. If None, all elements
        are processed.
    sem : str, optional
        Memory semantics for the atomic operation. Options include `"acquire"`,
        `"release"`, `"acq_rel"`, or `"relaxed"`. Defaults to relaxed semantics.
    scope : str, optional
        Scope of the atomic operation. Options include `"cta"`, `"cluster"`, or
        `"gpu"`. Defaults to CTA scope.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Must be provided by the JIT compiler; do not set
        manually.

    Returns
    -------
    tensor
        Tensor containing the original values from memory before the OR operation.

    Notes
    -----
    This function is specific to AMD CDNA3 architecture and uses buffer atomic
    instructions. The operation is equivalent to:

```python
     original = memory[addr]
     memory[addr] = original | value
     return original

 For proper synchronization in multi-threaded contexts, consider setting
 appropriate ``sem`` and ``scope`` parameters.

 This function must be called from a kernel decorated with ``@gluon.jit``.

 Examples
 --------
 .. code-block:: python

     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl
     from triton.experimental.gluon.language.amd import cdna3

     @gluon.jit
     def kernel(ptr, offsets, value):
         result = cdna3.buffer_atomic_or(ptr, offsets, value)
         # result contains original memory values before OR
```

---

### triton.experimental.gluon.language.amd.cdna3.buffer_atomic_xor

```python
buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```



---

### triton.experimental.gluon.language.amd.cdna3.buffer_load

```python
buffer_load(ptr, offsets, mask=None, other=None, cache=None, _semantic=None)
```

AMD CDNA3 buffer load from global memory using a scalar base pointer and offsets tensor.

This operation loads data directly from global memory into registers using AMD's
buffer load instruction, which takes a scalar base pointer and a tensor of
offsets rather than a tensor of pointers.

### Parameters
ptr : pointer to scalar
    Global memory scalar base pointer to load from.
offsets : tensor
    Tensor of offsets specifying memory locations relative to `ptr`.
mask : tensor, optional
    Boolean mask tensor for predicated loads. Elements where mask is False
    will not be loaded. Defaults to None.
other : tensor or scalar, optional
    Default values for masked elements. Used when `mask` is provided.
    Defaults to None.
cache : str, optional
    Cache modifier specifier for load operation (e.g., "ca", "cg").
    Defaults to None (no cache modifier).

### Returns
tensor
    A Gluon tensor containing the loaded data with the same shape as
    `offsets` and element type matching `ptr`.

### Notes
This function is specific to AMD CDNA3 architecture and uses the buffer
load instruction for efficient memory access patterns. The scalar base
pointer approach can be more efficient than pointer tensors for certain
access patterns.

The `mask` and `other` parameters enable predicated loads, allowing
conditional memory access without branching.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(ptr, offsets):
     # Load from global memory using scalar pointer + offsets
     data = ttgl.amd.cdna3.buffer_load(ptr, offsets)

 @gluon.jit
 def kernel_masked(ptr, offsets, mask):
     # Predicated load with default value for masked elements
     data = ttgl.amd.cdna3.buffer_load(ptr, offsets, mask=mask, other=0.0)
```

---

### triton.experimental.gluon.language.amd.cdna3.buffer_store

```python
buffer_store(stored_value, ptr, offsets, mask=None, cache=None, _semantic: 'GluonSemantic' = None)
```

**`triton.experimental.gluon.language.amd.cdna3.buffer_store(stored_value, ptr, offsets, mask=None, cache=None)`**

    Store a tensor to global memory via a scalar base pointer and offsets tensor on AMD CDNA3 GPUs.

    Performs a buffer store operation using a scalar pointer and a tensor of offsets, rather than a tensor of pointers. This is specific to AMD CDNA3 architecture and enables efficient memory access patterns.

    Parameters
    ----------
    stored_value : tensor
        The tensor data to be stored to global memory.
    ptr : pointer
        Scalar base pointer to global memory.
    offsets : tensor
        Tensor of offsets added to `ptr` for address calculation.
    mask : tensor, optional
        Predication mask. Elements are stored only where `mask` is true.
    cache : str, optional
        Cache modifier specifier (e.g., `.cg`, `.cs`).

    Notes
    -----
    This operation is specific to AMD CDNA3 hardware. It uses buffer store instructions which differ from standard pointer-based stores. The `offsets` tensor determines the address for each element relative to `ptr`.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(x_ptr, offsets, value, BLOCK_SIZE: ttgl.constexpr):
         pid = ttgl.program_id(0)
         offs = ttgl.arange(0, BLOCK_SIZE)
         mask = offs < BLOCK_SIZE
         val = value + offs
         ttgl.amd.cdna3.buffer_store(val, x_ptr, offs, mask=mask, cache=".cg")
```

---

### triton.experimental.gluon.language.amd.cdna3.mfma

```python
mfma(a, b, acc, _semantic: 'GluonSemantic' = None)
```

## mfma


**`mfma(a, b, acc, _semantic=None)`**

    Computes matrix-multiplication of `a * b + acc` using AMD CDNA3 matrix core units.

    Emits AMD MFMA (Matrix Fused Multiply-Add) instructions for CDNA3 architecture.

    Parameters
    ----------
    a : tensor
        The first operand of MFMA. Must have a `DotOperandLayout` with operand index 0.
    b : tensor
        The second operand of MFMA. Must have a `DotOperandLayout` with operand index 1.
    acc : tensor
        The accumulator tensor. Must have a `BlockedLayout`. The output tensor will have
        the same type as `acc`.
    _semantic : GluonSemantic, optional
        Internal semantic object. Do not set manually.

    Returns
    -------
    tensor
        A Gluon tensor containing the result of `a * b + acc` with the same type as `acc`.

    Raises
    ------
    AssertionError
        If `acc` is None, or if the layouts of `a`, `b`, and `acc` are incompatible
        for matrix multiplication.

    Notes
    -----
    This function is hardware-specific to AMD CDNA3 GPUs. The layouts of `a`, `b`, and
    `acc` must satisfy the following constraints:

    - `a` must have a `DotOperandLayout` with `operand_index=0`
    - `b` must have a `DotOperandLayout` with `operand_index=1`
    - `acc` must have a `BlockedLayout`
    - Both `a` and `b` layouts must have the same parent as `acc` layout

    The operation computes `a @ b + acc` using native matrix core instructions.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> @gluon.jit
    ... def matmul_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...     # Load tiles from global memory
    ...     a = ttgl.load(a_ptr)
    ...     b = ttgl.load(b_ptr)
    ...     # Initialize accumulator
    ...     acc = ttgl.full((BLOCK_M, BLOCK_N), 0.0, dtype=ttgl.float16)
    ...     # Perform matrix multiplication using CDNA3 MFMA
    ...     result = ttgl.amd.cdna3.mfma(a, b, acc)
    ...     ttgl.store(c_ptr, result)

---


## triton.experimental.gluon.language.amd.cdna4

### triton.experimental.gluon.language.amd.cdna4.buffer_atomic_add

```python
buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_add


.. autofunction:: buffer_atomic_add

Perform an atomic add operation on AMD CDNA4 buffer memory.

This function atomically adds a value to a memory location in buffer address space,
specifically optimized for AMD CDNA4 (Instinct MI300X/MI325X) GPU architecture.
The operation uses AMD's buffer atomic instructions for improved performance on
CDNA4 hardware.

### Parameters
ptr : pointer
    Pointer to the buffer memory location. Must be a buffer pointer type
    compatible with CDNA4 buffer addressing.
offsets : tensor
    Tensor of offsets to apply to the base pointer. Each offset specifies
    which memory location to operate on.
value : tensor
    The value to atomically add to each memory location. Must have the same
    shape as `offsets`.
mask : tensor, optional
    Boolean tensor specifying which lanes should execute the atomic operation.
    If `None`, all lanes execute the operation.
sem : str, optional
    Memory semantics for the atomic operation. Options include:
    `"acquire"`, `"release"`, `"acq_rel"`, or `"relaxed"`.
    Defaults to relaxed semantics.
scope : str, optional
    Memory scope for the atomic operation. Options include:
    `"cta"`, `"cluster"`, or `"gpu"`. Defaults to CTA scope.
_semantic : GluonSemantic, optional
    Internal semantic handler. Automatically provided by the Gluon JIT
    compiler. Do not set manually.

### Returns
tensor
    Tensor containing the original values at each memory location before
    the atomic add was performed. Has the same shape as `offsets`.

### Notes
This function is architecture-specific and only works on AMD CDNA4 GPUs.
For portable code, consider using the generic `atomic_add` from
`triton.experimental.gluon.language` instead.

Buffer atomic operations on CDNA4 provide better performance than global
memory atomics for certain access patterns, particularly when working
with buffer address space.

The operation is equivalent to:
`old_value = memory[ptr + offsets]; memory[ptr + offsets] += value`

### Examples

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.amd import cdna4

 @gluon.jit
 def buffer_atomic_add_kernel(
     ptr,
     offsets,
     values,
     BLOCK_SIZE: ttgl.constexpr,
 ):
     # Load offsets for this program
     pid = ttgl.program_id(0)
     offset_ptrs = pid * BLOCK_SIZE + ttgl.arange(0, BLOCK_SIZE)
     offsets = ttgl.load(offsets + offset_ptrs)
     
     # Load values to add
     values = ttgl.load(values + offset_ptrs)
     
     # Perform atomic add on buffer memory
     old_values = cdna4.buffer_atomic_add(ptr, offsets, values)
     
     # Store old values for verification
     ttgl.store(ptr + offset_ptrs + 1000, old_values)

 # Launch kernel
 buffer_atomic_add_kernel[(num_programs,)](
     buffer_ptr,
     offset_tensor,
     value_tensor,
     BLOCK_SIZE=256,
 )
```

---

### triton.experimental.gluon.language.amd.cdna4.buffer_atomic_and

```python
buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_and


**`buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None)`**

    Perform an atomic bitwise AND operation on AMD CDNA4 buffer memory.

    This function atomically reads a value from buffer memory, performs a bitwise
    AND with the provided value, and writes the result back. Returns the original
    value from memory before the operation.

    Parameters
    ----------
    ptr : pointer
        Pointer to the buffer memory location.
    offsets : tensor
        Tensor of offsets specifying which memory locations to operate on.
    value : tensor
        Tensor of values to AND with the memory contents.
    mask : tensor, optional
        Boolean tensor mask. Elements are only updated where mask is True.
    sem : str, optional
        Memory semantics for the atomic operation. Options include 'acquire',
        'release', 'acq_rel', or 'relaxed'.
    scope : str, optional
        Scope of the atomic operation. Options include 'cta', 'cluster', or 'gpu'.

    Returns
    -------
    tensor
        Tensor containing the original values from memory before the AND operation.

    Notes
    -----
    This operation is specific to AMD CDNA4 architecture and uses buffer atomic
    instructions. The operation computes::

        old_value = memory[offset]
        memory[offset] = old_value & value
        return old_value

    Only integer types are supported for buffer atomic AND operations.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(ptr, offsets, value):
         # Perform atomic AND on buffer memory
         old = ttgl.amd.cdna4.buffer_atomic_and(ptr, offsets, value)
         return old
```

---

### triton.experimental.gluon.language.amd.cdna4.buffer_atomic_max

```python
buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

Perform an atomic maximum operation on AMD CDNA4 buffer memory.

### Parameters
ptr : ttgl.pointer_type
    Pointer to the global memory location.
offsets : ttgl.tensor
    Offset indices for the buffer access.
value : ttgl.tensor
    The value to compare against.
mask : ttgl.tensor, optional
    Boolean mask to predicate the operation.
sem : ttgl.semantics, optional
    Memory semantics (e.g., ACQ_REL, RELAXED).
scope : ttgl.scope, optional
    Memory scope (e.g., CTA, SYSTEM).

### Returns
ttgl.tensor
    The original value(s) at the memory location before the update.

### Notes
This operation is specific to AMD CDNA4 architectures. It requires a Gluon JIT
context (`@gluon.jit <triton.experimental.gluon.jit>()`).

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 import triton.experimental.gluon.language.amd.cdna4 as cdna4

 @gluon.jit
 def kernel(ptr, offsets, value):
     old_val = cdna4.buffer_atomic_max(ptr, offsets, value)
     # Use old_val for subsequent computations
```

---

### triton.experimental.gluon.language.amd.cdna4.buffer_atomic_min

```python
buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_min


**`buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)`**

    Perform an atomic minimum operation on AMD CDNA4 buffer memory.

    Atomically computes `min(*ptr[offsets], value)` and stores the result back to
    memory. Returns the original value at the memory location before the operation.

    Parameters
    ----------
    ptr : pointer_type
        Pointer to the buffer memory location(s) to operate on.
    offsets : tensor
        Tensor of offsets specifying which elements to operate on.
    value : tensor
        Tensor of values to compare with memory contents.
    mask : tensor, optional
        Boolean tensor controlling which offsets participate in the operation.
        Elements where mask is False are skipped.
    sem : str, optional
        Memory semantics for the atomic operation (e.g., "acquire", "release", "acq_rel").
        Defaults to hardware default if not specified.
    scope : str, optional
        Memory scope for the atomic operation (e.g., "cta", "cluster", "gpu").
        Defaults to hardware default if not specified.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Must be provided by the JIT compiler.

    Returns
    -------
    tensor
        Tensor containing the original values from memory before the atomic minimum
        operation was applied.

    Notes
    -----
    This function is specific to AMD CDNA4 architecture and uses buffer atomic
    instructions. It must be called from within a kernel decorated with
    `@gluon.jit`.

    The operation is equivalent to:
    `old_value = atomicMin(ptr[offset], value)`

    Only integer and floating-point dtypes are supported for buffer atomics on
    CDNA4.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl
     from triton.experimental.gluon.language.amd import cdna4

     @gluon.jit
     def kernel(ptr, offsets, value):
         # Perform atomic minimum on buffer memory
         old_vals = cdna4.buffer_atomic_min(ptr, offsets, value)
         return old_vals
```

---

### triton.experimental.gluon.language.amd.cdna4.buffer_atomic_or

```python
buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_or


**`buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)`**

   Perform an atomic bitwise OR on buffer memory for AMD CDNA4 GPUs.

   Executes a read-modify-write operation that atomically computes `memory[offset] |= value`
   using AMD CDNA4 buffer atomic instructions. This operation is useful for implementing
   lock-free synchronization primitives and atomic flag updates in GPU kernels.

   Parameters
   ----------
   ptr : pointer
       Pointer to the buffer memory region. Must be a pointer type obtained from
       `triton.experimental.gluon.language.make_block_ptr()` or similar.
   offsets : tensor
       Tensor of offsets specifying which memory locations to operate on. Can be
       scalar or vector depending on the operation granularity.
   value : tensor
       The value to OR with the memory contents. Must have the same dtype as the
       memory region.
   mask : tensor, optional
       Boolean tensor controlling which offsets are active. If provided, only
       offsets where mask is True will execute the atomic operation.
   sem : str, optional
       Memory semantics for the atomic operation. Options include:
       - `"acquire"`: Acquire semantics
       - `"release"`: Release semantics
       - `"acq_rel"`: Acquire-release semantics
       - `"relaxed"`: Relaxed semantics (default)
   scope : str, optional
       Memory scope for the atomic operation. Options include:
       - `"cta"`: CTA scope (default)
       - `"cluster"`: Cluster scope
       - `"gpu"`: GPU scope
   _semantic : GluonSemantic, optional
       Internal semantic object. Do not set manually; this is handled by the
       `@gluon.jit()` decorator.

   Returns
   -------
   tensor
       Tensor containing the original values from memory before the OR operation
       was applied. This allows for compare-and-swap style patterns.

   Notes
   -----
   This function is specific to AMD CDNA4 architecture and uses the buffer
   atomic instruction set. For other architectures, use the appropriate
   architecture-specific atomic functions or the generic `atomic_or()`.

   The operation is lock-free and provides atomicity guarantees within the
   specified memory scope. Performance characteristics depend on memory
   coalescing and contention levels.

   Requires Gluon kernels decorated with `@gluon.jit()`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.amd import cdna4

   @gluon.jit
   def atomic_or_kernel(ptr, offsets, value):
       # Perform atomic OR on buffer memory
       old_value = cdna4.buffer_atomic_or(ptr, offsets, value)
       return old_value

   # Launch kernel
   # atomic_or_kernel[grid](ptr, offsets, value)
```

---

### triton.experimental.gluon.language.amd.cdna4.buffer_atomic_xor

```python
buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None)
```

## buffer_atomic_xor


**`buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None)`**

    Perform an atomic bitwise XOR operation on buffer memory for AMD CDNA4 GPUs.

    This function atomically reads a value from buffer memory, computes the bitwise
    XOR with the provided value, and writes the result back. Returns the original
    value read from memory before the operation.

    Parameters
    ----------
    ptr : pointer
        Pointer to the buffer memory location.
    offsets : tensor
        Tensor of offsets specifying which elements to operate on.
    value : tensor
        Tensor of values to XOR with the memory contents.
    mask : tensor, optional
        Boolean tensor mask for predicated execution. Elements where mask is False
        are not processed.
    sem : str, optional
        Memory semantics for the atomic operation. Common values include `"acq_rel"`
        (acquire-release) or `"relaxed"`. Defaults to hardware default.
    scope : str, optional
        Memory scope for the atomic operation. Common values include `"cta"`
        (block-level) or `"gpu"` (grid-level). Defaults to hardware default.

    Returns
    -------
    tensor
        Tensor containing the original values read from memory before the XOR
        operation was applied.

    Notes
    -----
    This operation is specific to AMD CDNA4 architecture and uses the buffer
    atomic read-modify-write (RMW) instruction path. The operation is performed
    element-wise for each active lane where the mask is True.

    Memory consistency guarantees depend on the `sem` and `scope` parameters.
    For correct synchronization across threads, appropriate memory semantics
    and scope must be specified.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl
     from triton.experimental.gluon.language.amd import cdna4

     @gluon.jit
     def kernel(ptr, offsets, value, BLOCK_SIZE: ttgl.constexpr):
         # Load offsets for each thread
         offsets = ttgl.arange(0, BLOCK_SIZE, layout=...)
         
         # Perform atomic XOR on buffer memory
         old_values = cdna4.buffer_atomic_xor(ptr, offsets, value, sem="acq_rel")
         
         # old_values contains the memory contents before XOR
```

---

### triton.experimental.gluon.language.amd.cdna4.buffer_load

```python
buffer_load(ptr, offsets, mask=None, other=None, cache=None, _semantic=None)
```

Load data from global memory using a scalar base pointer and offset tensor on AMD CDNA4 GPUs.

### Parameters
ptr : pointer_type
    Scalar base pointer to global memory.
offsets : tensor
    Tensor of byte offsets relative to `ptr`.
mask : tensor, optional
    Boolean tensor for predicated loads. Elements where `mask` is False
    will not be loaded from memory.
other : tensor or scalar, optional
    Value to fill for masked-off elements. If None, masked elements are
    undefined.
cache : str, optional
    Cache modifier specifier (e.g., '.ca', '.cg'). Controls cache behavior.

### Returns
tensor
    A Gluon tensor containing the loaded data.

### Notes
This operation is specific to AMD CDNA4 architecture. It uses buffer load
instructions that accept a scalar base pointer and a vector of offsets,
rather than a vector of pointers. Data is loaded directly into registers.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(ptr, offsets):
     # Load data using scalar base pointer and offset tensor
     data = ttgl.amd.cdna4.buffer_load(ptr, offsets)
     ...
```

---

### triton.experimental.gluon.language.amd.cdna4.buffer_store

```python
buffer_store(stored_value, ptr, offsets, mask=None, cache=None, _semantic: 'GluonSemantic' = None)
```

## buffer_store

**`buffer_store(stored_value, ptr, offsets, mask=None, cache=None)`**

   Store a tensor directly to global memory via a scalar base pointer and tensor of offsets.

   AMD CDNA4 buffer store operation that writes tensor data to global memory using a base
   pointer and offset tensor rather than a pointer tensor.

   Parameters
   ----------
   stored_value : tensor
       The tensor data to be stored to global memory.
   ptr : pointer
       Scalar base pointer to global memory location.
   offsets : tensor
       Tensor of offsets relative to the base pointer.
   mask : tensor, optional
       Boolean mask tensor for predicated store operations. Elements are only stored
       where mask is True. Defaults to None (unconditional store).
   cache : str, optional
       Cache modifier specifier for memory consistency semantics. Defaults to None.

   Returns
   -------
   None

   Notes
   -----
   This operation is specific to AMD CDNA4 architecture. The offsets tensor determines
   which memory locations relative to the base pointer receive data from stored_value.

   When mask is provided, only elements where mask evaluates to True are written to
   memory. This enables conditional/predicated memory operations.

   The stored_value, ptr, offsets, and mask tensors are broadcast to compatible shapes
   before the store operation executes.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr, offsets, value):
       ttgl.amd.cdna4.buffer_store(value, ptr, offsets)

   @gluon.jit
   def kernel_masked(ptr, offsets, value, mask):
       ttgl.amd.cdna4.buffer_store(value, ptr, offsets, mask=mask)
```

---

### triton.experimental.gluon.language.amd.cdna4.get_mfma_scale_layout

```python
get_mfma_scale_layout(dot_operand_layout, shape)
```

**`get_mfma_scale_layout(dot_operand_layout, shape)`**

   Compute the distributed linear layout for scale tensors required by AMD MFMA scaled operations.

   This constexpr function derives the correct memory layout for scale factors associated with microscaling formats (MX) on AMD CDNA4 architectures. The layout depends on the operand index (A or B) and the parent MFMA instruction configuration.

   Parameters
   ----------
   dot_operand_layout : DotOperandLayout
       The layout of the scaled operand (A or B). The parent layout must be an instance of `AMDMFMALayout`.
   shape : list[int]
       The logical shape of the scale tensor.

   Returns
   -------
   DistributedLinearLayout
       The layout describing the distribution of scale elements across threads and warps.

   Notes
   -----
   This function is specific to AMD CDNA4 hardware. It is typically used internally by `mfma_scaled()` but exposed for advanced layout manipulation.

   The operand index is inferred from `dot_operand_layout`. Ensure the parent layout matches the accumulator layout used in the MFMA operation.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(...):
       # Assume dot_layout is derived from accumulator layout
       scale_layout = ttgl.amd.cdna4.get_mfma_scale_layout(dot_layout, scale_shape)
```

---

### triton.experimental.gluon.language.amd.cdna4.mfma

```python
mfma(a, b, acc, _semantic: 'GluonSemantic' = None)
```

## mfma


**`mfma(a, b, acc, _semantic=None)`**

    Compute matrix multiplication using AMD CDNA4 Matrix Fused Multiply-Add (MFMA) units.

    Performs the operation `a @ b + acc` using AMD's native matrix core instructions
    on CDNA4 architecture GPUs.

    Parameters
    ----------
    a : tensor
        First operand of the matrix multiplication. Must have a `DotOperandLayout`
        with `operand_index=0`.
    b : tensor
        Second operand of the matrix multiplication. Must have a `DotOperandLayout`
        with `operand_index=1`.
    acc : tensor
        Accumulator tensor. Must have a `BlockedLayout` that is the parent layout
        of both `a` and `b`. The output type matches `acc.dtype`.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Do not pass explicitly; this is set automatically
        when using `@gluon.jit` decorated functions.

    Returns
    -------
    tensor
        Result of `a @ b + acc` with the same type as `acc`.

    Notes
    -----
    This operation is specific to AMD CDNA4 architecture and utilizes the MFMA
    instruction for accelerated matrix operations. Input tensors must have
    compatible layouts for matrix multiplication. The accumulator tensor is
    required and cannot be `None`.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl
     from triton.experimental.gluon.language.amd import cdna4

     @gluon.jit
     def matmul_kernel(...):
         # Perform matrix multiplication with accumulation
         result = cdna4.mfma(a, b, acc)
```

---

### triton.experimental.gluon.language.amd.cdna4.mfma_scaled

```python
mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None)
```

AMD scaled matrix fragment multiply-accumulate operation with microscaling formats.

Performs the computation `c = a * a_scale @ b * b_scale + acc` using AMD MFMA
instructions with microscaling (MX) operand formats.

### Parameters
a : tensor
    The left-hand operand tensor to be multiplied. Must use a
    `DotOperandLayout` with parent matching the MFMA layout.
a_scale : tensor, optional
    Scale factor tensor for operand `a`.
a_format : str
    Microscaling format of operand `a`. Available formats: `e2m1`,
    `e4m3`, `e5m2`.
b : tensor
    The right-hand operand tensor to be multiplied. Must use a
    `DotOperandLayout` with parent matching the MFMA layout.
b_scale : tensor, optional
    Scale factor tensor for operand `b`.
b_format : str
    Microscaling format of operand `b`. Available formats: `e2m1`,
    `e4m3`, `e5m2`.
acc : tensor
    Accumulator tensor. Must use an `AMDMFMALayout`.

### Returns
tensor
    Result tensor of the scaled MFMA operation with the same layout as
    `acc`.

### Notes
This operation is currently supported only on AMD CDNA4 hardware.

Operands `a` and `b` use microscaling formats described in the
"OCP Microscaling Formats (MX) Specification":
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Layout requirements:
- `acc` must have an `AMDMFMALayout`
- `a` and `b` must have `DotOperandLayout` with parent matching
  the MFMA layout of `acc`

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.amd import cdna4

 @gluon.jit
 def kernel(...):
     # Assume a, b, acc are properly laid out tensors
     result = cdna4.mfma_scaled(
         a, a_scale, "e4m3",
         b, b_scale, "e4m3",
         acc
     )
```

---


## triton.experimental.gluon.language.amd.gfx1250

### triton.experimental.gluon.language.amd.gfx1250.PartitionedSharedLayout

```python
PartitionedSharedLayout(num_partitions: 'int', num_groups: 'int', partition_dim: 'int', partition_layout: 'SharedLayout') -> None
```

## PartitionedSharedLayout

Represents a partitioned shared memory layout that splits a tensor across multiple physical shared memory partitions.

This reduces shared memory partition conflicts by placing different pieces of a tensor in separate physical memory slots.

### Parameters
num_partitions : int
    Number of physical memory partitions. Must be a power of two.
num_groups : int
    Number of groups. Each group has `num_partitions` pieces. Must be a power of two.
partition_dim : int
    Dimension along which to partition. Must be non-negative.
partition_layout : SharedLayout
    Inner layout for each piece (e.g., `SwizzledSharedLayout` or `PaddedSharedLayout`).

### Notes
Both `num_partitions` and `num_groups` must be powers of two. `partition_dim` must be non-negative.

### Examples
```python
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.amd.gfx1250 import PartitionedSharedLayout

 # Create a layout with 4 partitions and 2 groups along dimension 0
 layout = PartitionedSharedLayout(
     num_partitions=4,
     num_groups=2,
     partition_dim=0,
     partition_layout=ttgl.SwizzledSharedLayout(...)
 )
```

---

### triton.experimental.gluon.language.amd.gfx1250.buffer_load

```python
buffer_load(ptr, offsets, mask=None, other=None, cache=None, _semantic=None)
```

AMD buffer load from global memory via a scalar base pointer and tensor of offsets.

This operation loads data directly from global memory into registers using AMD's
buffer load instruction on GFX1250 architecture. Unlike pointer-based loads, this
uses a single base pointer with per-element offsets.

### Parameters
ptr : pointer to scalar
    Global memory scalar base pointer to load from.
offsets : tensor
    Tensor of byte offsets relative to `ptr`. Each element specifies the
    offset for the corresponding load operation.
mask : tensor, optional
    Boolean mask tensor for predicated loads. Elements where mask is False
    will not perform the load. Defaults to None.
other : tensor or scalar, optional
    Default values for masked elements. When `mask` is provided, elements
    where mask is False will receive values from `other`. Defaults to None.
cache : str, optional
    Cache modifier specifier controlling cache behavior (e.g., "ca", "cg",
    "lu", "cv"). Defaults to None (no cache modifier).

### Returns
tensor
    A Gluon tensor containing the loaded data. The tensor has the same shape
    as `offsets` and element type matching `ptr`'s pointee type.

### Notes
This is an AMD GFX1250-specific operation. The buffer load instruction
provides more efficient memory access patterns compared to traditional
pointer-based loads on supported hardware.

The `offsets` tensor determines the memory addresses accessed: each
address is computed as `ptr + offsets[i]`.

When using `mask`, ensure `other` is provided to define values for
masked-out elements, otherwise behavior is undefined.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(ptr, offsets, BLOCK: ttgl.constexpr):
     # Load 128 elements from global memory using buffer load
     offsets = ttgl.arange(0, BLOCK, layout=...)
     data = ttgl.amd.gfx1250.buffer_load(ptr, offsets)

 @gluon.jit
 def kernel_masked(ptr, offsets, mask, BLOCK: ttgl.constexpr):
     # Predicated load with default value for masked elements
     offsets = ttgl.arange(0, BLOCK, layout=...)
     data = ttgl.amd.gfx1250.buffer_load(
         ptr, offsets, mask=mask, other=0.0
     )
```

---

### triton.experimental.gluon.language.amd.gfx1250.buffer_store

```python
buffer_store(stored_value, ptr, offsets, mask=None, cache=None, _semantic: 'GluonSemantic' = None)
```

## buffer_store


**`buffer_store(stored_value, ptr, offsets, mask=None, cache=None, _semantic=None)`**

   Store a tensor to global memory using a scalar base pointer and offset tensor on AMD GFX1250.

   This operation writes `stored_value` to global memory addresses computed as `ptr + offsets`,
   using AMD buffer store instructions. Unlike regular `triton.experimental.gluon.language.store()`,
   this uses a scalar pointer with a tensor of offsets rather than a tensor of pointers.

   Parameters
   ----------
   stored_value : tensor
       The tensor data to store to global memory.
   ptr : pointer
       Scalar base pointer to global memory.
   offsets : tensor
       Tensor of offsets added to `ptr` to compute store addresses.
   mask : tensor, optional
       Predicate mask tensor for conditional stores. Elements where mask is False
       will not be stored. Defaults to None (unconditional store).
   cache : str, optional
       Cache modifier specifier for the store operation (e.g., `"ca"`, `"cg"`).
       Defaults to None (default cache behavior).
   _semantic : GluonSemantic, optional
       Internal semantic handler. Must be provided by the JIT compiler.

   Returns
   -------
   None

   Notes
   -----
   This function is specific to AMD GFX1250 (RDNA4) architecture and requires
   the kernel to be decorated with `@gluon.jit <triton.experimental.gluon.jit>()`.

   All tensor arguments (`stored_value`, `offsets`, `mask`) are broadcasted
   to a common shape before the store operation. If shapes mismatch after broadcasting,
   a ValueError is raised.

   The buffer store instruction provides more efficient memory access patterns
   compared to pointer-based stores for certain access patterns on AMD hardware.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr, offsets, value):
       # Store value to global memory at ptr + offsets
       ttgl.amd.gfx1250.buffer_store(value, ptr, offsets)

   @gluon.jit
   def kernel_masked(ptr, offsets, value, mask):
       # Predicated store - only stores where mask is True
       ttgl.amd.gfx1250.buffer_store(value, ptr, offsets, mask=mask)

   @gluon.jit
   def kernel_cached(ptr, offsets, value):
       # Store with cache modifier
       ttgl.amd.gfx1250.buffer_store(value, ptr, offsets, cache="ca")
```

---

### triton.experimental.gluon.language.amd.gfx1250.get_wmma_scale_layout

```python
get_wmma_scale_layout(dot_operand_layout, shape)
```

Compute the distributed layout for scale factors required by AMD WMMA scaled operations.

Derives the scale tensor layout from the matrix operand layout, ensuring correct
mapping of scale factors to threads and warps for :py`ttgl.amd.gfx1250.wmma_scaled()`
on AMD GFX1250 architectures.

### Parameters
dot_operand_layout : DotOperandLayout
    The layout of the matrix operand (A or B) involved in the WMMA operation.
    The parent layout must be an instance of `AMDWMMALayout`.
shape : list[int]
    The logical shape of the scale tensor (e.g., `[M, K]` for operand A).

### Returns
layout : DistributedLinearLayout
    The distributed layout describing how scale elements are partitioned
    across threads, warps, and CGAs.

### Notes
This is a `constexpr_function` evaluated at compile time.
Specific to AMD RDNA4 (GFX1250) hardware features.
Typically invoked internally by :py`ttgl.amd.gfx1250.wmma_scaled()`, but
exposed for custom layout manipulation.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.amd import gfx1250

 @gluon.jit
 def kernel(...):
     # Assume 'a' is a tensor with WMMA layout
     layout = a.type.layout
     scale_shape = [M, K]  # Derived from problem shape
     scale_layout = gfx1250.get_wmma_scale_layout(layout, scale_shape)
     # Use scale_layout to allocate shared memory or load scales
```

---

### triton.experimental.gluon.language.amd.gfx1250.wmma

```python
wmma(a, b, acc, _semantic=None)
```

## triton.experimental.gluon.language.amd.gfx1250.wmma


**`wmma(a, b, acc, _semantic=None)`**

   Computes matrix multiplication of `a @ b + acc` using the AMD WMMA instruction.

   Performs a warp-level matrix multiply-accumulate operation on AMD GFX1250 (RDNA4)
   GPUs. This is a hardware-specific intrinsic that provides accelerated matrix
   multiplication for supported data types.

   Parameters
   ----------
   a : tensor
      The first operand tensor to be multiplied. Must have a `DotOperandLayout`
      with operand index 0.
   b : tensor
      The second operand tensor to be multiplied. Must have a `DotOperandLayout`
      with operand index 1.
   acc : tensor
      The accumulator tensor. Must have a `BlockedLayout` compatible with the
      parent layout of operands `a` and `b`.
   _semantic : GluonSemantic, optional
      Internal semantic handler. Do not set manually; this is provided automatically
      by the `@gluon.jit` decorator.

   Returns
   -------
   tensor
      A tensor containing the result of the matrix multiply-accumulate operation
      `a @ b + acc`. Has the same shape and type as `acc`.

   Notes
   -----
   This function is specific to AMD GFX1250 (RDNA4) architecture. The operands must
   satisfy layout constraints:

   - `a` must have a `DotOperandLayout` with operand index 0
   - `b` must have a `DotOperandLayout` with operand index 1
   - `acc` must have a `BlockedLayout` that is the parent layout of both
     operands
   - All tensors must have matching batch dimensions (2D or 3D)

   The WMMA instruction provides hardware-accelerated matrix multiplication for
   specific data type combinations (e.g., FP16, INT8, BF16 depending on GPU
   capabilities).

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.amd import gfx1250

   @gluon.jit
   def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
       # Program ID
       pid = ttgl.program_id(0)

       # Define block sizes
       BLOCK_M = 64
       BLOCK_N = 64
       BLOCK_K = 32

       # Compute starting positions
       m = pid * BLOCK_M
       n = 0

       # Allocate accumulators
       acc = ttgl.full((BLOCK_M, BLOCK_N), 0.0, dtype=ttgl.float16)

       # Load tiles from global memory
       a = ttgl.load(a_ptr + m * K + ttgl.arange(0, BLOCK_K)[:, None])
       b = ttgl.load(b_ptr + ttgl.arange(0, BLOCK_K)[None, :] * N)

       # Perform WMMA operation
       result = gfx1250.wmma(a, b, acc)

       # Store result
       ttgl.store(c_ptr + m * N, result)
```

---

### triton.experimental.gluon.language.amd.gfx1250.wmma_scaled

```python
wmma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None)
```

## AMD Scaled WMMA Operation

**`triton.experimental.gluon.language.amd.gfx1250.wmma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None)`**

   Perform scaled matrix multiply-accumulate using AMD WMMA instructions with microscaling formats.

   Computes $c = (a \times a\_scale) @ (b \times b\_scale) + acc$.

   Parameters
   ----------
   a : tensor
       The left-hand operand A to be multiplied. Must use a microscaling format.
   a_scale : tensor, optional
       Scale factor for operand A.
   a_format : str
       Format of operand A. Available formats: `'e2m1'`, `'e4m3'`, `'e5m2'`.
   b : tensor
       The right-hand operand B to be multiplied. Must use a microscaling format.
   b_scale : tensor, optional
       Scale factor for operand B.
   b_format : str
       Format of operand B. Available formats: `'e2m1'`, `'e4m3'`, `'e5m2'`.
   acc : tensor
       Accumulator tensor. Must have layout shape `[16, 16, 128]`.
   _semantic : GluonSemantic, optional
       Internal semantic handler. Do not set manually.

   Returns
   -------
   tensor
       Result of the scaled matrix multiply-accumulate operation.

   Notes
   -----
   Operand formats `a` and `b` use microscaling formats described in the
   `OCP Microscaling Formats (MX) Specification`_:

   .. _OCP Microscaling Formats (MX) Specification: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

   Layout requirements:

   - Accumulator tensor must have `AM DWMMALayout` with `instr_shape = [16, 16, 128]`.
   - When `a_format` or `b_format` is `'e2m1'`, the corresponding operand must have
     `instr_shape = [16, 16, 64]`.

   This function is AMD GFX1250-specific and requires appropriate hardware support.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.amd import gfx1250

   @gluon.jit
   def kernel(...):
       # Assume a, b, acc are properly laid out tensors
       result = gfx1250.wmma_scaled(
           a, a_scale, "e4m3",
           b, b_scale, "e4m3",
           acc
       )
```

---


## triton.experimental.gluon.language.amd.gfx1250.async_copy

### triton.experimental.gluon.language.amd.gfx1250.async_copy.commit_group

```python
commit_group(_semantic=None)
```



---

### triton.experimental.gluon.language.amd.gfx1250.async_copy.global_to_shared

```python
global_to_shared(smem, pointer, mask=None, other=None, cache_modifier='', _semantic=None)
```

## async_copy.global_to_shared

**`triton.experimental.gluon.language.amd.gfx1250.async_copy.global_to_shared(smem, pointer, mask=None, other=None, cache_modifier='', _semantic=None)`**

   Asynchronously copy elements from global memory to shared memory on AMD GFX1250 GPUs.

   Parameters
   ----------
   smem : shared_memory_descriptor
      Destination shared memory descriptor. Must have shape matching `pointer`.
   pointer : tensor
      Source pointer tensor with distributed layout. Must be a block tensor.
   mask : tensor, optional
      Mask tensor for predicated loads. If provided, elements where mask is False
      will not be loaded. Defaults to None.
   other : tensor or scalar, optional
      Default values for masked elements. Used when `mask` is provided and
      evaluates to False. Defaults to None (zero).
   cache_modifier : str, optional
      Cache modifier specifier for load operation. Defaults to empty string.

   Returns
   -------
   None

   Notes
   -----
   This is an asynchronous copy operation that does not block execution. The
   loaded data is not immediately available in shared memory. You must call
   `wait_group()` to synchronize before accessing the data in `smem`.

   The `pointer` tensor must have a `DistributedLayout`. The shape of `smem`
   must exactly match the shape of `pointer`.

   This operation is specific to AMD GFX1250 architecture and leverages hardware
   async copy capabilities.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(ptr, smem):
       # Asynchronously copy from global to shared memory
       ttgl.amd.gfx1250.async_copy.global_to_shared(smem, ptr)
       
       # Must wait before accessing smem contents
       ttgl.amd.gfx1250.async_copy.wait_group(0)
       
       # Now safe to use smem
       data = smem.load(...)
```

---

### triton.experimental.gluon.language.amd.gfx1250.async_copy.mbarrier_arrive

```python
mbarrier_arrive(mbarrier, _semantic=None)
```

**`triton.experimental.gluon.language.amd.gfx1250.async_copy.mbarrier_arrive(mbarrier, _semantic=None)`**

   Arrive on the memory barrier once all outstanding async copies are complete.

   This operation signals to the barrier that all asynchronous copy operations associated with it have finished. It is required before threads waiting on the barrier can proceed.

   Parameters
   ----------
   mbarrier : shared_memory_descriptor
       Barrier object to arrive on.
   _semantic : GluonSemantic, optional
       Internal semantic context injected by the JIT compiler. Users should not pass this argument.

   Returns
   -------
   None

   Notes
   -----
   This function is specific to AMD GFX1250 hardware (RDNA4/CDNA4). It interfaces with the hardware memory barrier mechanism for async copies (LDS barrier arrive). Ensure all relevant async copy operations are issued before calling this function to guarantee correct synchronization.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>> @gluon.jit
   ... def kernel(mbarrier_ptr):
   ...     # Load barrier descriptor from global memory
   ...     mbarrier = ttgl.load(mbarrier_ptr)
   ...     # Issue async copies ...
   ...     # Signal completion to the barrier
   ...     ttgl.amd.gfx1250.async_copy.mbarrier_arrive(mbarrier)

---

### triton.experimental.gluon.language.amd.gfx1250.async_copy.shared_to_global

```python
shared_to_global(pointer, smem, mask=None, cache_modifier='', _semantic=None)
```

**`shared_to_global(pointer, smem, mask=None, cache_modifier='', _semantic=None)`**

    Asynchronously copy elements from shared memory to global memory.

    This operation initiates an asynchronous transfer from shared memory to global
    memory. Completion must be enforced manually using `wait_group()` before
    the data is guaranteed to be visible in global memory.

    Parameters
    ----------
    pointer : tensor
        Destination pointer tensor. Must have a `DistributedLayout`.
    smem : shared_memory_descriptor
        Source shared memory descriptor. Shape must match `pointer`.
    mask : tensor, optional
        Predicate mask for predicated stores. Defaults to `None`.
    cache_modifier : str, optional
        Cache modifier specifier (e.g., `.cg`, `.cs`). Defaults to `''`.
    _semantic : GluonSemantic, optional
        Internal semantic argument. Do not set manually.

    Returns
    -------
    None

    Notes
    -----
    Specific to AMD GFX1250 architecture. Requires manual synchronization via
    `wait_group()` after issuance before accessing the stored data globally.
    The shape of `smem` must exactly match the shape of `pointer`. The pointer
    type layout must be a `DistributedLayout`.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(...):
         # Allocate shared memory
         smem = ttgl.allocate_shared_memory(...)
         # ... compute data into smem ...

         # Async copy to global
         ptr = ttgl.make_block_ptr(...)
         ttgl.amd.gfx1250.async_copy.shared_to_global(ptr, smem)

         # Must wait before data is visible globally
         ttgl.amd.gfx1250.async_copy.wait_group(0)
```

---

### triton.experimental.gluon.language.amd.gfx1250.async_copy.wait_group

```python
wait_group(num_outstanding=0, _semantic=None)
```

Wait for outstanding async copy commit groups.

Blocks until the number of outstanding commit groups is less than or equal to
`num_outstanding`. Uncommitted async operations are waited upon even if
`num_outstanding` is 0.

### Parameters
num_outstanding : int, optional
    The number of outstanding commit groups to wait for. Defaults to 0.

### Notes
This function is specific to AMD GFX1250 hardware async copy mechanisms.
It ensures synchronization of asynchronous memory operations before proceeding.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language.amd.gfx1250.async_copy as async_copy

 @gluon.jit
 def kernel(...):
     async_copy.copy(...)
     async_copy.wait_group(num_outstanding=0)
```

---


## triton.experimental.gluon.language.amd.gfx1250.cluster

### triton.experimental.gluon.language.amd.gfx1250.cluster.arrive

```python
arrive(_semantic=None)
```

Signals that the cluster has arrived at a cluster barrier.

Synchronizes execution of CTAs within the same cluster on AMD GFX1250 hardware. This primitive is used in conjunction with cluster wait operations to ensure coordination across CTAs.

### Returns
None

### Notes
This API is specific to AMD GFX1250 architectures. It requires the kernel to be configured with cluster support. All CTAs in the cluster must eventually signal arrival to avoid deadlock.

### Examples
```python
 import triton.experimental.gluon as gluon
 from triton.experimental.gluon.language.amd.gfx1250 import cluster

 @gluon.jit
 def my_kernel(...):
     # Do work
     ...
     # Signal arrival at the cluster barrier
     cluster.arrive()
     # Wait for synchronization
     ...
```

---

### triton.experimental.gluon.language.amd.gfx1250.cluster.wait

```python
wait(_semantic=None)
```

**`triton.experimental.gluon.language.amd.gfx1250.cluster.wait()`**

   Wait for all CTAs in a cluster to arrive at a cluster barrier.

   Synchronizes execution across all CTAs within the same cluster on AMD GFX1250 (RDNA4) hardware. This operation blocks until all CTAs in the cluster have signaled arrival at the corresponding barrier.

   Returns
   -------
   None

   Notes
   -----
   This function is specific to AMD GFX1250 hardware. `wait` must be paired with a corresponding `arrive` operation. Calling `wait` before `arrive`, or calling `arrive` multiple times without a matching `wait`, results in undefined behavior.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.amd.gfx1250 import cluster

   @gluon.jit
   def kernel():
       # Signal arrival at the cluster barrier
       cluster.arrive()
       # Wait for all CTAs in the cluster to arrive
       cluster.wait()
```

---


## triton.experimental.gluon.language.amd.gfx1250.mbarrier

### triton.experimental.gluon.language.amd.gfx1250.mbarrier.MBarrierLayout

```python
MBarrierLayout(cga_layout=None)
```

**`MBarrierLayout(cga_layout=None)`**

   Layout for mbarrier synchronization on AMD GFX1250 hardware.

   Inherits from `triton.experimental.gluon.language._layouts.SwizzledSharedLayout`.
   This layout configures shared memory descriptors for hardware memory barriers
   used in thread block synchronization.

   Parameters
   ----------
   cga_layout : list of list of int, optional
      CGA layout bases. Defaults to empty list.

   Notes
   -----
   Specific to AMD GFX1250 (and compatible CDNA/RDNA) architectures.
   The layout ensures correct alignment and swizzling for mbarrier operations
   such as `triton.experimental.gluon.language.amd.gfx1250.mbarrier.init()`,
   `triton.experimental.gluon.language.amd.gfx1250.mbarrier.wait()`, and
   `triton.experimental.gluon.language.amd.gfx1250.mbarrier.arrive()`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   from triton.experimental.gluon.language.amd.gfx1250 import mbarrier

   # Create default mbarrier layout
   layout = mbarrier.MBarrierLayout()

   # Create with specific CGA layout
   layout = mbarrier.MBarrierLayout(cga_layout=[[1, 2], [3, 4]])
```

---

### triton.experimental.gluon.language.amd.gfx1250.mbarrier.arrive

```python
arrive(mbarrier, *, count=1, _semantic=None)
```

**`triton.experimental.gluon.language.amd.gfx1250.mbarrier.arrive(mbarrier, *, count=1, _semantic=None)`**

    Arrive at a memory barrier and decrement the pending count.

    Signals arrival at the specified `mbarrier` by decreasing the pending arrival
    count by `count`. If the pending count reaches zero, the barrier phase changes
    (decremented in a wraparound manner) and the pending count is reloaded with the
    initial count value.

    Parameters
    ----------
    mbarrier : shared_memory_descriptor
        Handle to the barrier to be signaled.
    count : int, optional
        Count to arrive with. Defaults to 1.
    _semantic : GluonSemantic, optional
        Internal semantic context. Do not pass directly.

    Returns
    -------
    phase : tensor
        0-D tensor of `int32` representing the mbarrier's phase parity (0 for even,
        1 for odd) prior to the arrive operation.

    Notes
    -----
    This operation is specific to AMD GFX1250 hardware. The returned phase parity
    can be used to track barrier state transitions across warp groups. The operation
    requires a `count` attribute of at least 1.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> @gluon.jit
    ... def kernel(...):
    ...     mbarrier = ttgl.allocate_shared_memory(...)
    ...     phase = ttgl.amd.gfx1250.mbarrier.arrive(mbarrier, count=1)
    ...     ttgl.debug_barrier()

---

### triton.experimental.gluon.language.amd.gfx1250.mbarrier.init

```python
init(mbarrier, count, _semantic=None)
```

**`init(mbarrier, count)`**

   Initialize a memory barrier (`mbarrier`) in shared memory with a specified arrival count.

   An mbarrier consists of an init count, a pending count, and a phase. At initialization, the init count and pending count are set to the given `count`, and the phase is initialized to 0. This operation is specific to AMD GFX1250 hardware.

   Parameters
   ----------
   mbarrier : shared_memory_descriptor
       The shared memory object representing the barrier to initialize.
   count : int
       The initial count for the barrier. Must be a positive integer representing
       the expected number of arrivals to trigger the barrier.

   Returns
   -------
   None

   Notes
   -----
   This API is hardware-specific to AMD GFX1250 (RDNA3/CDNA3 class) GPUs.
   The barrier state is stored in shared memory and must be accessed by
   threads within the same workgroup. The phase bit toggles upon barrier
   arrival completion to support multiple usage rounds.

   Examples
   --------
   Initialize a barrier in shared memory and use it to synchronize threads:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from ttgl.amd import gfx1250

   @gluon.jit
   def kernel(X, stride):
       # Allocate shared memory for the barrier (size depends on hardware spec)
       mbarrier = ttgl.allocate_shared_memory(ttgl.int32, [1], ttgl.SharedLayout())

       # Initialize the barrier with expected thread count (e.g., 128)
       gfx1250.mbarrier.init(mbarrier, 128)

       # ... perform work ...

       # Arrive at the barrier
       gfx1250.mbarrier.arrive(mbarrier)

       # Wait for all threads to arrive
       gfx1250.mbarrier.wait(mbarrier, 0)
```

---

### triton.experimental.gluon.language.amd.gfx1250.mbarrier.wait

```python
wait(mbarrier, phase, _semantic=None)
```

## wait

Wait until the mbarrier's phase differs from the provided phase value.

### Parameters
mbarrier : shared_memory_descriptor
    The barrier object to wait on.
phase : int
    The phase value to compare against. The wait completes when the barrier's
    phase becomes different from this value.
_semantic : GluonSemantic, optional
    Internal semantic context. Do not set manually.

### Notes
This is an AMD GFX1250-specific barrier wait operation for thread
synchronization. The function blocks execution until the specified phase has
completed, which is indicated by the barrier's phase value changing from the
provided `phase` argument.

This operation is typically used in conjunction with mbarrier arrive operations
to coordinate work between thread groups in shared memory.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(...):
     # Allocate shared memory with mbarrier
     mbarrier = ttgl.allocate_shared_memory(...)

     # Signal arrival at phase 0
     ttgl.amd.gfx1250.mbarrier.arrive(mbarrier, 0)

     # Wait for phase 0 to complete before proceeding
     ttgl.amd.gfx1250.mbarrier.wait(mbarrier, 0)

     # Continue with synchronized work
     ...
```

---


## triton.experimental.gluon.language.amd.gfx1250.tdm

### triton.experimental.gluon.language.amd.gfx1250.tdm.async_load

```python
async_load(src: 'tensor_descriptor', offsets: 'List[ttgl.constexpr | ttgl.tensor]', dest: 'shared_memory_descriptor', pred=1, mbarrier: 'shared_memory_descriptor' = None, _semantic=None) -> 'None'
```

## async_load

Load a block of tensor from global memory to shared memory asynchronously using AMD GFX1250 TDM.

### Parameters
src : tensor_descriptor
    The source tensor descriptor specifying the global memory location and layout.
offsets : List[constexpr | tensor]
    The offsets from the base pointer in the tensor descriptor. Can be compile-time
    constants or runtime tensors.
dest : shared_memory_descriptor
    The shared memory destination descriptor where the loaded data will be stored.
pred : constexpr or tensor, optional
    Predicate to enable or disable the load. Defaults to 1 (always execute).
mbarrier : shared_memory_descriptor, optional
    The memory barrier object to signal "arrive" on upon load completion. If None,
    no barrier signaling occurs.

### Returns
None

### Notes
This function is specific to AMD GFX1250 architecture and uses TDM (Tensor Data Movement)
for asynchronous memory transfers. The operation is non-blocking and returns immediately
while the data transfer continues in the background. Use `mbarrier` to synchronize
on completion if needed.

The `offsets` parameter allows specifying multiple offset values for complex access
patterns. Each offset is applied relative to the base pointer in the tensor descriptor.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.amd.gfx1250.tdm import async_load

 @gluon.jit
 def kernel(src_desc, dst_desc, mbarrier):
     offsets = [ttgl.constexpr(0)]
     async_load(
         src=src_desc,
         offsets=offsets,
         dest=dst_desc,
         mbarrier=mbarrier
     )
     # Wait for the async load to complete
     ttgl.mbarrier.arrive_and_wait(mbarrier)
```

---

### triton.experimental.gluon.language.amd.gfx1250.tdm.async_scatter

```python
async_scatter(desc: 'tensor_descriptor', dst_row_indices: 'ttgl.tensor', dst_col_offset, src: 'shared_memory_descriptor', mbarrier: 'shared_memory_descriptor' = None, _semantic=None) -> 'None'
```

## async_scatter

**`triton.experimental.gluon.language.amd.gfx1250.tdm.async_scatter(desc, dst_row_indices, dst_col_offset, src, mbarrier=None)`**

   Scatter data from shared memory to non-contiguous rows in global memory asynchronously.

   This operation uses TDM (Tensor Data Movement) scatter mode to write data to arbitrary
   rows in global memory. Unlike `async_store()` which writes to contiguous rows,
   scatter allows writing to non-contiguous rows specified by the `dst_row_indices`
   tensor.

   Parameters
   ----------
   desc : tensor_descriptor
       The destination tensor descriptor. Must be 2D.
   dst_row_indices : ttgl.tensor
       1D tensor of row indices (int16 or int32) in the destination tensor.
   dst_col_offset : int or tensor
       The starting column offset in the destination tensor for all scattered rows.
   src : shared_memory_descriptor
       The shared memory source containing data to scatter. Must be 2D.
   mbarrier : shared_memory_descriptor, optional
       The barrier object to signal "arrive" on. If None, no barrier signaling occurs.

   Returns
   -------
   None

   Notes
   -----
   The dtype of `dst_row_indices` determines the index size and scattering capacity:

   - **int16**: up to 16 rows can be scattered per TDM instruction
   - **int32**: up to 8 rows can be scattered per TDM instruction

   If more rows are needed than a single TDM instruction supports, multiple TDM
   instructions will be automatically issued.

   This is an AMD GFX1250-specific operation using TDM scatter mode. The operation
   is asynchronous and returns immediately without waiting for the memory transfer
   to complete.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.amd import gfx1250

   @gluon.jit
   def kernel(desc, src, mbarrier):
       # Create row indices for non-contiguous scatter
       row_indices = ttgl.full([8], 0, dtype=ttgl.int16)
       row_indices = ttgl.arange(0, 8, layout=...)  # customize layout

       # Scatter from shared memory to global memory at specified rows
       gfx1250.tdm.async_scatter(
           desc=desc,
           dst_row_indices=row_indices,
           dst_col_offset=0,
           src=src,
           mbarrier=mbarrier
       )
```

---

### triton.experimental.gluon.language.amd.gfx1250.tdm.async_wait

```python
async_wait(num_outstanding=0, _semantic=None) -> 'None'
```

## async_wait

Wait for outstanding asynchronous tensor data movement (TDM) operations to complete.

### Parameters
num_outstanding : int, optional
    Number of outstanding async tensor operations to wait for. Default is 0,
    which waits for all outstanding operations to complete.
_semantic : GluonSemantic, optional
    Internal semantic context passed automatically by the Gluon JIT compiler.
    Do not set this argument manually.

### Returns
None

### Notes
This function is specific to AMD GFX1250 hardware and the TDM (Tensor Data
Movement) asynchronous execution model. It inserts a hardware wait instruction
that blocks until the specified number of outstanding async operations have
completed.

When `num_outstanding=0`, the wait blocks until all pending async TDM
operations finish. When set to a positive integer N, it waits until N
operations remain outstanding (i.e., waits for all but N operations to
complete).

This is a hardware barrier for async tensor operations and should be used
to ensure data consistency before accessing results from async loads or
stores.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel_async_wait():
     # Wait for all outstanding async TDM operations to complete
     ttgl.amd.gfx1250.tdm.async_wait(num_outstanding=0)

     # Wait until only 2 operations remain outstanding
     ttgl.amd.gfx1250.tdm.async_wait(num_outstanding=2)
```

---

### triton.experimental.gluon.language.amd.gfx1250.tdm.make_tensor_descriptor

```python
make_tensor_descriptor(base: 'ttgl.tensor', shape: 'List[ttgl.constexpr | ttgl.tensor]', strides: 'List[ttgl.constexpr | ttgl.tensor]', block_shape: 'List[ttgl.constexpr]', layout: 'PaddedSharedLayout | SwizzledSharedLayout', _semantic=None) -> 'tensor_descriptor'
```

## make_tensor_descriptor


**`make_tensor_descriptor(base, shape, strides, block_shape, layout)`**

   Create a tensor descriptor object for TMA (Tensor Memory Accelerator) operations.

   Parameters
   ----------
   base : tensor
      Base pointer of the tensor in global memory. Must be a pointer type.
   shape : list of constexpr or tensor
      Shape of the tensor. Each dimension must be 1-5 total dimensions.
   strides : list of constexpr or tensor
      Strides of the tensor. Must have same length as `shape`.
   block_shape : list of constexpr
      Block shape of the tensor for TMA operations. Must have same length as `shape`.
   layout : PaddedSharedLayout or SwizzledSharedLayout
      The layout of the tensor in shared memory. For `SwizzledSharedLayout`, `max_phase` must be 1.

   Returns
   -------
   tensor_descriptor
      The created tensor descriptor object for use in TMA load/store operations.

   Notes
   -----
   - Number of dimensions must satisfy `1 <= ndim <= 5`.
   - `shape`, `strides`, and `block_shape` must all have the same length.
   - `base` must be a pointer type (`base.dtype` is `pointer_type`).
   - `layout` must be either `PaddedSharedLayout` or `SwizzledSharedLayout`.
   - For `SwizzledSharedLayout`, `max_phase` must equal 1.
   - Shape values are converted to i32, stride values to i64.
   - Padding option is set to "zero" by default.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.amd.gfx1250.tdm import make_tensor_descriptor
   from triton.experimental.gluon.language._layouts import PaddedSharedLayout

   @gluon.jit
   def kernel(base_ptr, shape, strides):
       # Create a tensor descriptor for TMA operations
       layout = PaddedSharedLayout(shape=[128, 128], padding=0)
       desc = make_tensor_descriptor(
           base=base_ptr,
           shape=[1024, 1024],
           strides=[1024, 1],
           block_shape=[128, 128],
           layout=layout
       )
       # Use descriptor in TMA load/store operations
       ...
```

---

### triton.experimental.gluon.language.amd.gfx1250.tdm.prefetch

```python
prefetch(src: 'tensor_descriptor', offsets: 'List[ttgl.constexpr | ttgl.tensor]', pred: 'bool' = True, speculative: 'bool' = False, _semantic=None) -> 'None'
```

## prefetch

Prefetches a block of tensor from global memory into L2 cache using AMD GFX1250 TDM.

### Parameters
src : tensor_descriptor
    The source tensor descriptor specifying the memory region to prefetch.
offsets : List[ttgl.constexpr | ttgl.tensor]
    The offsets from the base pointer in the tensor descriptor.
pred : bool, optional
    Predicate to enable or disable the prefetch. Defaults to `True`.
speculative : bool, optional
    Whether the prefetch is speculative. Defaults to `False`.

### Returns
None

### Notes
Speculative prefetches can generate more efficient assembly because they do not
require out-of-bounds checks. However, they are dropped by the hardware if their
virtual address translation is not cached. Therefore, `speculative` should only
be set to `True` if previous iterations have accessed the same virtual page
(e.g., column-major access patterns).

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(src_desc, offsets):
     # Prefetch tensor block into L2 cache
     ttgl.amd.gfx1250.tdm.prefetch(src_desc, offsets)

 @gluon.jit
 def kernel_speculative(src_desc, offsets):
     # Use speculative prefetch for known access patterns
     ttgl.amd.gfx1250.tdm.prefetch(src_desc, offsets, speculative=True)
```

---

### triton.experimental.gluon.language.amd.gfx1250.tdm.tensor_descriptor

```python
tensor_descriptor(handle: 'ir.value', shape: 'ttgl.tuple', strides: 'ttgl.tuple', type: 'tensor_descriptor_type') -> None
```

## class tensor_descriptor


**`tensor_descriptor(handle, shape, strides, type)`**

   A descriptor representing a tensor in global memory.

   Tensor descriptors encapsulate metadata required for asynchronous tensor data
   movement (TDM) operations on AMD GFX1250 architectures. They contain a handle
   to the global memory tensor, its shape, strides, and type information including
   block layout.

   Parameters
   ----------
   handle : ir.value
       IR value representing the tensor descriptor handle.
   shape : ttgl.tuple
       Tuple containing the tensor's shape dimensions.
   strides : ttgl.tuple
       Tuple containing the tensor's stride values.
   type : tensor_descriptor_type
       The type descriptor containing block type, shape type, strides type, and layout.

   Attributes
   ----------
   handle : ir.value
       The IR handle for the tensor descriptor.
   shape : ttgl.tuple
       The shape of the tensor as a tuple.
   strides : ttgl.tuple
       The strides of the tensor as a tuple.
   type : tensor_descriptor_type
       The tensor descriptor type object.

   Properties
   ----------
   block_type : ttgl.block_type
       The block type of the tensor descriptor.
   block_shape : tuple
       The shape of each memory block.
   dtype : dtype
       The element data type of the tensor.
   layout : PaddedSharedLayout | SwizzledSharedLayout
       The shared memory layout configuration.

   Notes
   -----
   Tensor descriptors are typically created using `make_tensor_descriptor()`
   rather than instantiated directly. They are used with asynchronous TDM operations
   such as `async_load()`, `async_store()`, `async_gather()`, and
   `async_scatter()` for efficient global-to-shared memory transfers.

   The descriptor supports 1 to 5 dimensions. Strides are 64-bit integers while
   shape dimensions are 32-bit integers.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.amd.gfx1250.tdm import tensor_descriptor

   # Tensor descriptors are typically created via make_tensor_descriptor
   @gluon.jit
   def kernel(...):
       desc = tdm.make_tensor_descriptor(
           base=global_ptr,
           shape=[height, width],
           strides=[stride_h, stride_w],
           block_shape=[block_h, block_w],
           layout=layout
       )
       # Use descriptor with async operations
       tdm.async_load(desc, offsets=[0, 0], dest=shared_mem)
```

---

### triton.experimental.gluon.language.amd.gfx1250.tdm.tensor_descriptor_type

```python
tensor_descriptor_type(block_type: 'ttgl.block_type', shape_type: 'ttgl.tuple_type', strides_type: 'ttgl.tuple_type', layout: 'PaddedSharedLayout | SwizzledSharedLayout') -> None
```

## class tensor_descriptor_type


**`tensor_descriptor_type(block_type, shape_type, strides_type, layout)`**

   The type for a tensor descriptor in AMD GFX1250 TDM operations.

   Represents the static type information for a tensor descriptor, including the
   block type, shape and stride types, and shared memory layout. This type is
   used to define tensor descriptors for asynchronous tensor memory operations
   on AMD GFX1250 hardware.

   Parameters
   ----------
   block_type : ttgl.block_type
       The block type defining the element type and block shape of the tensor.
   shape_type : ttgl.tuple_type
       The type for the tensor shape tuple (i32 values).
   strides_type : ttgl.tuple_type
       The type for the tensor strides tuple (i64 values).
   layout : PaddedSharedLayout | SwizzledSharedLayout
       The shared memory layout for the tensor block. Must be either
       `PaddedSharedLayout` or `SwizzledSharedLayout`.

   Returns
   -------
   tensor_descriptor_type
       A type object representing the tensor descriptor structure.

   Notes
   -----
   This is a type definition class, not a runtime value. Instances are used
   to construct `tensor_descriptor` values via `make_tensor_descriptor`.

   The tensor descriptor type encodes all static information needed for TDM
   operations, including element type, block dimensions, and memory layout.
   For GFX1250, `SwizzledSharedLayout` requires `max_phase == 1`.

   Tensor descriptors support 1 to 5 dimensions. Shape values are i32, while
   stride values are i64.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>> from triton.experimental.gluon.language._layouts import PaddedSharedLayout
   >>>
   >>> # Define block type for 128x64 f16 tensor
   >>> block_ty = ttgl.block_type(ttgl.float16, [128, 64])
   >>>
   >>> # Define shape and stride types (2D tensor)
   >>> shape_ty = ttgl.tuple_type([ttgl.int32, ttgl.int32])
   >>> strides_ty = ttgl.tuple_type([ttgl.int64, ttgl.int64])
   >>>
   >>> # Create tensor descriptor type with padded layout
   >>> layout = PaddedSharedLayout()
   >>> desc_ty = gluon.language.amd.gfx1250.tdm.tensor_descriptor_type(
   ...     block_ty, shape_ty, strides_ty, layout
   ... )
   >>>
   >>> print(desc_ty)
   tensor_descriptor<block<f16, [128, 64]>, PaddedSharedLayout()>

---


## triton.experimental.gluon.language.amd.rdna3

### triton.experimental.gluon.language.amd.rdna3.wmma

```python
wmma(a, b, acc, _semantic=None)
```

## triton.experimental.gluon.language.amd.rdna3.wmma


**`wmma(a, b, acc, _semantic=None)`**

    Computes matrix-multiplication of `a * b + acc` using AMD WMMA instruction on RDNA3 GPUs.

    Parameters
    ----------
    a : tensor
        The first operand tensor to be multiplied. Must have a `DotOperandLayout` with operand index 0.
    b : tensor
        The second operand tensor to be multiplied. Must have a `DotOperandLayout` with operand index 1.
    acc : tensor
        The accumulator tensor. Must have a `BlockedLayout` compatible with the output of the matrix multiplication.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Do not pass explicitly; this is set automatically by the `@gluon.jit` decorator.

    Returns
    -------
    tensor
        A tensor containing the result of `a @ b + acc`.

    Notes
    -----
    This operation uses AMD's Wave Matrix Multiply Accumulate (WMMA) instruction available on RDNA3 architecture GPUs.
    The input tensors must have layouts compatible with WMMA operations. Typically, `a` and `b` should be loaded
    from shared memory with appropriate `DotOperandLayout` configurations, and `acc` should have a matching
    `BlockedLayout`.

    This function is only available when targeting AMD RDNA3 GPUs. Attempting to use it on other hardware will result
    in compilation errors.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> from triton.experimental.gluon.language.amd import rdna3
    >>>
    >>> @gluon.jit
    ... def kernel(
    ...     a_ptr, b_ptr, c_ptr,
    ...     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ... ):
    ...     # Allocate shared memory
    ...     a_shared = ttgl.allocate_shared_memory(
    ...         ttgl.float16, (BLOCK_M, BLOCK_K),
    ...         ttgl.SharedLayout.make_blocked([1, 0], [BLOCK_M, BLOCK_K])
    ...     )
    ...     b_shared = ttgl.allocate_shared_memory(
    ...         ttgl.float16, (BLOCK_K, BLOCK_N),
    ...         ttgl.SharedLayout.make_blocked([0, 1], [BLOCK_K, BLOCK_N])
    ...     )
    ...     acc = ttgl.full((BLOCK_M, BLOCK_N), 0.0, ttgl.float16)
    ...     # Load from global to shared
    ...     # ... (loading code)
    ...     # Perform WMMA operation
    ...     result = rdna3.wmma(a, b, acc)
    ...     # Store result
    ...     # ... (storing code)

---


## triton.experimental.gluon.language.amd.rdna4

### triton.experimental.gluon.language.amd.rdna4.wmma

```python
wmma(a, b, acc, _semantic=None)
```

## ttgl.amd.rdna4.wmma


**`wmma(a, b, acc, _semantic=None)`**

   Computes matrix multiplication `a @ b + acc` using the AMD RDNA4 WMMA 
   (Wave Matrix Multiply Accumulate) instruction.

   Parameters
   ----------
   a : tensor
       First operand tensor with `DotOperandLayout` (operand index 0).
   b : tensor
       Second operand tensor with `DotOperandLayout` (operand index 1).
   acc : tensor
       Accumulator tensor with `BlockedLayout`. Must have compatible shape 
       and dtype for the matrix multiply-accumulate operation.
   _semantic : GluonSemantic, optional
       Internal semantic handler. Must be provided automatically by the 
       `@gluon.jit` decorator. Do not set manually.

   Returns
   -------
   tensor
       Result tensor containing `a @ b + acc` with the same layout and dtype 
       as `acc`.

   Notes
   -----
   This function is specific to AMD RDNA4 GPUs (GFX1250 and later). The WMMA 
   instruction provides hardware-accelerated matrix multiplication with 
   accumulation.

   Layout requirements:
   
   - `a` must have `DotOperandLayout` with `operand_index=0`
   - `b` must have `DotOperandLayout` with `operand_index=1`
   - `acc` must have `BlockedLayout`
   - All three tensors must share the same parent layout
   - Shapes must be compatible for matrix multiplication (2D or 3D batched)

   This is a low-level primitive. For most use cases, prefer higher-level 
   `ttgl.dot` operations when available.

   Must be called within a function decorated with `@gluon.jit`. The 
   `_semantic` parameter is injected automatically by the JIT compiler.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language import amd

   @gluon.jit
   def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
       pid = ttgl.program_id(0)
       
       # Allocate shared memory
       a_shared = ttgl.allocate_shared_memory(
           ttgl.float16, [BLOCK_M, BLOCK_K], 
           ttgl.SharedLayout.make_swizzled_layout(BLOCK_M, BLOCK_K)
       )
       b_shared = ttgl.allocate_shared_memory(
           ttgl.float16, [BLOCK_K, BLOCK_N],
           ttgl.SharedLayout.make_swizzled_layout(BLOCK_K, BLOCK_N)
       )
       
       # Load tiles from global memory
       offsets_m = ttgl.arange(0, BLOCK_M, layout=ttgl.AutoLayout())
       offsets_n = ttgl.arange(0, BLOCK_N, layout=ttgl.AutoLayout())
       offsets_k = ttgl.arange(0, BLOCK_K, layout=ttgl.AutoLayout())
       
       # Initialize accumulator
       acc = ttgl.full([BLOCK_M, BLOCK_N], 0.0, ttgl.float16, 
                      layout=ttgl.BlockedLayout([1, 1]))
       
       # Matrix multiply-accumulate loop
       for k in range(0, K, BLOCK_K):
           # Load tiles into shared memory (omitted for brevity)
           a_tile = a_shared.load(ttgl.DotOperandLayout.make_operand_layout(0))
           b_tile = b_shared.load(ttgl.DotOperandLayout.make_operand_layout(1))
           
           # WMMA operation
           acc = amd.rdna4.wmma(a_tile, b_tile, acc)
       
       # Store result
       # ... (store logic omitted)
```

---


## triton.experimental.gluon.language.nvidia.ampere

### triton.experimental.gluon.language.nvidia.ampere.mma_v2

```python
mma_v2(a, b, acc, input_precision=None, _semantic=None)
```

## mma_v2


**`mma_v2(a, b, acc, input_precision=None, _semantic=None)`**

   Perform a matrix multiply-accumulate operation using NVIDIA Ampere MMA version 2.0.

   This function executes a low-level tensor core MMA instruction on Ampere GPUs.
   Input tensors must have layouts compatible with the accumulator's MMA layout.

   Parameters
   ----------
   a : tensor
       First operand tensor with `DotOperandLayout`. Must have operand index 0
       and parent layout matching `acc`'s MMA layout.
   b : tensor
       Second operand tensor with `DotOperandLayout`. Must have operand index 1
       and parent layout matching `acc`'s MMA layout.
   acc : tensor
       Accumulator tensor with `NVMMADistributedLayout` version 2.0. The output
       tensor will have the same type as `acc`.
   input_precision : dtype, optional
       Precision for intermediate computation. If None, uses default precision.
   _semantic : GluonSemantic, optional
       Internal semantic handler. Must be provided by the JIT compiler.

   Returns
   -------
   tensor
       Result tensor with the same type and layout as `acc`.

   Raises
   ------
   AssertionError
       If tensor layouts do not meet MMA version 2.0 requirements.

   Notes
   -----
   This is a low-level primitive for explicit control over tensor core operations.
   Users should typically construct compatible layouts using Gluon layout APIs
   before calling this function.

   The MMA layout version must be exactly `[2, 0]` for Ampere tensor cores.
   Operand `a` must have operand index 0, and operand `b` must have operand
   index 1.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.ampere import mma_v2

   @gluon.jit
   def kernel(...):
       # Create tensors with compatible MMA layouts
       a = ...  # tensor with DotOperandLayout, operand_index=0
       b = ...  # tensor with DotOperandLayout, operand_index=1
       acc = ...  # tensor with NVMMADistributedLayout version 2.0

       result = mma_v2(a, b, acc)
```

---


## triton.experimental.gluon.language.nvidia.ampere.async_copy

### triton.experimental.gluon.language.nvidia.ampere.async_copy.async_copy_global_to_shared

```python
async_copy_global_to_shared(smem, pointer, mask=None, cache_modifier='', eviction_policy='', volatile=False, _semantic=None)
```

## async_copy_global_to_shared


**`async_copy_global_to_shared(smem, pointer, mask=None, cache_modifier='', eviction_policy='', volatile=False, _semantic=None)`**

   Asynchronously copy elements from global memory to shared memory.

   Parameters
   ----------
   smem : shared_memory_descriptor
       Destination shared memory descriptor.
   pointer : tensor
       Source pointer tensor.
   mask : tensor, optional
       Mask tensor for predicated loads. Defaults to `None`.
   cache_modifier : str, optional
       Cache modifier specifier (e.g., `.ca`, `.cg`). Defaults to `""`.
   eviction_policy : str, optional
       Eviction policy specifier (e.g., `.evict_normal`, `.evict_prioritize`). Defaults to `""`.
   volatile : bool, optional
       Whether the load is volatile. Defaults to `False`.

   Notes
   -----
   This operation initiates an asynchronous copy from global to shared memory on
   NVIDIA Ampere and later GPUs, allowing overlap of memory transfers with compute.
   The shapes of `smem` and `pointer` must match exactly.

   This is a low-level primitive that requires explicit management of shared memory
   descriptors and global pointers. Users should ensure proper synchronization
   (e.g., via `ttgl.barrier` or `ttgl.wait`) before accessing the copied data.

   Examples
   --------
```python
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl
    from triton.experimental.gluon.language import SharedLayout

    @gluon.jit
    def kernel(ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        # Allocate shared memory with explicit layout
        layout = SharedLayout((BLOCK_M, BLOCK_N), (16, 16))
        smem = ttgl.allocate_shared_memory(ttgl.float16, [BLOCK_M, BLOCK_N], layout)

        # Create global pointer
        offs_m = ttgl.arange(0, BLOCK_M, layout=ttgl.AutoLayout())
        offs_n = ttgl.arange(0, BLOCK_N, layout=ttgl.AutoLayout())
        ptrs = ttgl.make_block_pointer(ptr, [BLOCK_M, BLOCK_N], [offs_m, offs_n])

        # Async copy from global to shared
        ttgl.nvidia.ampere.async_copy.async_copy_global_to_shared(smem, ptrs)

        # Synchronize before using shared memory
        ttgl.barrier()

        # Load from shared memory for compute
        data = smem.load(layout=ttgl.DistributedLayout(...))
```

---

### triton.experimental.gluon.language.nvidia.ampere.async_copy.commit_group

```python
commit_group(_semantic=None)
```

## commit_group


**`commit_group(_semantic=None)`**

   Commit the current asynchronous copy group.

   This finalizes a set of asynchronous copy operations issued through
   asynchronous copy primitives (e.g., `async_copy()`). All copies
   issued before this call are grouped together and can be waited on
   as a unit using `wait_group()`.

   Parameters
   ----------
   _semantic : GluonSemantic, optional
       Internal semantic object passed by the JIT compiler. Users should not
       pass this argument directly when calling from a `@gluon.jit` kernel.

   Returns
   -------
   None

   Notes
   -----
   This function is typically called implicitly by the Gluon JIT compiler when
   using asynchronous copy operations. Direct calls are only needed in advanced
   use cases where explicit control over copy group boundaries is required.

   Asynchronous copies must be committed before they can be waited on using
   `wait_group()`. Multiple commit groups can be issued to create
   independent synchronization points.

   This operation is specific to NVIDIA Ampere architecture and later GPUs
   that support asynchronous copy instructions (Hopper, Blackwell).

   Examples
   --------
```python
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl

    @gluon.jit
    def async_copy_kernel(...):
        # Issue async copies to shared memory
        ttgl.nvidia.ampere.async_copy.async_copy(src_ptr, dst_desc, ...)
        
        # Commit the group before waiting
        ttgl.nvidia.ampere.async_copy.commit_group()
        
        # Wait for all copies in the group to complete
        ttgl.nvidia.ampere.async_copy.wait_group()
```

---

### triton.experimental.gluon.language.nvidia.ampere.async_copy.mbarrier_arrive

```python
mbarrier_arrive(mbarrier, increment_count=True, _semantic=None)
```

## mbarrier_arrive


**`mbarrier_arrive(mbarrier, increment_count=True, _semantic=None)`**

    Arrive on the mbarrier once all outstanding async copies are complete.

    Signals completion of async copy operations to the specified memory barrier.
    This synchronization primitive is used with NVIDIA Ampere and later GPU
    architectures for coordinating asynchronous memory transfers.

    Parameters
    ----------
    mbarrier : shared_memory_descriptor
        Barrier object to arrive on. Must be a shared memory descriptor
        allocated with `allocate_shared_memory()`.
    increment_count : bool, optional
        Whether to increment the arrival count. Defaults to `True`.
        If `False`, signals arrival without incrementing the counter.
    _semantic : GluonSemantic, optional
        Internal semantic handler. Do not set manually; this is provided
        automatically by the `@gluon.jit` decorator.

    Returns
    -------
    None

    Notes
    -----
    This function is part of the async copy API for NVIDIA Ampere+ GPUs.
    It must be called after initiating async copy operations to signal
    their completion to other warps or threads waiting on the barrier.

    The mbarrier must be properly initialized before use. Typical usage
    involves pairing `mbarrier_arrive()` with `mbarrier_wait()`
    for synchronization between producer and consumer warps.

    This operation is only available when using `@gluon.jit` decorated
    kernels, not standard `@triton.jit` kernels.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> from triton.experimental.gluon.language.nvidia.ampere.async_copy import mbarrier_arrive
    >>>
    >>> @gluon.jit
    ... def kernel(
    ...     mbarrier: ttgl.shared_memory_descriptor,
    ... ):
    ...     # Signal completion of async copies
    ...     mbarrier_arrive(mbarrier, increment_count=True)
    ...     # Other synchronization logic follows

---

### triton.experimental.gluon.language.nvidia.ampere.async_copy.wait_group

```python
wait_group(num_outstanding=0, _semantic=None)
```

Wait for outstanding asynchronous copy group operations to complete.

This primitive stalls the pipeline until the number of in-flight async copy groups
drops to the specified threshold. It is used to synchronize memory operations
initiated via asynchronous copy instructions (e.g., TMA) on supported hardware.

### Parameters
num_outstanding : int, optional
    The maximum number of async copy groups allowed to remain in-flight
    after the wait returns. Defaults to 0, which waits for all outstanding
    groups to complete.

### Returns
None

### Notes
This operation is specific to NVIDIA Ampere architecture and later
(including Hopper and Blackwell) when using asynchronous copy mechanisms.
It interacts directly with hardware status registers regarding Tensor Memory
Accelerator (TMA) or async copy queues.

Excessive use of `wait_group()` with `num_outstanding=0` may
serialize memory operations and reduce bandwidth utilization. Prefer
double-buffering or software pipelining patterns where possible to
overlap computation and memory transfers.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(...):
     # Issue async copy group 0
     ttgl.nvidia.ampere.async_copy.copy_global_to_shared(...)
     # Issue async copy group 1
     ttgl.nvidia.ampere.async_copy.copy_global_to_shared(...)
     # Wait until 0 groups are in-flight (all complete)
     ttgl.nvidia.ampere.async_copy.wait_group(num_outstanding=0)
     # Proceed with computation on shared memory
```

---


## triton.experimental.gluon.language.nvidia.ampere.mbarrier

### triton.experimental.gluon.language.nvidia.ampere.mbarrier.MBarrierLayout

```python
MBarrierLayout(cga_layout=None)
```

**`MBarrierLayout(cga_layout=None)`**

   Layout for mbarrier synchronization in Ampere and later architectures.

   Defines the memory layout for mbarrier objects in shared memory, supporting
   Cooperative Group Array (CGA) clustering for multi-CTA synchronization on
   NVIDIA Ampere and later GPUs.

   Parameters
   ----------
   cga_layout : List[List[int]], optional
       CGA layout bases. Defaults to `[]`.

   Notes
   -----
   Inherits from `SwizzledSharedLayout`. Requires NVIDIA Ampere architecture
   or later (SM80+). This layout is typically passed to
   `ttgl.allocate_shared_memory()` when allocating mbarriers.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   # Default layout
   layout = ttgl.nvidia.ampere.mbarrier.MBarrierLayout()

   # Multi-CTA layout for 4 CTAs
   layout = ttgl.nvidia.ampere.mbarrier.MBarrierLayout.multicta(4)

.. py:method:: multicta(num_ctas, two_cta=False)

   Create a multi-CTA mbarrier layout.

   Constructs a layout suitable for synchronizing across multiple CTAs using
   a single mbarrier object.

   Parameters
   ----------
   num_ctas : int
       Number of CTAs. Must be positive and a power of two.
   two_cta : bool, optional
       Whether the barrier should synchronize every other CTA. Defaults to
       ``False``.

   Returns
   -------
   MBarrierLayout
       The constructed layout.

   Notes
   -----
   Validates that ``num_ctas`` is positive and a power of two. If ``two_cta``
   is ``True``, ``num_ctas`` must be even. The layout bases are computed
   automatically to support the specified clustering mode.

   Examples
   --------
   .. code-block:: python

      import triton.experimental.gluon.language as ttgl

      # Layout for 8 CTAs
      layout = ttgl.nvidia.ampere.mbarrier.MBarrierLayout.multicta(8)

      # Layout for 8 CTAs in pairs (4 barriers)
      layout = ttgl.nvidia.ampere.mbarrier.MBarrierLayout.multicta(8, two_cta=True)
```

---

### triton.experimental.gluon.language.nvidia.ampere.mbarrier.allocate_mbarrier

```python
allocate_mbarrier(batch: triton.language.core.constexpr = None, two_ctas: triton.language.core.constexpr = False)
```

Allocate shared memory for a hardware memory barrier (mbarrier) on NVIDIA Ampere and later architectures.

### Parameters
batch : constexpr, optional
    If provided, allocates a batch of barriers with the specified size.
    The resulting shared memory shape is `[batch, num_elems]`. If None,
    the shape is `[num_elems]`. Defaults to None.
two_ctas : constexpr, bool
    If True, configures the barrier to synchronize every other CTA,
    effectively halving the participant count per barrier instance.
    Defaults to False.

### Returns
shared_memory_descriptor
    The allocated mbarrier object.

### Notes
Requires `num_ctas` to be a power of two. If `two_ctas` is True,
`num_ctas` must be even. The underlying type is `int64`. Layout is
handled via `MBarrierLayout.multicta()`.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.nvidia.ampere.mbarrier import allocate_mbarrier

 @gluon.jit
 def kernel(...):
     # Allocate a standard mbarrier for all CTAs
     bar = allocate_mbarrier()

     # Allocate a barrier synchronizing every other CTA
     bar_alt = allocate_mbarrier(two_ctas=True)

     # Allocate a batch of 4 barriers
     bar_batch = allocate_mbarrier(batch=4)
```

---

### triton.experimental.gluon.language.nvidia.ampere.mbarrier.arrive

```python
arrive(mbarrier, *, pred=True, _semantic=None)
```

**`arrive(mbarrier, *, pred=True, _semantic=None)`**

   Signal arrival at a memory barrier, incrementing the arrival count.

   Parameters
   ----------
   mbarrier : ttgl.shared_memory_descriptor
       The memory barrier object to signal. Must be allocated in shared memory.
   pred : bool or tensor, optional
       Predicate controlling the execution of the arrive instruction. If False,
       the operation is skipped. Defaults to True.
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually.

   Returns
   -------
   None

   Notes
   -----
   This operation is specific to NVIDIA Ampere architecture (and compatible with
   Hopper/Blackwell). It increments the barrier's arrival count by 1. Threads
   must arrive at the barrier before waiting on it using `ttgl.nvidia.ampere.mbarrier.wait()`.

   The predicate argument allows for conditional signaling, useful for control
   flow divergence within warps. The internal count increment is fixed to 1.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(mbarrier):
       # Signal arrival at the barrier
       ttgl.nvidia.ampere.mbarrier.arrive(mbarrier)
       # Wait for all threads to arrive
       ttgl.nvidia.ampere.mbarrier.wait(mbarrier, 1)
```

---

### triton.experimental.gluon.language.nvidia.ampere.mbarrier.init

```python
init(mbarrier, count, _semantic=None)
```

## triton.experimental.gluon.language.nvidia.ampere.mbarrier.init


**`init(mbarrier, count, _semantic=None)`**

   Initialize a memory barrier (mbarrier) in shared memory with a specified thread count.

   Parameters
   ----------
   mbarrier : shared_memory_descriptor
       The barrier object to initialize. Must be a shared memory descriptor allocated
       for mbarrier use on NVIDIA Ampere architecture GPUs.
   count : int
       The initial thread count for the barrier. Represents the number of threads
       expected to arrive at the barrier before it is released.
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not pass explicitly; automatically provided
       by the `@gluon.jit` decorator.

   Returns
   -------
   None

   Notes
   -----
   This function is specific to NVIDIA Ampere architecture (SM80) and later.
   The mbarrier is a hardware synchronization primitive that enables efficient
   thread coordination within a CTA. After initialization, threads can use
   `mbarrier.arrive()` and `mbarrier.wait()` to synchronize.

   The `count` parameter should match the number of threads that will participate
   in the barrier synchronization. Typical usage sets this to the number of
   threads in a warp or CTA.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.ampere import mbarrier

   @gluon.jit
   def kernel():
       # Allocate shared memory for mbarrier (128 bytes on Ampere)
       mb = ttgl.allocate_shared_memory(
           ttgl.int64, [128],
           ttgl.SharedLayout(128, 1)
       )
       # Initialize barrier with thread count of 128
       mbarrier.init(mb, 128)
       # ... use mbarrier.arrive/wait for synchronization ...
```

---

### triton.experimental.gluon.language.nvidia.ampere.mbarrier.invalidate

```python
invalidate(mbarrier, _semantic=None)
```

**`invalidate(mbarrier)`**

    Invalidate an mbarrier, resetting its state.

    Parameters
    ----------
    mbarrier : shared_memory_descriptor
        The barrier object to invalidate.

    Returns
    -------
    None

    Notes
    -----
    This operation is specific to NVIDIA Ampere architecture and later.
    It resets the completion count and state of the memory barrier,
    allowing it to be reused for subsequent synchronization operations.
    This is commonly used when managing TMA (Tensor Memory Accelerator)
    transfers or warp-group synchronization primitives.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> @gluon.jit
    ... def kernel(...):
    ...     # Allocate a mbarrier in shared memory
    ...     mbarrier = ttgl.nvidia.ampere.mbarrier.alloc(...)
    ...     # ... perform synchronization ...
    ...     # Invalidate to reset state for reuse
    ...     ttgl.nvidia.ampere.mbarrier.invalidate(mbarrier)

---

### triton.experimental.gluon.language.nvidia.ampere.mbarrier.wait

```python
wait(mbarrier, phase, pred=True, deps=(), _semantic=None)
```

Wait until the mbarrier completes a specified phase.

Blocks execution until the mbarrier reaches the given phase index. This is used
for synchronization between warps or thread blocks using hardware mbarriers.

### Parameters
mbarrier : shared_memory_descriptor
    The barrier object to wait on. Must be a shared memory descriptor with
    mbarrier semantics.
phase : int
    The phase index to wait for. The barrier must reach this phase before
    execution continues.
pred : bool, optional
    Predicate controlling execution. If False, the wait operation is skipped.
    Defaults to True.
deps : Sequence[shared_memory_descriptor], optional
    Dependent shared memory allocations that the barrier is waiting on. Used
    to track liveness of dependent allocations. Defaults to empty tuple.

### Returns
None

### Notes
This function is only available on NVIDIA Ampere and later architectures that
support hardware mbarriers. The phase parameter typically alternates between
0 and 1 for double-buffering patterns.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(mbarrier, phase):
     ttgl.nvidia.ampere.mbarrier.wait(mbarrier, phase)
```

---


## triton.experimental.gluon.language.nvidia.blackwell

### triton.experimental.gluon.language.nvidia.blackwell.TensorMemoryLayout

```python
TensorMemoryLayout(block: 'Tuple[int, int]', col_stride: 'int', cga_layout: 'List[List[int]]' = <factory>, two_ctas: 'bool' = False) -> None
```

## class triton.experimental.gluon.language.nvidia.blackwell.TensorMemoryLayout

Describes the layout for tensor memory in NVIDIA Blackwell architecture.

### Parameters
block : Tuple[int, int]
    Number of contiguous elements per row and column in a CTA. Must be a
    2-tuple of integers.
col_stride : int
    Number of 32-bit columns to advance between logically adjacent columns.
    Packed layouts use a stride of 1. Unpacked layouts use
    `32 / bitwidth`. Must be a power of two.
cga_layout : List[List[int]], optional
    CGA (Cooperative Group Array) layout bases. Each basis must be a
    2-element list. Defaults to empty list.
two_ctas : bool, optional
    Whether the layout is for two-CTA mode. Defaults to False.

### Raises
AssertionError
    If `block` is not length 2, if any basis in `cga_layout` is not
    length 2, or if `col_stride` is not a power of two.

### Notes
The `col_stride` parameter must be a power of two (e.g., 1, 2, 4, 8, ...).
This is enforced at construction time.

The `block` tuple specifies the tiling dimensions within a CTA, where
`block[0]` is the row dimension and `block[1]` is the column dimension.

CGA layout bases define how tensor memory is distributed across cooperative
groups. Each basis is a 2-element list representing strides in the row and
column dimensions.

This class is frozen (immutable) and uses value equality for comparisons.

### Examples
Create a packed tensor memory layout with 32x32 block tiling:

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel():
     layout = gluon.language.nvidia.blackwell.TensorMemoryLayout(
         block=(32, 32),
         col_stride=1,
     )
     # Use layout with allocate_tensor_memory or tensor_memory_descriptor

```
Create an unpacked layout for FP8 data (4 bits per element):

```python
 @gluon.jit
 def kernel():
     # FP8 unpacked: 32 / 4 = 8 columns per 32-bit word
     layout = gluon.language.nvidia.blackwell.TensorMemoryLayout(
         block=(64, 32),
         col_stride=8,
     )

```
Create a layout with CGA distribution across 2 CTAs:

```python
 @gluon.jit
 def kernel():
     layout = gluon.language.nvidia.blackwell.TensorMemoryLayout(
         block=(32, 32),
         col_stride=1,
         cga_layout=[[1, 0]],  # Distribute along row dimension
         two_ctas=True,
     )

```
### See Also
tensor_memory_descriptor : Create a tensor memory descriptor with a layout.
allocate_tensor_memory : Allocate tensor memory with a specified layout.
TensorMemoryScalesLayout : Layout for tensor memory scales.

---

### triton.experimental.gluon.language.nvidia.blackwell.TensorMemoryScalesLayout

```python
TensorMemoryScalesLayout(cga_layout: 'List[List[int]]' = <factory>) -> None
```

## class TensorMemoryScalesLayout

Describes the layout for tensor memory scales in Blackwell architecture.

### Parameters
cga_layout : list of list of int, optional
    CGA (Cooperative Group Array) layout bases. Each basis must be a 2-element
    list. Defaults to empty list.

### Notes
This layout type is used for scaled tensor memory operations on NVIDIA Blackwell
GPUs. All basis vectors in `cga_layout` must have exactly 2 elements.

The layout is immutable (frozen dataclass) and supports hashing for use in
caching and deduplication.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 
 # Create a simple scales layout with no CGA bases
 layout = gluon.language.nvidia.blackwell.TensorMemoryScalesLayout()
 
 # Create a layout with custom CGA bases
 layout = gluon.language.nvidia.blackwell.TensorMemoryScalesLayout(
     cga_layout=[[0, 1], [1, 0]]
 )
 
 # Use layout in tensor memory descriptor type
 desc_ty = ttgl.tensor_memory_descriptor_type(
     element_ty=ttgl.float16,
     shape=[128, 128],
     layout=layout,
     alloc_shape=[128, 128]
 )
```

---

### triton.experimental.gluon.language.nvidia.blackwell._TensorMemoryLinearLayout

```python
_TensorMemoryLinearLayout(rows: 'List[List[int]]', cols: 'List[List[int]]', shape: 'List[int]') -> None
```

## class _TensorMemoryLinearLayout

Print-only linear layout for Tensor Memory (TMEM) on Blackwell architecture.

This layout class is intended for debugging and inspection purposes only. It
cannot be materialized to IR and will raise an error if used in kernel
execution.

### Parameters
rows : List[List[int]]
    Row basis vectors mapping logical rows to physical dimensions.
cols : List[List[int]]
    Column basis vectors mapping logical columns to physical dimensions.
shape : List[int]
    The logical shape of the tensor memory region.

### Notes
This layout is **print-only** and does not support IR materialization.
Calling `_to_ir()` will raise a `RuntimeError`. Use
`TensorMemoryLayout` or `TensorMemoryScalesLayout` for
executable kernels.

The layout maps row/column indices to dimension 0/dimension 1 of the tensor
memory address space.

### Examples

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 # Create a print-only layout for inspection
 layout = gluon.language.nvidia.blackwell._TensorMemoryLinearLayout(
     rows=[[0, 1]],
     cols=[[1, 0]],
     shape=[128, 256]
 )

 # This will print layout information
 print(layout)

 # This will raise an error - cannot be used in kernels
 # layout._to_ir(builder)  # RuntimeError
```

---

### triton.experimental.gluon.language.nvidia.blackwell.allocate_tensor_memory

```python
allocate_tensor_memory(element_ty, shape, layout, value=None, _semantic=None)
```

## allocate_tensor_memory


**`allocate_tensor_memory(element_ty, shape, layout, value=None, _semantic=None)`**

   Allocate tensor memory on NVIDIA Blackwell GPUs.

   Parameters
   ----------
   element_ty : dtype
       The element data type of the tensor memory.
   shape : Sequence[int]
       The shape of the tensor memory descriptor.
   layout : TensorMemoryLayout
       The layout specification for the tensor memory.
   value : tensor, optional
       Initial tensor to copy into the allocated memory. Defaults to `None`.
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually.

   Returns
   -------
   tensor_memory_descriptor
       Descriptor for the allocated tensor memory.

   Notes
   -----
   This function is specific to NVIDIA Blackwell architecture (compute capability 90+).
   Tensor memory provides high-bandwidth access for tensor operations and is distinct
   from shared memory. Use `triton.experimental.gluon.language.allocate_shared_memory()`
   for shared memory allocations.

   The returned descriptor can be used with tensor memory load/store operations
   in subsequent kernel code.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.blackwell import allocate_tensor_memory
   from triton.experimental.gluon.language._layouts import TensorMemoryLayout

   @gluon.jit
   def kernel():
       # Allocate tensor memory for a 128x128 float32 tensor
       desc = allocate_tensor_memory(
           ttgl.float32,
           (128, 128),
           TensorMemoryLayout()
       )
```

---

### triton.experimental.gluon.language.nvidia.blackwell.fence_async_shared

```python
fence_async_shared(cluster=False, _semantic=None)
```

## fence_async_shared


.. autofunction:: fence_async_shared

.. rubric:: API Documentation

**`fence_async_shared(cluster=False, _semantic=None)`**

   Issue a fence to complete asynchronous shared memory operations.

   Inserts a hardware fence instruction that ensures all prior asynchronous
   shared memory operations are visible to subsequent memory operations.
   Required for correct synchronization when using TMA (Tensor Memory
   Accelerator) or other async shared memory primitives on NVIDIA Blackwell
   architectures.

   Parameters
   ----------
   cluster : bool, optional
       Whether to fence across the CTA cluster. When `True`, synchronizes
       across all CTAs in the cluster. When `False`, synchronizes within
       the current CTA only. Default is `False`.
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually; this is provided
       automatically by the Gluon JIT compiler.

   Returns
   -------
   None

   Notes
   -----
   This function is specific to NVIDIA Blackwell (SM 100) architectures.
   On earlier architectures, this operation may be a no-op or emit different
   fence instructions.

   The fence ensures memory ordering guarantees required for correct
   execution of asynchronous copy operations. Failure to issue appropriate
   fences may result in undefined behavior or data races.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>>
   >>> @gluon.jit
   >>> def kernel():
   ...     # Issue async shared memory operations
   ...     # ...
   ...     # Fence to ensure completion before subsequent accesses
   ...     ttgl.nvidia.blackwell.fence_async_shared()
   ...     # Safe to access shared memory now
   ...
   >>> # Fence across cluster when using multi-CTA kernels
   >>> @gluon.jit
   >>> def cluster_kernel():
   ...     # ... async operations ...
   ...     ttgl.nvidia.blackwell.fence_async_shared(cluster=True)

---

### triton.experimental.gluon.language.nvidia.blackwell.mma_v2

```python
mma_v2(a, b, acc, input_precision=None, _semantic=None)
```

## mma_v2


**`mma_v2(a, b, acc, input_precision=None, _semantic=None)`**

   Perform a matrix multiply-accumulate operation using NVIDIA Blackwell MMA v2 hardware instructions.

   Parameters
   ----------
   a : tensor
       First operand tensor with `DotOperandLayout` (operand index 0).
   b : tensor
       Second operand tensor with `DotOperandLayout` (operand index 1).
   acc : tensor
       Accumulator tensor with `NVMMADistributedLayout` version 2.0.
   input_precision : dtype, optional
       Precision for intermediate computation. If None, uses accumulator dtype.
   _semantic : GluonSemantic, optional
       Internal semantic context (automatically provided by JIT).

   Returns
   -------
   tensor
       Result tensor with the same type and layout as `acc`.

   Notes
   -----
   This function emits NVIDIA Blackwell MMA v2 instructions (SM90+). Strict layout
   requirements must be satisfied:

   * `acc` must have an `NVMMADistributedLayout` with version `[2, 0]`
   * `a` and `b` must have `DotOperandLayout` with matching parent layout
   * `a` operand index must be 0, `b` operand index must be 1
   * All tensors must have compatible shapes for matrix multiplication

   This is a low-level primitive for explicit GPU programming. Prefer higher-level
   `triton.experimental.gluon.language.dot()` when possible.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.blackwell import mma_v2

   @gluon.jit
   def kernel(ACC, A, B):
       # Load tensors with appropriate layouts
       a = ttgl.load(A)  # DotOperandLayout, operand_index=0
       b = ttgl.load(B)  # DotOperandLayout, operand_index=1
       acc = ttgl.load(ACC)  # NVMMADistributedLayout v2.0

       # Perform MMA v2 operation
       result = mma_v2(a, b, acc)

       ttgl.store(ACC, result)
```

---

### triton.experimental.gluon.language.nvidia.blackwell.tensor_memory_descriptor

```python
tensor_memory_descriptor(handle, element_ty, shape, layout, alloc_shape)
```

## class tensor_memory_descriptor


.. autoclass:: tensor_memory_descriptor

   Represents a tensor memory descriptor handle for Tensor Core Gen5 (TCGen5)
   operations on NVIDIA Blackwell architecture. This class provides explicit
   control over tensor memory allocation, load/store operations, and reduction
   primitives for high-performance GPU computing.

   Parameters
   ----------
   handle : ir.value
       Low-level IR handle for the tensor memory descriptor.
   element_ty : dtype
       The element data type of the tensor (e.g., `ttgl.float16`, `ttgl.bfloat16`).
   shape : Sequence[int]
       The logical shape of the tensor in memory.
   layout : TensorMemoryLayout or TensorMemoryScalesLayout
       The memory layout describing how tensor elements are distributed across
       warps and CTAs.
   alloc_shape : Sequence[int]
       The allocated shape, which may differ from logical shape for padding or
       alignment purposes.

   Attributes
   ----------
   dtype : dtype
       The element data type of the tensor.
   shape : List[int]
       The logical shape of the tensor.
   rank : int
       The number of dimensions (length of shape).
   layout : TensorMemoryLayout or TensorMemoryScalesLayout
       The memory layout configuration.

   Methods
   -------
   get_reg_layout(num_warps=None, instr_variant="32x32b")
       Return the register layout used to access this tensor memory descriptor.
   load(layout=None)
       Load a tensor from tensor memory into registers.
   load_min(layout=None, abs=False, propagate_nan=PROPAGATE_NAN.NONE)
       Load with MIN reduction along the N-dimension.
   load_max(layout=None, abs=False, propagate_nan=PROPAGATE_NAN.NONE)
       Load with MAX reduction along the N-dimension.
   store(value, pred=True)
       Store a tensor from registers into tensor memory.
   slice(start, length)
       Create a slice along the last dimension.
   index(index)
       Create a subview by indexing the first dimension.

   Notes
   -----
   Tensor memory descriptors are used with TCGen5 MMA instructions on NVIDIA
   Blackwell GPUs (compute capability 9.0+). They provide explicit control over
   tensor memory layout, enabling fine-grained optimization of memory access
   patterns for matrix multiplication workloads.

   The descriptor handle is managed by the Gluon compiler and should not be
   manipulated directly. Use the provided methods for safe memory operations.

   Examples
   --------
   Allocate tensor memory and perform load/store operations:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.blackwell import (
       allocate_tensor_memory, TensorMemoryLayout
   )

   @gluon.jit
   def kernel():
       # Define tensor memory layout: 128x128 block, stride 1
       layout = TensorMemoryLayout(block=(128, 128), col_stride=1)
       
       # Allocate tensor memory for 1024x1024 float16 matrix
       tmem = allocate_tensor_memory(
           element_ty=ttgl.float16,
           shape=[1024, 1024],
           layout=layout
       )
       
       # Load tensor from tensor memory to registers
       tensor = tmem.load()
       
       # Perform computation...
       
       # Store result back to tensor memory
       tmem.store(tensor)

Use reduction load operations:

.. code-block:: python

   @gluon.jit
   def reduction_kernel():
       layout = TensorMemoryLayout(block=(64, 64), col_stride=1)
       tmem = allocate_tensor_memory(
           element_ty=ttgl.float16,
           shape=[512, 512],
           layout=layout
       )
       
       # Load with MIN reduction along N-dimension
       tensor, reduced = tmem.load_min()
       # reduced contains min values along axis 1

Create tensor memory slices and views:

.. code-block:: python

   @gluon.jit
   def sliced_kernel():
       layout = TensorMemoryLayout(block=(128, 128), col_stride=1)
       tmem = allocate_tensor_memory(
           element_ty=ttgl.float16,
           shape=[1024, 1024],
           layout=layout
       )
       
       # Slice last dimension (columns 0-256)
       sliced = tmem.slice(start=0, length=256)
       
       # Index first dimension (row 5)
       indexed = tmem.index(5)

See Also
--------
allocate_tensor_memory : Allocate new tensor memory
TensorMemoryLayout : Define tensor memory layout parameters
tcgen05_mma : Execute Tensor Core Gen5 matrix multiplication
triton.experimental.gluon.language.distributed_type : Create distributed tensor types
```

---

### triton.experimental.gluon.language.nvidia.blackwell.tensor_memory_descriptor_type

```python
tensor_memory_descriptor_type(element_ty, shape, layout, alloc_shape)
```

## tensor_memory_descriptor_type


**`tensor_memory_descriptor_type(element_ty, shape, layout, alloc_shape)`**

   Type descriptor for tensor memory in NVIDIA Blackwell architecture.

   Represents the type of a tensor memory descriptor used for Tensor Core Gen5
   (tcgen05) operations. Encapsulates element type, tensor shape, memory layout,
   and allocation shape for tensor memory regions.

   Parameters
   ----------
   element_ty : dtype
       The element data type of the tensor (e.g., `ttgl.float16`, `ttgl.bfloat16`).
   shape : Sequence[int]
       The logical shape of the tensor as a sequence of integers.
   layout : TensorMemoryLayout or TensorMemoryScalesLayout
       The memory layout describing how elements are arranged in tensor memory.
       Must be an instance of `TensorMemoryLayout` or `TensorMemoryScalesLayout`.
   alloc_shape : Sequence[int]
       The allocation shape, which may differ from the logical shape for
       memory allocation purposes.

   Attributes
   ----------
   element_ty : dtype
       The element data type.
   shape : Tuple[int, ...]
       The tensor shape.
   layout : TensorMemoryLayout or TensorMemoryScalesLayout
       The memory layout configuration.
   alloc_shape : Tuple[int, ...]
       The allocation shape.

   Methods
   -------
   get_reg_layout(num_warps=None, instr_variant="32x32b")
       Return a DistributedLinearLayout compatible with TMEM load/store
       instructions for this descriptor type.
   to_ir(builder)
       Convert the type descriptor to IR representation.
   mangle()
       Return a mangled string representation for type identification.

   Notes
   -----
   This type is used internally by Gluon to represent tensor memory descriptors
   for Blackwell GPU architectures. Users typically interact with
   `tensor_memory_descriptor` instances rather than constructing this type
   directly.

   The layout parameter must specify either a `TensorMemoryLayout` for regular
   tensor memory or `TensorMemoryScalesLayout` for scale factors in scaled
   MMA operations.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.blackwell import (
       tensor_memory_descriptor_type,
       TensorMemoryLayout,
   )

   # Create a tensor memory layout for 32x32 blocks
   layout = TensorMemoryLayout(block=(32, 32), col_stride=1)

   # Create a type descriptor for a 128x128 FP16 tensor
   ty = tensor_memory_descriptor_type(
       element_ty=ttgl.float16,
       shape=(128, 128),
       layout=layout,
       alloc_shape=(128, 128),
   )

   # Get the register layout for 4 warps with 32x32b instruction variant
   reg_layout = ty.get_reg_layout(num_warps=4, instr_variant="32x32b")

See Also
--------
tensor_memory_descriptor : The runtime handle for tensor memory regions.
TensorMemoryLayout : Layout configuration for tensor memory.
TensorMemoryScalesLayout : Layout configuration for tensor memory scales.
allocate_tensor_memory : Function to allocate tensor memory with this type.
```

---


## triton.experimental.gluon.language.nvidia.blackwell.clc

### triton.experimental.gluon.language.nvidia.blackwell.clc.clc_result

```python
clc_result(handle)
```

**`clc_result(handle)`**

   CLC response loaded into registers.

   Represents the result of a Cluster Launch Control (CLC) operation on NVIDIA
   Blackwell GPUs. The response is loaded directly into registers, allowing
   efficient querying of cancellation status and program IDs without re-reading
   shared memory.

   Parameters
   ----------
   handle : ir.value
       Internal IR handle representing the CLC result value. Typically obtained
       via `triton.experimental.gluon.language.nvidia.blackwell.clc.load_result()`.

   Notes
   -----
   Only supported on NVIDIA Blackwell architectures (SM100+). This class provides
   methods to inspect the outcome of a `try_cancel()` operation.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(...):
       # Assume shared_mem_ptr is a shared_memory_descriptor
       result = ttgl.nvidia.blackwell.clc.load_result(shared_mem_ptr)
       if result.is_canceled():
           pid_x = result.program_id(0)
```

---

### triton.experimental.gluon.language.nvidia.blackwell.clc.load_result

```python
load_result(src, _semantic=None)
```

## load_result

Load the CLC (Cluster Launch Coordinator) response from shared memory into registers.

### Parameters
src : shared_memory_descriptor
    The int64x2 CLC response buffer in shared memory.

### Returns
CLCResult
    Object with `is_canceled()` and `get_first_ctaid(dim)` methods for
    querying the CLC response status and CTA ID information.

### Notes
This function is specific to NVIDIA Blackwell architecture and requires
a valid CLC response buffer allocated in shared memory. The returned
`CLCResult` object provides methods to inspect whether the cluster
launch was canceled and to retrieve the first CTA ID for a given
dimension.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel(clc_buf):
     # Load CLC response from shared memory
     result = ttgl.nvidia.blackwell.clc.load_result(clc_buf)
     
     # Check if launch was canceled
     if result.is_canceled():
         return
     
     # Get first CTA ID for dimension 0
     first_ctaid = result.get_first_ctaid(0)
```

---

### triton.experimental.gluon.language.nvidia.blackwell.clc.try_cancel

```python
try_cancel(result: 'shared_memory_descriptor', barrier, multicast=False, _semantic=None)
```

**`try_cancel(result, barrier, multicast=False)`**

   Issue a CLC try_cancel request to atomically cancel a pending cluster launch.

   Parameters
   ----------
   result : shared_memory_descriptor
      16-byte aligned int64x2 shared memory for the response.
   barrier : shared_memory_descriptor
      8-byte aligned mbarrier for completion signaling.
   multicast : bool, optional
      If True, broadcast result to all CTAs in cluster (default is False).

   Returns
   -------
   None

   Notes
   -----
   Only supported on SM100+ (Blackwell) architectures.

   The `result` buffer must be allocated with sufficient alignment (16 bytes)
   and shape to hold an int64x2 response. The `barrier` must be a valid
   mbarrier initialized for completion signaling.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(...):
       # Allocate shared memory for result (int64x2) and barrier
       result = ttgl.allocate_shared_memory(...)
       barrier = ttgl.allocate_shared_memory(...)
       ttgl.nvidia.blackwell.clc.try_cancel(result, barrier, multicast=True)
```

---


## triton.experimental.gluon.language.nvidia.blackwell.tma

### triton.experimental.gluon.language.nvidia.blackwell.tma.async_copy_global_to_shared

```python
async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, multicast=False, _semantic=None)
```

## async_copy_global_to_shared


**`async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, multicast=False)`**

   Asynchronously copy data from global memory to shared memory using Tensor Memory Accelerator (TMA).

   This operation initiates an async copy from a global memory tensor descriptor to a shared memory descriptor. The copy is non-blocking and synchronizes via the provided barrier. Available on NVIDIA Blackwell architectures.

   Parameters
   ----------
   tensor_desc : tensor_descriptor
       Tiled tensor descriptor for the source data in global memory.
   coord : tensor or sequence of int
       Coordinates specifying the tile location in the source tensor. Each coordinate corresponds to a dimension of the tensor descriptor.
   barrier : barrier
       Barrier object for synchronization. The barrier tracks completion of the async copy operation.
   result : shared_memory_descriptor
       Destination shared memory descriptor where data will be copied.
   pred : tensor or bool, optional
       Predicate for conditional execution. If False, the copy is not performed. Default is True.
   multicast : bool, optional
       Enable multicast mode for copying to multiple CTAs. Default is False.

   Returns
   -------
   None

   Notes
   -----
   This is a low-level TMA primitive that requires explicit management of synchronization via barriers. The operation returns immediately without waiting for completion; use `barrier_wait()` or related primitives to ensure the copy completes before accessing the shared memory.

   When `multicast=True`, the copy operation can target multiple CTAs within a cluster, enabling efficient data distribution.

   Coordinate values must be aligned to the tensor descriptor's tile shape. Misaligned coordinates may result in undefined behavior.

   This function is only available on NVIDIA Blackwell GPUs (compute capability >= 90) with TMA support enabled.

   Examples
   --------
   >>> import triton.experimental.gluon as gluon
   >>> import triton.experimental.gluon.language as ttgl
   >>> from triton.experimental.gluon.language.nvidia.blackwell import tma
   >>>
   >>> @gluon.jit
   >>> def kernel(
   >>>     global_desc,
   >>>     shared_mem,
   >>>     barrier,
   >>>     BLOCK_M: tl.constexpr,
   >>>     BLOCK_N: tl.constexpr,
   >>> ):
   >>>     # Create coordinate tensor for tile position
   >>>     coord = ttgl.arange(0, 2, layout=ttgl.AutoLayout())
   >>>     coord = ttgl.expand_dims(coord, axis=0)
   >>>     coord = coord + ttgl.full((1, 2), [0, 0], dtype=ttgl.int32)
   >>>
   >>>     # Initiate async copy from global to shared memory
   >>>     tma.async_copy_global_to_shared(
   >>>         global_desc,
   >>>         coord,
   >>>         barrier,
   >>>         shared_mem,
   >>>     )
   >>>
   >>>     # Wait for copy completion before accessing shared_mem
   >>>     barrier.wait()
   >>>
   >>>     # Now safe to use shared_mem
   >>>     data = shared_mem.load(layout=ttgl.BlockedLayout(...))

---

### triton.experimental.gluon.language.nvidia.blackwell.tma.async_copy_shared_to_global

```python
async_copy_shared_to_global(tensor_desc, coord, src, _semantic=None)
```

## triton.experimental.gluon.language.nvidia.blackwell.tma.async_copy_shared_to_global


**`async_copy_shared_to_global(tensor_desc, coord, src, _semantic=None)`**

   Initiate an asynchronous TMA copy from shared memory to global memory on NVIDIA Blackwell GPUs.

   Parameters
   ----------
   tensor_desc : shared_memory_descriptor
       Tensor descriptor specifying the global memory destination, including element type, shape, and layout.
   coord : tensor or sequence of int
       Coordinates defining the region of global memory to write to. Must match the rank of `tensor_desc`.
   src : tensor
       Source tensor in shared memory containing data to copy. Must be compatible with `tensor_desc` element type.
   _semantic : GluonSemantic, optional
       Internal semantic object. Do not pass explicitly; handled automatically by the `@gluon.jit` decorator.

   Returns
   -------
   None

   Notes
   -----
   This function is specific to NVIDIA Blackwell architecture (compute capability 10.x) and requires TMA (Tensor Memory Accelerator) support.

   The copy is asynchronous - execution continues immediately without waiting for the transfer to complete. Use appropriate barriers or wait operations to ensure completion before accessing the data.

   When `enable_iisan` is enabled, alignment checks are performed on the innermost coordinate to ensure correct memory access patterns.

   Coordinates are converted to IR values without requiring 64-bit integers.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.blackwell import tma

   @gluon.jit
   def kernel(
       global_ptr,
       shared_desc,
   ):
       # Create tensor descriptor for global memory
       tensor_desc = tma.make_tensor_descriptor(
           global_ptr,
           shape=[128, 128],
           dtype=ttgl.float16,
           block_shape=[64, 64],
       )

       # Load data into shared memory
       src = shared_desc.load(ttgl.AutoLayout())

       # Define coordinates for the copy
       coord = [0, 0]

       # Initiate async copy from shared to global
       tma.async_copy_shared_to_global(tensor_desc, coord, src)

       # Issue TMA wait to ensure copy completion
       tma.wait_copy_pending()
```

---

### triton.experimental.gluon.language.nvidia.blackwell.tma.async_gather

```python
async_gather(tensor_desc, x_offsets, y_offset, barrier, result, pred=True, _semantic=None)
```

## async_gather


.. autofunction:: async_gather

Asynchronously gather elements from global memory to shared memory using TMA.

### Parameters
tensor_desc : tensor_descriptor
    The tensor descriptor specifying the global memory source.
x_offsets : tensor
    1D tensor of X offsets for the gather operation.
y_offset : int
    Scalar Y offset for the gather operation.
barrier : shared_memory_descriptor
    Barrier that will be signaled when the operation is complete.
result : tensor_memory_descriptor
    Result shared memory, must have `NVMMASharedLayout`.
pred : bool, optional
    Scalar predicate. Operation is skipped if predicate is False. Defaults to True.

### Returns
None
    This function performs an asynchronous memory operation and does not return a value.

### Notes
This operation is specific to NVIDIA Blackwell architecture and requires TMA (Tensor Memory Accelerator) hardware support. The gather operation asynchronously copies elements from global memory to shared memory, allowing compute to proceed while the transfer is in progress.

The `result` shared memory must be allocated with `NVMMASharedLayout` to ensure compatibility with TMA operations. The `barrier` parameter is used for synchronization - threads can wait on the barrier to ensure the async operation has completed before accessing the data.

When `pred=False`, the operation is skipped entirely. The predicate is evaluated per thread block.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.nvidia.blackwell import tma

 @gluon.jit
 def kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
     # Allocate shared memory with NVMMASharedLayout
     smem = ttgl.allocate_shared_memory(
         element_ty=ttgl.float16,
         shape=[BLOCK_SIZE, BLOCK_SIZE],
         layout=NVMMASharedLayout(...)
     )
     
     # Create barrier for synchronization
     barrier = ttgl.shared_memory_descriptor(...)
     
     # Create tensor descriptor for global memory
     tensor_desc = tma.make_tensor_descriptor(x_ptr, ...)
     
     # Generate x offsets
     x_offsets = ttgl.arange(0, BLOCK_SIZE, layout=...)
     
     # Async gather from global to shared memory
     tma.async_gather(
         tensor_desc=tensor_desc,
         x_offsets=x_offsets,
         y_offset=0,
         barrier=barrier,
         result=smem,
         pred=True
     )
     
     # Wait for async operation to complete
     ttgl.barrier()
     
     # Now safe to use data in smem
     data = smem.load(layout=...)
```

---

### triton.experimental.gluon.language.nvidia.blackwell.tma.async_scatter

```python
async_scatter(tensor_desc, x_offsets, y_offset, src, _semantic=None)
```

## async_scatter

**`triton.experimental.gluon.language.nvidia.blackwell.tma.async_scatter(tensor_desc, x_offsets, y_offset, src, _semantic=None)`**

   Asynchronously scatter elements from shared memory to global memory using TMA.

   Initiates an asynchronous scatter operation that transfers data from shared memory
   to global memory via the Tensor Memory Accelerator (TMA) on NVIDIA Blackwell GPUs.
   This operation is non-blocking and returns immediately while the transfer proceeds
   in the background.

   Parameters
   ----------
   tensor_desc : tensor_descriptor
       The tensor descriptor specifying the global memory destination.
   x_offsets : tensor
       1D tensor of X offsets specifying which columns to scatter to.
   y_offset : int
       Scalar Y offset specifying the row coordinate for the scatter operation.
   src : tensor_memory_descriptor
       The source data in shared memory. Must be in `NVMMASharedLayout`.
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually.

   Returns
   -------
   None
       This operation performs an asynchronous side effect.

   Notes
   -----
   This function is specific to NVIDIA Blackwell architecture and requires TMA
   support. The operation is non-blocking - use appropriate barriers or waits
   to ensure completion before accessing the scattered data.

   The `src` tensor must be allocated with `NVMMASharedLayout` for TMA
   compatibility. When `enable_iisan` is enabled, alignment and non-negative
   checks are emitted for the offsets.

   Examples
   --------
```python
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl
    from triton.experimental.gluon.language.nvidia.blackwell import tma

    @gluon.jit
    def kernel(tensor_desc, src, x_offsets, y_offset):
        tma.async_scatter(tensor_desc, x_offsets, y_offset, src)
        # Issue a barrier to ensure scatter completion before subsequent access
        ttgl.barrier()
```

---

### triton.experimental.gluon.language.nvidia.blackwell.tma.make_tensor_descriptor

```python
make_tensor_descriptor(base: 'ttgl.tensor', shape: 'List[ttgl.tensor]', strides: 'List[ttgl.tensor]', block_shape: 'List[ttgl.constexpr]', layout: 'NVMMASharedLayout', padding_option='zero', _semantic=None) -> 'tensor_descriptor'
```

## make_tensor_descriptor


**`make_tensor_descriptor(base, shape, strides, block_shape, layout, padding_option='zero')`**

   Create a tensor descriptor for TMA (Tensor Memory Accelerator) operations on NVIDIA Blackwell GPUs.

   Parameters
   ----------
   base : tensor
      Base pointer to the tensor data in global memory. Must be a pointer type.
   shape : List[tensor]
      List of tensors specifying the logical shape of the tensor (one per dimension).
   strides : List[tensor]
      List of tensors specifying the strides for each dimension. The last stride must be 1.
   block_shape : List[constexpr]
      List of compile-time constants specifying the block shape for TMA operations (one per dimension).
   layout : NVMMASharedLayout
      The shared memory layout for the tensor descriptor.
   padding_option : str, optional
      Padding option for out-of-bounds access. Either `'zero'` (default) or `'nan'`.
      Note: `'nan'` is not supported for integer blocks.

   Returns
   -------
   tensor_descriptor
      A tensor descriptor object that can be used with TMA load/store operations.

   Raises
   ------
   ValueError
      If the number of dimensions is not between 1 and 5, if strides/block_shape lengths
      don't match the shape, if the last dimension block size is less than 16 bytes,
      if the last stride is not 1, or if `'nan'` padding is used with integer blocks.

   Notes
   -----
   The tensor descriptor enables efficient async memory operations via TMA on Blackwell
   GPUs. Key constraints:

   * Number of dimensions must be 1-5
   * The last dimension block size must be at least 16 bytes (e.g., 16 elements for int8,
     8 for int16, 4 for int32, etc.)
   * The last stride must be 1 (contiguous in the last dimension)
   * Block shape must be statically known at compile time

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.blackwell.tma import make_tensor_descriptor
   from triton.experimental.gluon.language._layouts import NVMMASharedLayout

   @gluon.jit
   def kernel(...):
      # Create a 2D tensor descriptor for TMA operations
      base = ttgl.make_tensor_ptr(...)
      shape = [ttgl.full((), 1024, ttgl.int32), ttgl.full((), 512, ttgl.int32)]
      strides = [ttgl.full((), 512, ttgl.int64), ttgl.full((), 1, ttgl.int64)]
      block_shape = [ttgl.constexpr(64), ttgl.constexpr(64)]
      layout = NVMMASharedLayout(...)

      desc = make_tensor_descriptor(
         base=base,
         shape=shape,
         strides=strides,
         block_shape=block_shape,
         layout=layout,
         padding_option='zero'
      )

      # Use desc with TMA load/store operations
      ...
```

---

### triton.experimental.gluon.language.nvidia.blackwell.tma.store_wait

```python
store_wait(pendings, _semantic=None)
```

**`store_wait(pendings, _semantic=None)`**

Wait for pending asynchronous Tensor Memory Accelerator (TMA) store operations to complete.

Inserts a barrier that stalls execution until all outstanding asynchronous store
transactions tracked by `pendings` have finished. This ensures data visibility
and consistency before proceeding.

### Parameters
pendings : int or tensor
    The pending transaction count or token returned by asynchronous store operations.
    Specifies the number of pending transactions to wait for.
_semantic : GluonSemantic, optional
    Internal argument used by the Gluon JIT compiler. Do not provide manually.

### Returns
None

### Notes
This function is specific to NVIDIA Blackwell architecture and requires TMA to be
configured appropriately. It must be called within a `@gluon.jit()` decorated
kernel.

### Examples
```python
 import triton.experimental.gluon as gluon
 from triton.experimental.gluon.language.nvidia.blackwell import tma

 @gluon.jit
 def kernel(ptr, pendings):
     # Wait for all pending stores to complete
     tma.store_wait(pendings)
```

---

### triton.experimental.gluon.language.nvidia.blackwell.tma.tensor_descriptor

```python
tensor_descriptor(handle, shape: 'List[ttgl.tensor]', strides: 'List[ttgl.tensor]', block_type: 'ttgl.block_type', layout: 'NVMMASharedLayout')
```

**`tensor_descriptor(handle, shape, strides, block_type, layout)`**

   Represents a tiled tensor descriptor for Tensor Memory Accelerator (TMA) operations.

   This class encapsulates the global memory address, dimensions, strides, and shared memory layout required for hardware-accelerated async copies on NVIDIA Hopper and Blackwell GPUs. Instances are typically created via `make_tensor_descriptor()` rather than direct instantiation.

   Parameters
   ----------
   handle : ir.value
       Internal IR handle representing the descriptor base pointer.
   shape : List[ttgl.tensor]
       List of tensors defining the global dimensions of the tensor.
   strides : List[ttgl.tensor]
       List of tensors defining the strides (in elements) for each dimension.
   block_type : ttgl.block_type
       Specifies the element dtype and the static block shape to be transferred.
   layout : NVMMASharedLayout
       The shared memory layout configuration (e.g., swizzle, CGA) for the destination block.

   Notes
   -----
   This descriptor is used with TMA async copy operations such as `async_copy_global_to_shared()`.
   Direct instantiation is reserved for low-level IR manipulation; users should prefer `make_tensor_descriptor()`.

   Examples
   --------
   Typically created via the helper function:

```python
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl
    from triton.experimental.gluon.language.nvidia.blackwell import tma

    @gluon.jit
    def kernel(...):
        # Create a tensor descriptor for a 2D tensor
        desc = tma.make_tensor_descriptor(
            base=global_ptr,
            shape=[height, width],
            strides=[stride_h, stride_w],
            block_shape=[block_h, block_w],
            layout=tma.NVMMASharedLayout(...)
        )
        # Use in async copy
        tma.async_copy_global_to_shared(desc, coord, barrier, shared_ptr)
```

---

### triton.experimental.gluon.language.nvidia.blackwell.tma.tensor_descriptor_type

```python
tensor_descriptor_type(block_type: 'ttgl.block_type', shape_type: 'ttgl.tuple_type', strides_type: 'ttgl.tuple_type', layout: 'NVMMASharedLayout', _type_name: 'str' = 'tensor_descriptor', _mangle_prefix: 'str' = 'TD') -> None
```

## class triton.experimental.gluon.language.nvidia.blackwell.tma.tensor_descriptor_type

Type for tiled tensor descriptors.

Represents a typed descriptor for Tensor Memory Accelerator (TMA) operations on NVIDIA Blackwell/Hopper GPUs. This type encapsulates the block shape, tensor shape, strides, and shared memory layout required for efficient tiled global-to-shared memory transfers.

### Parameters
block_type : ttgl.block_type
    The block type defining the element type and block shape for tiled access.
shape_type : ttgl.tuple_type
    Tuple type representing the runtime tensor shape dimensions.
strides_type : ttgl.tuple_type
    Tuple type representing the runtime tensor stride dimensions.
layout : NVMMASharedLayout
    Shared memory layout specifying the arrangement of data in shared memory (e.g., swizzling, CGA partitioning).
_type_name : str, optional
    Internal type name identifier. Default is `"tensor_descriptor"`.
_mangle_prefix : str, optional
    Internal mangle prefix for type naming. Default is `"TD"`.

### Returns
tensor_descriptor_type
    A typed descriptor for TMA tiled tensor operations.

### Notes
This type is typically constructed via `triton.experimental.gluon.language.nvidia.blackwell.tma.make_tensor_descriptor()` rather than instantiated directly. The descriptor enables efficient asynchronous memory transfers using TMA hardware units on NVIDIA Blackwell and Hopper architectures.

The `block_type` must satisfy alignment requirements: the last dimension must contain at least 16 bytes of data. For FP4 layouts with padding, 64-byte alignment is required.

### See Also
make_tensor_descriptor : Construct a tensor descriptor from base pointer, shape, and strides.
tensor_descriptor_im2col_type : Type for im2col tensor descriptors (convolution-friendly patterns).
async_copy_global_to_shared : Perform async TMA copy from global to shared memory.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language._layouts import NVMMASharedLayout

 @gluon.jit
 def kernel(...):
     # Define block shape (static)
     block_shape = [128, 64]
     
     # Define layout (e.g., swizzled shared memory layout)
     layout = NVMMASharedLayout(...)
     
     # Create tensor descriptor type
     block_type = ttgl.block_type(ttgl.float16, block_shape)
     shape_type = ttgl.tuple_type([ttgl.int32, ttgl.int32])
     strides_type = ttgl.tuple_type([ttgl.int64, ttgl.int64])
     
     desc_type = ttgl.nvidia.blackwell.tma.tensor_descriptor_type(
         block_type=block_type,
         shape_type=shape_type,
         strides_type=strides_type,
         layout=layout
     )
     
     # Use with make_tensor_descriptor to create actual descriptor
     desc = ttgl.nvidia.blackwell.tma.make_tensor_descriptor(
         base=global_ptr,
         shape=[h, w],
         strides=[stride_h, stride_w],
         block_shape=block_shape,
         layout=layout
     )
```

---


## triton.experimental.gluon.language.nvidia.hopper

### triton.experimental.gluon.language.nvidia.hopper.fence_async_shared

```python
fence_async_shared(cluster=False, _semantic=None)
```

## fence_async_shared


.. autofunction:: fence_async_shared

.. rubric:: Description

Issue a fence to complete asynchronous shared memory operations on NVIDIA Hopper GPUs.

### Parameters
cluster : bool, optional
    Whether to fence across the CTA cluster. Defaults to `False`, which fences
    within the current CTA only. Set to `True` to synchronize across all CTAs
    in the cluster.

### Returns
None

### Notes
This function is specific to NVIDIA Hopper architecture (compute capability 90+).
It ensures that all prior asynchronous shared memory operations (such as TMA
loads/stores) are completed before subsequent memory operations proceed.

When `cluster=True`, the fence synchronizes across all CTAs in the cluster,
which is required for correct coordination of cluster-wide shared memory
accesses. This incurs additional overhead compared to CTA-local fencing.

This builtin requires the `_semantic` argument to be provided by the Gluon
JIT compiler and should not be called directly outside of a `@gluon.jit`
decorated kernel.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl

 @gluon.jit
 def kernel_async_fence(...):
     # Issue async shared memory operations (e.g., TMA load)
     # ...

     # Fence to ensure async operations complete
     ttgl.nvidia.hopper.fence_async_shared()

     # Safe to access shared memory now
     # ...

 @gluon.jit
 def kernel_cluster_fence(...):
     # Cluster-wide async operations
     # ...

     # Fence across entire cluster
     ttgl.nvidia.hopper.fence_async_shared(cluster=True)

     # All CTAs in cluster see consistent shared memory state
     # ...
```

---

### triton.experimental.gluon.language.nvidia.hopper.mma_v2

```python
mma_v2(a, b, acc, input_precision=None, _semantic=None)
```

## mma_v2


**`mma_v2(a, b, acc, input_precision=None, _semantic=None)`**

    Perform a matrix multiply-accumulate operation using NVIDIA Hopper MMA version 2.0.

    This function executes a tensor core MMA instruction with explicit layout control.
    The accumulator tensor determines the MMA layout version and distribution, while
    input tensors must have compatible dot operand layouts.

    Parameters
    ----------
    a : tensor
        First input tensor with DotOperandLayout (operand_index=0). The parent layout
        must match the accumulator's NVMMADistributedLayout.
    b : tensor
        Second input tensor with DotOperandLayout (operand_index=1). The parent layout
        must match the accumulator's NVMMADistributedLayout.
    acc : tensor
        Accumulator tensor with NVMMADistributedLayout version 2.0. Defines the output
        layout and dtype.
    input_precision : dtype, optional
        Precision for intermediate computation (e.g., float32, tf32). Defaults to None,
        using the accumulator's dtype.
    _semantic : GluonSemantic, optional
        Internal parameter for semantic context. Do not set manually.

    Returns
    -------
    tensor
        Result tensor with the same type and layout as the accumulator.

    Notes
    -----
    This is a low-level primitive for explicit GPU programming. All tensors must have
    compatible layouts as verified by runtime assertions:

    - `acc` must have `NVMMADistributedLayout` with version `[2, 0]`
    - `a` must have `DotOperandLayout` with `operand_index=0`
    - `b` must have `DotOperandLayout` with `operand_index=1`
    - Both operand layouts must share the same parent as `acc`

    The function requires execution within a `@gluon.jit` decorated kernel.

    Examples
    --------
    >>> import triton.experimental.gluon as gluon
    >>> import triton.experimental.gluon.language as ttgl
    >>> from triton.experimental.gluon.language.nvidia import hopper
    >>>
    >>> @gluon.jit
    >>> def kernel(...):
    ...     # Create MMA layout version 2.0
    ...     mma_layout = hopper.NVMMADistributedLayout(version=[2, 0], ...)
    ...
    ...     # Create dot operand layouts
    ...     a_layout = ttgl.DotOperandLayout(parent=mma_layout, operand_index=0)
    ...     b_layout = ttgl.DotOperandLayout(parent=mma_layout, operand_index=1)
    ...
    ...     # Load or create tensors with appropriate layouts
    ...     a = ...  # tensor with a_layout
    ...     b = ...  # tensor with b_layout
    ...     acc = ttgl.full(shape, 0.0, dtype=ttl.float16, layout=mma_layout)
    ...
    ...     # Perform MMA v2 operation
    ...     result = hopper.mma_v2(a, b, acc)

---

### triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma

```python
warpgroup_mma(a, b, acc, *, use_acc=True, precision=None, max_num_imprecise_acc=None, is_async=False, _semantic=None)
```

## triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma


**`warpgroup_mma(a, b, acc, *, use_acc=True, precision=None, max_num_imprecise_acc=None, is_async=False)`**

    Perform warpgroup MMA (Tensor Core) operations on NVIDIA Hopper GPUs.

    Computes `acc = a * b + (acc if use_acc else 0)` using Tensor Core matrix
    multiply-accumulate instructions. This operation is optimized for Hopper
    architecture and supports both synchronous and asynchronous execution modes.

    Parameters
    ----------
    a : tensor or shared_memory_descriptor
        Left hand side operand of the matrix multiplication.
    b : shared_memory_descriptor
        Right hand side operand of the matrix multiplication. Must be in shared
        memory.
    acc : tensor
        Accumulator tensor that holds the result.
    use_acc : bool, optional
        Whether to use the initial value of the accumulator. Defaults to `True`.
        If `False`, the accumulator is initialized to zero.
    precision : str, optional
        Dot input precision (e.g., `"tf32"`, `"fp16"`, `"fp32"`). Defaults to
        the builder's default precision setting.
    max_num_imprecise_acc : int, optional
        Maximum number of imprecise accumulations. Used for fp8 -> fp32 dot
        operations to control how many accumulations are done in limited precision
        before upcasting. Defaults to `None`, which means no upcasting is done
        for non-fp8 types, and uses the builder default for fp8.
    is_async : bool, optional
        Whether the operation executes asynchronously. Defaults to `False`.

    Returns
    -------
    tensor or warpgroup_mma_accumulator
        If `is_async=False`, returns a tensor containing the result. If
        `is_async=True`, returns a `warpgroup_mma_accumulator` token
        that can be used to load the value once computation completes.

    Notes
    -----
    This function is specific to NVIDIA Hopper architecture and requires the
    Gluon JIT decorator (`@gluon.jit`). The operands must have compatible
    layouts for Tensor Core operations.

    For fp8 inputs, `max_num_imprecise_acc` controls numerical accuracy by
    limiting the number of accumulations performed in lower precision before
    upcasting to fp32. The value must not exceed the K dimension of the matrix
    multiplication.

    Asynchronous mode allows overlapping computation with other operations but
    requires explicit synchronization to retrieve results.

    Examples
    --------
    Synchronous warpgroup MMA operation:

```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(...):
         # Load operands from shared memory
         a = ttgl.load(...)
         b = ttgl.load(...)
         acc = ttgl.full(shape, 0.0, dtype=ttgl.float32)

         # Perform synchronous MMA
         result = ttgl.nvidia.hopper.warpgroup_mma(a, b, acc)

 Asynchronous warpgroup MMA operation:

 .. code-block:: python

     @gluon.jit
     def kernel(...):
         a = ttgl.load(...)
         b = ttgl.load(...)
         acc = ttgl.full(shape, 0.0, dtype=ttgl.float32)

         # Perform asynchronous MMA
         mma_token = ttgl.nvidia.hopper.warpgroup_mma(a, b, acc, is_async=True)

         # Wait for completion and load result
         result = mma_token.wait()

 fp8 dot with imprecise accumulation control:

 .. code-block:: python

     @gluon.jit
     def kernel(...):
         a = ttgl.load(..., dtype=ttgl.float8e4nv)
         b = ttgl.load(..., dtype=ttgl.float8e4nv)
         acc = ttgl.full(shape, 0.0, dtype=ttgl.float32)

         # Limit imprecise accumulations to 64 before upcasting
         result = ttgl.nvidia.hopper.warpgroup_mma(
             a, b, acc,
             max_num_imprecise_acc=64
         )
```

---

### triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma_wait

```python
warpgroup_mma_wait(num_outstanding=0, deps=None, _semantic=None)
```

## warpgroup_mma_wait


**`warpgroup_mma_wait(num_outstanding=0, deps=None)`**

   Wait until the specified number of warpgroup MMA operations are in-flight.

   Blocks execution until `num_outstanding` or fewer asynchronous warpgroup
   matrix multiply-accumulate operations remain pending. Use this to control
   MMA operation scheduling and ensure dependencies are preserved during
   asynchronous execution.

   Parameters
   ----------
   num_outstanding : int, optional
       Maximum number of outstanding warpgroup MMA operations to allow.
       Defaults to 0, which waits until all pending MMAs complete.
   deps : Sequence[tensor], optional
       List of tensor dependencies that must be kept alive while the MMA
       operations are unfinished. This parameter is required at runtime
       despite the default value in the signature.

   Returns
   -------
   tensor or Tuple[tensor, ...]
       The dependency tensors returned to maintain liveness. If `deps`
       contains a single tensor, returns that tensor directly. Otherwise
       returns a tuple of all dependency tensors.

   Notes
   -----
   This function is specific to NVIDIA Hopper architecture and controls
   warpgroup-level asynchronous MMA operations. The `deps` parameter
       ensures that input tensors remain valid until MMA completion.

   Passing `deps=None` will raise a :exc:`ValueError` at runtime.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma_wait

   @gluon.jit
   def kernel(...):
       # Issue asynchronous warpgroup MMA operations
       acc = ttgl.dot(a, b, acc)

       # Wait until 2 or fewer MMA operations are in-flight
       # Keep acc alive during the wait
       warpgroup_mma_wait(num_outstanding=2, deps=[acc])

       # Continue with dependent operations
       ...
```

---


## triton.experimental.gluon.language.nvidia.hopper.cluster

### triton.experimental.gluon.language.nvidia.hopper.cluster.arrive

```python
arrive(relaxed: 'bool' = False, _semantic=None)
```

**`triton.experimental.gluon.language.nvidia.hopper.cluster.arrive(relaxed=False)`**

    Signal arrival at a cluster synchronization barrier.

    Parameters
    ----------
    relaxed : bool, optional
        If True, use relaxed consistency semantics. Defaults to False.

    Notes
    -----
    Specific to NVIDIA Hopper architecture. Signals that the CTA has reached
    a synchronization point within a cluster. Must be paired with
    `wait()` or `expect()` for correct cluster-wide synchronization.

    Requires the kernel to be launched with a cluster configuration.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(...):
         ttgl.nvidia.hopper.cluster.arrive()
         ttgl.nvidia.hopper.cluster.wait()
```

---

### triton.experimental.gluon.language.nvidia.hopper.cluster.barrier

```python
barrier(relaxed: 'bool' = False, _semantic=None)
```

## barrier


**`barrier(relaxed=False, _semantic=None)`**

   Insert a barrier to synchronize all CTAs within a CTA cluster on NVIDIA Hopper GPUs.

   Parameters
   ----------
   relaxed : bool, optional
       Whether to use relaxed arrival semantics. When True, CTAs may arrive
       at the barrier without strict ordering guarantees. Defaults to False.
   _semantic : GluonSemantic, optional
       Internal semantic context. Do not set manually.

   Returns
   -------
   None

   Notes
   -----
   This barrier synchronizes across all CTAs in the cluster, enabling
   coordination for multi-CTA kernels. All CTAs in the cluster must reach
   this barrier before any can proceed.

   Relaxed semantics may improve performance but provide weaker memory
   ordering guarantees. Use default (False) for strong consistency.

   Requires multi-CTA kernel configuration. Has no effect when num_ctas == 1.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def multi_cta_kernel(...):
       # Perform work
       ...
       
       # Synchronize all CTAs in the cluster
       ttgl.nvidia.hopper.cluster.barrier()
       
       # Continue with coordinated work
       ...
```

---

### triton.experimental.gluon.language.nvidia.hopper.cluster.wait

```python
wait(_semantic=None)
```

## gluon.language.nvidia.hopper.cluster.wait


**`wait(_semantic=None)`**

   Wait for all CTAs in the cluster to arrive at the cluster barrier.

   Synchronizes all cooperative thread arrays (CTAs) within a cluster on
   NVIDIA Hopper GPUs. All CTAs must call this function to ensure
   cluster-wide synchronization before proceeding.

   Parameters
   ----------
   _semantic : GluonSemantic, optional
       Internal semantic argument injected by the Gluon JIT compiler.
       Do not pass this argument when calling from user code.

   Returns
   -------
   None

   Notes
   -----
   This function is only available on NVIDIA Hopper architecture (H100, H200,
   etc.) and requires the kernel to be launched with multiple CTAs in a
   cluster. All CTAs in the cluster must reach this barrier before any can
   proceed.

   This is a hardware-level cluster synchronization primitive. For CTA-local
   synchronization, use `gluon.language.barrier()` with `cluster=False`.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.hopper import cluster

   @gluon.jit
   def kernel(...):
       # Perform work before cluster synchronization
       ...

       # Wait for all CTAs in the cluster to arrive
       cluster.wait()

       # All CTAs continue together after barrier
       ...
```

---


## triton.experimental.gluon.language.nvidia.hopper.mbarrier

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.MBarrierLayout

```python
MBarrierLayout(cga_layout=None)
```

**`MBarrierLayout(cga_layout=None)`**

    Layout descriptor for hardware mbarrier synchronization on NVIDIA Ampere 
    and later architectures.

    Parameters
    ----------
    cga_layout : list of list of int, optional
        CGA layout bases. Defaults to `[]`.

    .. staticmethod:: multicta(num_ctas, two_cta=False)

        Create a multi-CTA mbarrier layout.

        Parameters
        ----------
        num_ctas : int
            Number of CTAs. Must be a power of two.
        two_cta : bool, optional
            Whether the barrier should synchronize every other CTA. 
            Defaults to `False`.

        Returns
        -------
        MBarrierLayout
            The configured layout instance.

        Notes
        -----
        If `two_cta` is `True`, `num_ctas` must be even.

    Notes
    -----
    Inherits from `SwizzledSharedLayout`. This layout is typically 
    used with `triton.experimental.gluon.language.allocate_shared_memory()` 
    to create mbarrier objects for cross-CTA synchronization.

    Examples
    --------
```python
     import triton.experimental.gluon.language as ttgl
     from triton.experimental.gluon.language.nvidia.hopper.mbarrier import MBarrierLayout

     # Basic instantiation
     layout = MBarrierLayout()

     # Using the multicta helper for multi-CTA synchronization
     layout = MBarrierLayout.multicta(num_ctas=4)
```

---

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.allocate_mbarrier

```python
allocate_mbarrier(batch: triton.language.core.constexpr = None, two_ctas: triton.language.core.constexpr = False)
```

**`allocate_mbarrier(batch: triton.language.core.constexpr = None, two_ctas: triton.language.core.constexpr = False)`**

   Allocate shared memory for a multi-CTA memory barrier (mbarrier).

   Parameters
   ----------
   batch : constexpr, optional
       Number of barrier instances to allocate in a batch. If `None`, allocates
       a 1D array sized by the CTA count.
   two_ctas : constexpr, bool
       If `True`, configures the barrier to synchronize every other CTA. This
       halves the number of participating CTAs per barrier instance.

   Returns
   -------
   shared_memory_descriptor
       The allocated mbarrier object ready for initialization.

   Notes
   -----
   The underlying shared memory layout is configured using `MBarrierLayout`.
   Requires the grid dimension to be a power of two. If `two_ctas` is `True`,
   the total number of CTAs must be even. Compatible with NVIDIA Ampere, Hopper,
   and Blackwell architectures.

   Examples
   --------
```python
    import triton.experimental.gluon as gluon
    import triton.experimental.gluon.language as ttgl

    @gluon.jit
    def kernel(...):
        # Allocate a barrier for all CTAs
        bar = ttgl.nvidia.hopper.mbarrier.allocate_mbarrier()
        ttgl.nvidia.hopper.mbarrier.init(bar, count=1024)

        # Allocate a batch of 4 barriers
        bar_batch = ttgl.nvidia.hopper.mbarrier.allocate_mbarrier(batch=4)

        # Allocate barrier for every other CTA
        bar_pair = ttgl.nvidia.hopper.mbarrier.allocate_mbarrier(two_ctas=True)
```

---

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.arrive

```python
arrive(mbarrier, *, count=1, pred=True, _semantic=None)
```

**`arrive(mbarrier, *, count=1, pred=True)`**

   Arrive at an mbarrier with a specified count.

   Parameters
   ----------
   mbarrier : shared_memory_descriptor
       Barrier to be signalled.
   count : int, optional
       Count to arrive with. Defaults to 1.
   pred : bool, optional
       Scalar predicate. Operation is skipped if predicate is False. Defaults to True.

   Returns
   -------
   None

   Notes
   -----
   This operation is specific to NVIDIA Hopper architectures. It signals arrival at a
   memory barrier located in shared memory, optionally incrementing the transaction count.
   The operation is predicated; if `pred` is False, no state change occurs.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(mb_ptr):
       # Assume mb_ptr is a shared_memory_descriptor representing an mbarrier
       ttgl.nvidia.hopper.mbarrier.arrive(mb_ptr, count=1)
```

---

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.expect

```python
expect(mbarrier, bytes_per_cta=None, pred=True, _semantic=None)
```

Expect a specific number of bytes to be copied before signaling the mbarrier.

When the expected byte count is reached, the barrier is automatically signaled,
allowing synchronization between asynchronous copy operations and compute.

### Parameters
mbarrier : shared_memory_descriptor
    Barrier that will be signaled when the expected byte count is reached.
    Must be a shared memory descriptor allocated for mbarrier use.
bytes_per_cta : int, optional
    Expected byte count per CTA (Cooperative Thread Array). If None,
    uses a default value determined by the backend.
pred : bool, optional
    Scalar predicate controlling execution. Operation is skipped if
    predicate is False. Defaults to True.

### Returns
None
    This function operates via side effects on the mbarrier state.

### Notes
This function is specific to NVIDIA Hopper architecture and requires Gluon JIT
context. The `_semantic` parameter is automatically provided by the Gluon JIT
infrastructure and should not be passed by users.

The mbarrier must be properly allocated before calling this function. The
byte count represents the total expected transfers across all participating
warps in the CTA.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.nvidia.hopper import mbarrier

 @gluon.jit
 def kernel(...):
     # Allocate shared memory for mbarrier
     mb = mbarrier.alloc(shape=[1], dtype=ttgl.uint64)
     
     # Expect 256 bytes to be copied via TMA
     mbarrier.expect(mb, bytes_per_cta=256)
     
     # ... initiate async copy operations ...
     
     # Wait until the expected bytes are copied
     mbarrier.wait(mb, 0)
```

---

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.fence_init_release_cluster

```python
fence_init_release_cluster(_semantic=None)
```

## fence_init_release_cluster


.. autofunction:: fence_init_release_cluster

Insert a memory fence that makes prior mbarrier initialization visible across the CTA cluster.

This fence ensures that mbarrier initialization operations performed before the fence are visible to all CTAs in the cluster. It must be used in conjunction with `cluster.barrier()` with `relaxed=True` to establish proper synchronization ordering.

### Parameters
_semantic : GluonSemantic, optional
    Internal semantic builder. Do not pass this argument directly; it is automatically provided by the `@gluon.jit` decorator.

### Returns
None

### Notes
This function is specific to NVIDIA Hopper architecture and requires multi-CTA cluster configuration.

The fence must be paired with `triton.experimental.gluon.language.barrier()` called with `cluster=True` and `relaxed=True` to ensure correct synchronization semantics across the CTA cluster.

Typical usage pattern::

    # Initialize mbarrier
    mbarrier.init(...)
    # Make initialization visible across cluster
    mbarrier.fence_init_release_cluster()
    # Synchronize across cluster
    gluon.barrier(cluster=True)

This ordering ensures all CTAs in the cluster observe the mbarrier initialization before proceeding.

### Examples

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.nvidia.hopper import mbarrier

 @gluon.jit
 def kernel_fn(...):
     # Initialize mbarrier in shared memory
     mbarrier.init(mbarrier_desc, transaction_bytes=...)
     
     # Ensure initialization is visible across CTA cluster
     mbarrier.fence_init_release_cluster()
     
     # Synchronize all CTAs in the cluster
     gluon.barrier(cluster=True)
     
     # Now safe to use mbarrier across cluster
     ...
```

---

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.init

```python
init(mbarrier, count, _semantic=None)
```

**`init(mbarrier, count)`**

   Initialize a shared memory barrier with a specified transaction count.

   Parameters
   ----------
   mbarrier : shared_memory_descriptor
       The shared memory barrier object to initialize.
   count : int
       The initial transaction count for the barrier.

   Returns
   -------
   None
       This function operates in-place and does not return a value.

   Notes
   -----
   This API is specific to NVIDIA Hopper architecture. It configures the mbarrier for tracking asynchronous memory transactions, such as those performed by the Tensor Memory Accelerator (TMA). The `count` typically represents the number of bytes or cache lines expected in the transaction.

   Examples
   --------
   Initialize an mbarrier for a 128-byte transaction within a Gluon kernel:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(...):
       mbarrier = ttgl.nvidia.hopper.mbarrier.alloc(...)
       ttgl.nvidia.hopper.mbarrier.init(mbarrier, count=128)
       # Proceed with TMA operations using the initialized mbarrier
```

---

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.invalidate

```python
invalidate(mbarrier, _semantic=None)
```

Invalidate an mbarrier, resetting its state.

### Parameters
mbarrier : `shared_memory_descriptor`
    The barrier object to invalidate.

### Returns
None

### Notes
This operation is specific to NVIDIA Hopper architecture. Invalidating a
memory barrier resets its transaction count and internal state, allowing it
to be re-armed for subsequent synchronization operations. This is typically
used when reusing barrier descriptors in loops or multiple
production/consumption phases.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from ttgl.nvidia.hopper import mbarrier

 @gluon.jit
 def kernel(...):
     # Allocate a shared memory barrier
     mb = mbarrier.alloc(...)

     # ... synchronize and wait ...

     # Reset the barrier state for reuse
     mbarrier.invalidate(mb)
```

---

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.sync_cluster_init

```python
sync_cluster_init()
```

**`sync_cluster_init()`**

   Ensure mbarrier initialization is visible across the CTA cluster.

   Notes
   -----
   This function emits a `fence_init_release_cluster` followed by a
   `cluster.barrier(relaxed=True)`. It is required when initializing
   mbarriers that are accessed by multiple CTAs within a cluster on
   NVIDIA Hopper architectures.

   Without this synchronization, mbarrier initialization writes performed
   by one CTA may not be visible to other CTAs in the cluster, leading to
   undefined behavior during `wait` or `expect` operations.

   Examples
   --------
   Initialize an mbarrier and synchronize across the cluster:

```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def kernel(...):
       # Allocate and initialize mbarrier in shared memory
       mbarrier = ttgl.nvidia.hopper.mbarrier.allocate_mbarrier(...)
       ttgl.nvidia.hopper.mbarrier.init(mbarrier, ...)

       # Ensure initialization is visible to other CTAs in the cluster
       ttgl.nvidia.hopper.mbarrier.sync_cluster_init()

       # Proceed with coordinated access
       ...
```

---

### triton.experimental.gluon.language.nvidia.hopper.mbarrier.wait

```python
wait(mbarrier, phase, pred=True, deps=(), _semantic=None)
```

## mbarrier.wait

Wait until the mbarrier object completes its current phase.

### Parameters
mbarrier : shared_memory_descriptor
    The barrier object to wait on.
phase : int
    The phase index to wait for.
pred : bool, optional
    Predicate. Operation is skipped if predicate is False. Defaults to True.
deps : Sequence[shared_memory_descriptor], optional
    Dependent allocations barrier is waiting on. Used to track liveness of
    dependent allocations. Defaults to ().

### Returns
None

### Notes
This function is specific to NVIDIA Hopper architecture and requires the
mbarrier to be properly initialized before calling wait. The phase parameter
specifies which phase of the barrier to wait for completion.

The predicate parameter allows conditional execution - if pred is False, the
wait operation is skipped entirely.

The deps parameter tracks dependent shared memory allocations to ensure
proper liveness tracking during barrier synchronization.

### Examples
```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.nvidia.hopper import mbarrier

 @gluon.jit
 def kernel():
     # Create a mbarrier in shared memory
     mb = mbarrier.alloc(shape=[128], layout=ttgl.SharedLayout())
     
     # Initialize the barrier
     mbarrier.init(mb, count=1024)
     
     # Wait for phase 0 to complete
     mbarrier.wait(mb, phase=0)
     
     # Wait for phase 1 with predicate
     mbarrier.wait(mb, phase=1, pred=True)
```

---


## triton.experimental.gluon.language.nvidia.hopper.tma

### triton.experimental.gluon.language.nvidia.hopper.tma.async_copy_global_to_shared

```python
async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, multicast=False, _semantic=None)
```

## async_copy_global_to_shared


**`async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, multicast=False)`**

   Copy data from global memory to shared memory using Tensor Memory Accelerator (TMA).

   This function initiates an asynchronous copy operation from global memory to shared
   memory on NVIDIA Hopper GPUs. The operation uses TMA hardware features for efficient
   memory transfers and requires proper synchronization via barriers.

   Parameters
   ----------
   tensor_desc : TensorDescriptor
       Tiled tensor descriptor describing the source data in global memory.
   coord : tensor or sequence of int
       Coordinates specifying the tile location in the source tensor.
   barrier : Barrier
       Synchronization barrier for coordinating async copy completion.
   result : shared_memory_descriptor
       Destination shared memory descriptor where data will be copied.
   pred : bool or tensor, optional
       Predicate for conditional execution. If False, the copy is skipped.
       Default is True.
   multicast : bool, optional
       Enable multicast mode for copying to multiple destinations.
       Default is False.

   Notes
   -----
   This is a low-level TMA primitive for NVIDIA Hopper architecture. The copy is
   asynchronous and non-blocking - use the provided barrier to synchronize before
   accessing the copied data in shared memory.

   The tensor descriptor must be properly aligned according to TMA requirements.
   When `enable_iisan` is enabled, alignment checks are performed automatically.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl
   from triton.experimental.gluon.language.nvidia.hopper import tma

   @gluon.jit
   def kernel(...):
       # Create shared memory descriptor
       smem = ttgl.allocate_shared_memory(
           element_ty=ttgl.float16,
           shape=[128, 128],
           layout=ttgl.SharedLayout(...)
       )

       # Create global tensor descriptor
       desc = tma.make_tensor_descriptor(...)

       # Create barrier for synchronization
       barrier = tma.barrier_alloc(...)

       # Async copy from global to shared memory
       coord = ttgl.arange(0, 128, layout=...)
       tma.async_copy_global_to_shared(
           tensor_desc=desc,
           coord=coord,
           barrier=barrier,
           result=smem
       )

       # Wait for copy to complete before using smem
       tma.barrier_wait(barrier)
```

---

### triton.experimental.gluon.language.nvidia.hopper.tma.async_copy_global_to_shared_im2col

```python
async_copy_global_to_shared_im2col(tensor_desc, coord, offsets, barrier, result, pred=True, multicast=False, _semantic=None)
```

## async_copy_global_to_shared_im2col

**`triton.experimental.gluon.language.nvidia.hopper.tma.async_copy_global_to_shared_im2col(tensor_desc, coord, offsets, barrier, result, pred=True, multicast=False)`**

   Asynchronously copy data from global memory to shared memory using TMA in im2col mode.

   Parameters
   ----------
   tensor_desc : TensorDescriptor
       Tensor descriptor for the source global memory tensor configured for im2col access pattern.
   coord : tensor
       Coordinates in the source tensor specifying the copy region. Must be convertible to IR values.
   offsets : tensor or sequence of int
       Im2col offset values (must be i16). Number of offsets depends on source tensor rank:
       
       - 3D tensors: 1 offset
       - 4D tensors: 2 offsets
       - 5D tensors: 3 offsets
   barrier : Barrier
       Barrier handle for synchronization. Signals completion when the async copy finishes.
   result : shared_memory_descriptor
       Destination shared memory descriptor where data will be copied.
   pred : tensor or bool, optional
       Predicate for conditional execution. If False, the copy is skipped (default: True).
   multicast : bool, optional
       Enable multicast for cluster-wide copy operation (default: False).

   Returns
   -------
   None

   Notes
   -----
   This function is NVIDIA Hopper-specific and requires TMA (Tensor Memory Accelerator) hardware support. The operation is asynchronous and returns immediately without waiting for the copy to complete. Use the provided barrier to synchronize completion before accessing the shared memory.

   When `multicast=True`, the copy is broadcast to all CTAs in the cluster. All participating CTAs must call this function with matching parameters.

   Examples
   --------
```python
   import triton.experimental.gluon as gluon
   import triton.experimental.gluon.language as ttgl

   @gluon.jit
   def im2col_kernel(...):
       # Setup tensor descriptor for global memory
       desc = ttgl.make_tensor_descriptor(...)
       
       # Allocate shared memory with appropriate layout
       smem = ttgl.allocate_shared_memory(...)
       
       # Create barrier for synchronization
       barrier = ttgl.barrier()
       
       # Compute coordinates and im2col offsets
       coord = ttgl.arange(0, block_size, layout=...)
       offsets = [h_offset, w_offset]  # For 4D tensor
       
       # Async copy with im2col offsets
       ttgl.nvidia.hopper.tma.async_copy_global_to_shared_im2col(
           desc,
           coord,
           offsets,
           barrier,
           smem
       )
       
       # Wait for copy completion before using smem
       ttgl.barrier_wait(barrier)
```

---

### triton.experimental.gluon.language.nvidia.hopper.tma.async_copy_shared_to_global

```python
async_copy_shared_to_global(tensor_desc, coord, src, _semantic=None)
```

**`async_copy_shared_to_global(tensor_desc, coord, src, _semantic=None)`**

    Asynchronously copy data from shared memory to global memory using Tensor Memory Accelerator (TMA).

    This operation initiates a non-blocking copy from a shared memory allocation to a global memory tensor described by a TMA descriptor. It is specific to NVIDIA Hopper architecture.

    Parameters
    ----------
    tensor_desc : tensor_descriptor
        The TMA tensor descriptor defining the global memory destination layout and address.
    coord : tensor or tuple of int
        Coordinates specifying the offset in the global tensor where the data will be written.
        Converted to IR values during compilation.
    src : shared_memory_descriptor
        The shared memory descriptor containing the source data to copy.

    Returns
    -------
    None
        This operation performs a side-effect and does not return a value.

    Notes
    -----
    This function is available only on NVIDIA Hopper GPUs (compute capability 90+).
    The copy is asynchronous; use barriers or wait operations to ensure completion before
    accessing the global memory data.
    Alignment checks are performed if the `enable_iisan` builder option is enabled.
    Ensure the shared memory layout and global tensor descriptor are compatible for TMA operations.

    Examples
    --------
```python
     import triton.experimental.gluon as gluon
     import triton.experimental.gluon.language as ttgl

     @gluon.jit
     def kernel(desc, smem, coord):
         # Assume desc is a TMA tensor_descriptor and smem is a shared_memory_descriptor
         ttgl.nvidia.hopper.tma.async_copy_shared_to_global(desc, coord, smem)
         # Insert barrier to wait for copy completion if needed
         ttgl.barrier()
```

---

### triton.experimental.gluon.language.nvidia.hopper.tma.store_wait

```python
store_wait(pendings, _semantic=None)
```

## store_wait

Wait for pending asynchronous TMA store operations to complete.

```python
 triton.experimental.gluon.language.nvidia.hopper.tma.store_wait(pendings, _semantic=None)

```
### Parameters

pendings : int
    The number of pending store operations to wait for. Typically the return value
    from `triton.experimental.gluon.language.nvidia.hopper.tma.store()` or
    `triton.experimental.gluon.language.nvidia.hopper.tma.async_store()`.

_semantic : GluonSemantic, optional
    Internal parameter used by the Gluon JIT compiler. Do not set manually.

### Returns

None

### Notes

This function inserts a `tma.store_wait` PTX instruction that stalls until the
specified number of pending TMA store transactions have completed. It is required
for correctness when reading data that was previously written via async TMA stores.

On NVIDIA Hopper (and later) architectures, TMA operations are asynchronous with
respect to the thread pipeline. Proper use of `store_wait()` ensures memory
consistency before subsequent loads or computes depend on the stored data.

The `pendings` argument should match the number of outstanding store operations
from prior TMA store calls in the same kernel region.

### Examples

```python
 import triton.experimental.gluon as gluon
 import triton.experimental.gluon.language as ttgl
 from triton.experimental.gluon.language.nvidia.hopper import tma

 @gluon.jit
 def kernel(
     input_ptr,
     output_ptr,
     BLOCK_SIZE: ttgl.constexpr,
 ):
     # Allocate shared memory for TMA transfer
     smem = ttgl.allocate_shared_memory(
         ttgl.float16,
         [BLOCK_SIZE],
         ttgl.swizzled_shared_layout(BLOCK_SIZE, 8),
     )

     # Async TMA store from global to shared memory
     pending = tma.store(input_ptr, smem)

     # Wait for store to complete before reading from shared memory
     tma.store_wait(pending)

     # Now safe to load from shared memory
     data = smem.load(ttgl.blocked_layout([BLOCK_SIZE]))

     # ... process data ...

     ttgl.store(output_ptr, data)
```

---
