# Checklist of items needed for Numba CUDA to support Awkward Arrays

## `@numba.extending.register_model`

**Requirement:** `@nb.extending.register_model` for StructModels that:

- [X] contain numeric types 
- [X] `nb.types.Array` - will probably work for having an array as a struct
  member.
- [X] one of which contains a `nb.types.pyobject`, but it's only used to track a
  reference, not use it in the Python API. *this should be OK as long as it's
  not accessed*.

**Description of support:** Models registered with `register_model` are already
supported by the CUDA target, and are used in its
[`models`](https://github.com/numba/numba/blob/master/numba/cuda/models.py)
module.


## Boxing and unboxing

**Requirement:** objects described by the registered models should be supported by:

- [ ] Boxing
- [ ] Unboxing
- [ ] `incref` / `decref` - presumably from `numba.core.pythonapi`

**Description of support:** Presently there is no support for boxing and
unboxing.  Further, the CUDA target argument preparation will only accept
objects which it knows about - see the
[`prepare_args()`](https://github.com/numba/numba/blob/6b82cd7b508b17d9eeb48e54f22dd18c67b711a2/numba/cuda/compiler.py#L744)
function. You can register an extension that maps anopther type to types that it
knows about.

**Question:** Can registering an extension that maps an Awkward Array to a type
that Numba knows about provide a workaround for the Boxing / Unboxing
requirement?


## Typing

**Requirement:** register typing for operators, functions, methods, and attributes:


- [X] `@nb.core.typing.templates.infer_global(operator.getitem)` using an
  `AbstractTemplate` and optionally all the other operators:
  - `abs`, `operator.inv`, `operator.invert`, `operator.neg`, `operator.not_`,
    `operator.pos`, `operator.truth`, `operator.add`, `operator.and_`,
    `operator.contains`, `operator.eq`, `operator.floordiv`, `operator.ge`, `operator.gt`,
    `operator.le`, `operator.lshift`, `operator.lt`, `operator.mod`, `operator.mul`,
    `operator.ne`, `operator.or_`, `operator.pow`, `operator.rshift`, `operator.sub`,
    `operator.truediv`, `operator.xor`, `operator.matmul`
- [X] `@nb.core.typing.templates.infer_global(len)`
- [X] `@nb.core.typing.templates.infer_getattr` for methods and properties

**Description of support:** typing is generally well-supported for the CUDA
target and there are no limitations on what can be typed (often the tricky
things happen in lowering instead!)


## Lowering

**Requirement:** A way to register lowering with:

- [X] `@nb.extending.lower_builtin` 
- [X] `@nb.extending.lower_builtin(operator.getitem, ...)`
- [X] `@nb.extending.lower_getattr_generic`
- [X] `@nb.core.imputils.lower_constant` for a `StructModel`

**Description of support:** The first three *probably* work already by importing
`numba.cuda.cudaimpl.registry as cuda_registry` and then using
`@cuda_registry.lower_builtin` - other extensions and the target use the `lower`
and `lower_attr` methods of this registry for lowering. Lowering a constant for
a StructModel should work.


## `@numba.extending.overload`

**Requirement:** Support for overloading functions

- [X] `@nb.extending.overload(operator.contains)`

**Description of support:** Overloading an operator should work - support for
overloads in the CUDA was added in May 2021 and is tested by the code in
[`numba.cuda.tests.cudapy.test_overload`](https://github.com/numba/numba/blob/master/numba/cuda/tests/cudapy/test_overload.py).


## Memory allocation

**Requirement:** Support for creating arrays:

- [ ] `@nb.extending.lower_builtin(np.asarray, ...)`
- [ ] `@nb.extending.overload(np.array)`
- [ ] `@nb.extending.type_callable(np.asarray)`

**Description of support:** It looks like `@overload(np.array)` and
`@lower_builtin(np.asarray)` refer to the construction of arrays - memory
allocation is not presently supported in CUDA, so this is a big hurdle.

**Question:** Is the above description correct? If so, is it necessary to have
these allocations in kernels?


## Typing context methods

**Requirement:** Unify a set of types using the method:

- [X] `unify_types`

**Description of support:** Already used in the CUDA target and by other
extensions.


## Target context methods

**Requirement:** 

- [X] `make_helper`,
- [X] `make_constant_array - if this doesn't work some alternative will be available.
- [ ] `if enable_nrt` - NRT will not be available
- [X] `get_constant`,
- [X] `get_value_type`,
- [X] `make_tuple`,
- [X] `compile_internal`,
- [X] `get_dummy_value`

**Description of support:** Generally used within the CUDA target or will be
available for use, apart from the exception noted above

**Not expected to work, but used for the CPU:**

- [ ] `add_dynamic_addr` - may work
- [ ] `call_conv.return_user_exc` - may work
- [ ] `get_python_api`
- [ ] `get_function_pointer_type`
- [ ] `call_function_pointer`

**Description of support:** There is some exception support on CUDA, and the
implementation of `add_dynamic_addr` looks like it might compile. The last three
functions pertain to unsupported functionality in CUDA.


## Builder functions:

**Requirement:** Using the llvmlite IR builder to construct IR for:

- [X] `sub`
- [X] `load`
- [X] `icmp_signed`
- [X] `if_then`
- [X] `store`
- [X] `mul`
- [X] `or_`
- [X] `not_`
- [X] `and_`
- [X] `inttoptr`
- [X] `bitcast`
- [X] `icmp_unsigned`
- [X] `zext`
- [X] `sitofp`
- [X] `uitofp`
- [X] `fpext`
- [X] `fptrunc`
- [X] `if_else`
- [X] `add`
- [X] `sdiv`
- [X] `srem`
- [X] `lshr`
- [X] `shl`
- [X] `sext`
- [X] `trunc]`

**Description of support:** These are all regularly used for building IR in the
CUDA target.


## `cgutils` module

**Requirement:** The following utility functions:

- [X] `cgutils.is_not_null` (for boxing),
- [X] `cgutils.increment_index`
- [X] `cgutils.pointer_add`
- [X] `cgutils.alloca_once_value`
- [X] `cgutils.as_bool_bit`
- [X] ` cgutils.for_range`
- [X] ` cgutils.false_bit`
- [X] ` cgutils.get_null_value`
- [X] `cgutils.true_bit`

**Description of support:** These are all convenience functions for building IR,
and many are used in the CUDA target.


## `imputils` module

**Requirement:**: The following utility function:

- [X] `imputils.impl_ret_new_ref`

**Description of support:**: This functions just returns its argument, so it
will work. However, the use of this does imply a general reliance on the Numba
Runtime (NRT) for reference counting, which is not supported on the CUDA target
(see [Memory allocation] above) so this may point to a general problem.


## To investigate

Items from the original list that need investigation:

- [ ] `SimpleIteratorType` (has an `EphemeralPointer(nb.intp)`) with `@nb.core.typing.templates.infer` for key `"getiter"` and `@nb.extending.lower_builtin("getiter", ...)`/`@nb.extending.lower_builtin("iternext", ...)`/`@nb.core.imputils.iternext_impl(RefType.BORROWED)`.


