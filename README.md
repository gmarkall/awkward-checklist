# Checklist of items needed for Numba CUDA to support Awkward Arrays

## `@numba.extending.register_model`

**Requirement:** `@nb.extending.register_model` for StructModels that:

- [X] contain numeric types 
- [ ] `nb.types.Array`,
- [ ] one of which contains a `nb.types.pyobject`, but it's only used to track a
  reference, not use it in the Python API.

**Description of support:** Models registered with `register_model` are already
supported by the CUDA target, and are used in its
[`models`](https://github.com/numba/numba/blob/master/numba/cuda/models.py)
module.


## Boxing and unboxing

**Requirement:** objects described by the registered models should be supported by:

- [ ] Boxing
- [ ] Unboxing

**Description of support:** Presently there is no support for boxing and
unboxing.  Further, the CUDA target argument preparation will only accept
objects which it knows about - see the
[`prepare_args()`](https://github.com/numba/numba/blob/6b82cd7b508b17d9eeb48e54f22dd18c67b711a2/numba/cuda/compiler.py#L744)
function. You can register an extension that maps anopther type to types that it
knows about.

**Question:** Can registering an extension that maps an Awkward Array to a type
that Numba knows about provide a workaround for the Boxing / Unboxing
requirement?


## `@numba.extending.lower_builtin`

**Requirement:** A way to register lowering with:

- [X] `@nb.extending.lower_builtin` 
- [X] `@nb.extending.lower_builtin(operator.getitem, ...)`

**Description of support:** This *probably* works already by importing
`numba.cuda.cudaimpl.registry as cuda_registry` and then using
`@cuda_registry.lower_builtin` - other extensions and the target use the `lower`
and `lower_attr` methods of this registry for lowering.


## `@numba.extending.overload`

**Requirement:** Support for overloading functions

- [X] `@nb.extending.overload(operator.contains)`

**Description of support:** Overloading an operator should work - support for
overloads in the CUDA was added in May 2021 and is tested by the code in
[`numba.cuda.tests.cudapy.test_overload](https://github.com/numba/numba/blob/master/numba/cuda/tests/cudapy/test_overload.py).


## Memory allocation

**Requirement:** Support for creating arrays:

- [ ] `@nb.extending.lower_builtin(np.asarray, ...)`
- [ ] `@nb.extending.overload(np.array)`

**Description of support:** It looks like `@overload(np.array)` and
`@lower_builtin(np.asarray)` refer to the construction of arrays - memory
allocation is not presently supported in CUDA, so this is a big hurdle.

**Question:** Is the above description correct? If so, is it necessary to have
these allocations in kernels?


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


## To investigate

Items from the original list that need investigation:

- [ ] `@nb.core.typing.templates.infer_global(operator.getitem)` and optionally all the other operators, `[abs, operator.inv, operator.invert, operator.neg, operator.not_, operator.pos, operator.truth, operator.add, operator.and_, operator.contains, operator.eq, operator.floordiv, operator.ge, operator.gt, operator.le, operator.lshift, operator.lt, operator.mod, operator.mul, operator.ne, operator.or_, operator.pow, operator.rshift, operator.sub, operator.truediv, operator.xor, operator.matmul]`, for `nb.core.typing.templates.AbstractTemplate`.
- [ ] Also, `@nb.core.typing.templates.infer_global(len)`
- [ ] `@nb.core.typing.templates.infer_getattr` for methods and properties
- [ ] `@nb.extending.lower_getattr_generic`
- [ ] `SimpleIteratorType` (has an `EphemeralPointer(nb.intp)`) with `@nb.core.typing.templates.infer` for key `"getiter"` and `@nb.extending.lower_builtin("getiter", ...)`/`@nb.extending.lower_builtin("iternext", ...)`/`@nb.core.imputils.iternext_impl(RefType.BORROWED)`.
- [ ] `@nb.extending.type_callable(np.asarray)`
- [ ] `@nb.core.imputils.lower_constant` for a `StructModel`
- [ ] Things in the context that I use: `[make_helper, make_constant_array, if enable_nrt, incref/decref (only in boxing), get_constant, get_value_type, make_tuple, compile_internal, get_dummy_value, unify_types]`.
- [ ] Things in the context that I use, but I don't expect it to work in CUDA: `[add_dynamic_addr, call_conv.return_user_exc, get_python_api, get_function_pointer_type, call_function_pointer]`.
- [ ] Things in nb.core that I use (other than typing): `[cgutils.is_not_null (boxing), imputils.impl_ret_new_ref, cgutils.increment_index, cgutils.pointer_add, alloca_once_value, cgutils.as_bool_bit, cgutils.for_range, cgutils.false_bit, cgutils.get_null_value, cgutils.true_bit]`.


