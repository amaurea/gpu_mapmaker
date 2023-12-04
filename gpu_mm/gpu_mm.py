# This source file will eventually become a proper python module that 'pip' can build.
#
# Currently, it contains some hackery (e.g. hardcoded pathname ../lib/libgpu_mm.so).
# To use it, you'll need to build the C++ library ('make -j' from the toplevel gpu_mm dir),
# and then do 'import gpu_mm' from the gpu_mm/scripts directory.

import ctypes, os
import numpy as np
import cupy as cp

_libgpu_mm = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),"libgpu_mm.so"))

# py_reference_map2tod(float *tod, const float *map, const float *xpointing, int ndet, int nt, int ndec, int nra);
_reference_map2tod = _libgpu_mm.py_reference_map2tod
_reference_tod2map = _libgpu_mm.py_reference_tod2map
_gpu_map2tod = _libgpu_mm.py_gpu_map2tod
_gpu_tod2map = _libgpu_mm.py_gpu_tod2map
_construct_cpu_plan1 = _libgpu_mm.py_construct_cpu_plan1
_construct_cpu_plan2 = _libgpu_mm.py_construct_cpu_plan2

# All function arguments are pointers, ints, or longs
_i = ctypes.c_int
_l = ctypes.c_long
_p = ctypes.c_void_p

# _reference_map2tod(tod, map, xpointing, ndet, nt, ndec, nra)
# _reference_tod2map(map, tod, xpointing, ndet, nt, ndec, nra)
# _gpu_map2tod(tod, map, xpointing, ndet, nt, ndec, nra)
# _gpu_tod2map(map, tod, xpointing, plan_cltod_list, plan_quadruples, plan_ncltod, plan_nquadruples, ndet, nt, ndec, nra)
# _construct_cpu_plan1(out, xpointing, ndet, nt, ndec, nra, verbose)
# _construct_cpu_plan2(plan_cltod_list, plan_quadruples, cookie, ncl_inflated, num_quadruples)

_reference_map2tod.argtypes = (_p, _p, _p, _i, _i, _i, _i)
_reference_tod2map.argtypes = (_p, _p, _p, _i, _i, _i, _i)
_gpu_map2tod.argtypes = (_p, _p, _p, _i, _i, _i, _i)
_gpu_tod2map.argtypes = (_p, _p, _p, _p, _p, _i, _i, _i, _i, _i, _i)
_construct_cpu_plan1.argtypes = (_p, _p, _i, _i, _i, _i, _i)
_construct_cpu_plan2.argtypes = (_p, _p, _l, _l, _l)


####################################################################################################


def check_array(func_name, arg_name, arr, ndim, dtype, gpu):
    if gpu and (not isinstance(arr, cp.ndarray)):
        raise RuntimeError(f'{func_name}: {arg_name}: expected cupy.ndarray, got {type(arr)}')
    if (not gpu) and (not isinstance(arr, np.ndarray)):
        raise RuntimeError(f'{func_name}: {arg_name}: expected numpy.ndarray, got {type(arr)}')
    if arr.dtype != dtype:
        raise RuntimeError(f'{func_name}: {arg_name}: expected {dtype=}, got {arr.dtype}')
    if arr.ndim != ndim:
        raise RuntimeError(f'{func_name}: {arg_name}: expected {ndim}-dimensional array, got dimension={arr.ndim}')
    if arr.size == 0:
        raise RuntimeError(f'{func_name}: {arg_name}: array has size=0 (shape={arr.shape})')
    if not arr.flags.c_contiguous:
        raise RuntimeError(f'{func_name}: {arg_name}: array is not C-contiguous')


def check_tmx_args(func_name, tod_arr, map_arr, xpointing_arr, gpu):
    """Error-checks (tod_arr, map_arr, xpointing_arr)."""
    
    dtype = cp.float32 if gpu else np.float32
    
    check_array(func_name, "'tod' argument", tod_arr, 2, dtype, gpu)
    check_array(func_name, "'map' argument", map_arr, 3, dtype, gpu)
    check_array(func_name, "'xpointing' argument", xpointing_arr, 3, dtype, gpu)

    if map_arr.shape[0] != 3:
        raise RuntimeError(f"{func_name}: {map_arr.shape=}, expected (3,ndec,nra)")
    if xpointing_arr.shape[0] != 3:
        raise RuntimeError(f"{func_name}: {xpointing_arr.shape=}, expected (3,ndet,nt)")
    if xpointing_arr.shape[1:] != tod_arr.shape:
        raise RuntimeError(f"{func_name}: {tod_arr.shape=} and {xpointing_arr.shape=} are inconsistent (should be (ndet,nt) and (3,ndet,nt))")

    ndet, nt = tod_arr.shape
    ndec, nra = map_arr.shape[1:]
    check_dims(func_name, ndet, nt, ndec, nra)


def check_dims(func_name, ndet, nt, ndec, nra):
    if (nt <=0) or (nt % 32):
        raise RuntimeError(f"{func_name}: got {nt=}, current inplementation assumes multiple of 32 (could be relaxed if needed)")
    if (ndec <= 0) or (ndec % 64):
        raise RuntimeError(f"{func_name}: got {ndec=}, current inplementation assumes multiple of 64 (could be relaxed if needed)")
    if (nra <= 0) or (nra % 64):
        raise RuntimeError(f"{func_name}: got {nra=}, current inplementation assumes multiple of 64 (could be relaxed if needed)")


####################################################################################################


def reference_map2tod(tod_out, map_in, xpointing):
    """
    Slow (single-threaded CPU) reference implementation of map2tod().

    tod_out: float32 numpy array of shape (ndetectors, ntimes).
        This array gets overwritten by reference_map2tod().
        Current implementation asserts (ntimes % 32) == 0.

    map_in: float32 numpy array of shape (3, ndec, nra).
        Current implementation asserts (ndec % 64) == (nra % 64) == 0.
        The length-3 axis is {I,Q,U}.

    xpointing: float32 numpy array of shape (3, ndetectors, ntimes).
   
        The length-3 axis is {px_dec, px_ra, alpha}, where:

            - px_dec is declination in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - px_ra is right ascension in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - alpha is detector angle in radians.

        ('xpointing' is short for "exploded pointing", i.e. per-detector pointing)
    """

    check_tmx_args('reference_map2tod()', tod_out, map_in, xpointing, gpu=False)
    
    ndet, nt = tod_out.shape
    ndec, nra = map_in.shape[1:]

    # _reference_map2tod(tod, map, xpointing, ndet, nt, ndec, nra)
    _reference_map2tod(tod_out.ctypes.data, map_in.ctypes.data, xpointing.ctypes.data, ndet, nt, ndec, nra)


def gpu_map2tod(tod_out, map_in, xpointing):
    """
    GPU implementation of map2tod().

    tod_out: float32 cupy array of shape (ndetectors, ntimes).
        This array gets overwritten by reference_map2tod().
        Current implementation asserts (ntimes % 32) == 0.

    map_in: float32 cupy array of shape (3, ndec, nra).
        Current implementation asserts (ndec % 64) == (nra % 64) == 0.
        The length-3 axis is {I,Q,U}.

    xpointing: float32 cupy array of shape (3, ndetectors, ntimes).
   
        The length-3 axis is {px_dec, px_ra, alpha}, where:

            - px_dec is declination in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - px_ra is right ascension in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - alpha is detector angle in radians.

        ('xpointing' is short for "exploded pointing", i.e. per-detector pointing)

    FIXME: currently calls cudaDeviceSynchronize() before and after kernel launch,
    which could slow things down.
    """

    check_tmx_args('gpu_map2tod()', tod_out, map_in, xpointing, gpu=True)

    ndet, nt = tod_out.shape
    ndec, nra = map_in.shape[1:]
    
    # _gpu_map2tod(tod, map, xpointing, ndet, nt, ndec, nra)
    _gpu_map2tod(tod_out.data.ptr, map_in.data.ptr, xpointing.data.ptr, ndet, nt, ndec, nra)


####################################################################################################


def reference_tod2map(map_out, tod_in, xpointing):
    """
    Slow (single-threaded CPU) reference implementation of tod2map().

    map_out: float32 numpy array of shape (3, ndec, nra).
        This array gets overwritten by reference_map2tod().
        Current implementation asserts (ndec % 64) == (nra % 64) == 0.
        The length-3 axis is {I,Q,U}.

    tod_out: float32 numpy array of shape (ndetectors, ntimes).
        Current implementation asserts (ntimes % 32) == 0.

    xpointing: float32 numpy array of shape (3, ndetectors, ntimes).
   
        The length-3 axis is {px_dec, px_ra, alpha}, where:

            - px_dec is declination in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - px_ra is right ascension in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - alpha is detector angle in radians.

        ('xpointing' is short for "exploded pointing", i.e. per-detector pointing)
    """

    check_tmx_args('reference_tod2map()', tod_in, map_out, xpointing, gpu=False)
    
    ndet, nt = tod_in.shape
    ndec, nra = map_out.shape[1:]

    # _reference_tod2map(map, tod, xpointing, ndet, nt, ndec, nra)
    _reference_tod2map(map_out.ctypes.data, tod_in.ctypes.data, xpointing.ctypes.data, ndet, nt, ndec, nra)


def gpu_tod2map(map_out, tod_in, xpointing, plan):
    """
    GPU implementation of tod2map().

    map_out: float32 numpy array of shape (3, ndec, nra).
        This array gets overwritten by gpu_map2tod().
        Current implementation asserts (ndec % 64) == (nra % 64) == 0.
        The length-3 axis is {I,Q,U}.

    tod_out: float32 numpy array of shape (ndetectors, ntimes).
        Current implementation asserts (ntimes % 32) == 0.

    xpointing: float32 numpy array of shape (3, ndetectors, ntimes).
   
        The length-3 axis is {px_dec, px_ra, alpha}, where:

            - px_dec is declination in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - px_ra is right ascension in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - alpha is detector angle in radians.

        ('xpointing' is short for "exploded pointing", i.e. per-detector pointing)

    plan: instance of 'class PointingPlan', see below.
    """

    assert isinstance(plan, PointingPlan)
    check_tmx_args('gpu_tod2map()', tod_in, map_out, xpointing, gpu=True)

    # Check consistency between plan and tod/map args
    assert plan.ncl_uninflated == ((tod_in.shape[0] * tod_in.shape[1]) // 32)
    assert plan.ndec == map_out.shape[1]
    assert plan.nra == map_out.shape[2]
    
    ndet, nt = tod_in.shape
    ndec, nra = map_out.shape[1:]

    # _gpu_tod2map(map, tod, xpointing, plan_cltod_list, plan_quadruples, plan_ncltod, plan_nquadruples, ndet, nt, ndec, nra)
    _gpu_tod2map(map_out.data.ptr, tod_in.data.ptr, xpointing.data.ptr,
                 plan.cltod_list.data.ptr, plan.quadruples.data.ptr,
                 plan.ncl_inflated, plan.num_quadruples, ndet, nt, ndec, nra)


####################################################################################################


class PointingPlan:
    def __init__(self, xpointing, ndec, nra, verbose=True):
        """
        The PointingPlan object must be precomputed from 'xpointing', before calling gpu_map2tod().
        It contains some cupy arrays which keep track of tiling info (see C++/cuda code for more info).

        The same PointingPlan may be used for multiple calls to gpu_map2tod(), if the 'xpointing'
        argument is the same for each call. For example, in a CG solver, you'd need one precomputed
        PointingPlan for each TOD, but you could re-use PointingPlans between CG iterations.

        The PointingPlans are ~30MB per TOD, and need 1.5 seconds(!) to compute.

        FIXME: the slow PointingPlan construction time is because I'm currently using slow
        single-threaded CPU code. I'll move plan construction to the GPU soon!

        ---------------------
        Constructor arguments
        ---------------------
        
        - xpointing: float32 numpy array of shape (3, ndetectors, ntimes).
   
          The length-3 axis is {px_dec, px_ra, alpha}, where:

            - px_dec is declination in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - px_ra is right ascension in "pixel units", i.e. in (0, ndec-1).
              WARNING: if an array element is outside this range, the kernel
              will either return garbage or segfault! (This can be fixed later.)

            - alpha is detector angle in radians.

            ('xpointing' is short for "exploded pointing", i.e. per-detector pointing)

        - ndec, nra: map dimensions

        -------------
        Class members
        -------------

          self.cltod_list: cupy array used by gpu_map2tod(), see C++/cuda code for more info
          self.quadruples: cupy array used by gpu_map2tod(), see C++/cuda code for more info
          self.ncl_uninflated: number of TOD cache lines (= ndet*nt/32)
          self.ncl_inflated: number of nonempty (TOD cache line, map tile) pairs.
          self.num_quadruples: number of nonempty map tiles
          self.ndec: map dimension (same meaning as constructor argument with same name)
          self.nra: map dimension (same meaning as constructor argument with same name)
        """

        func_name = "PointingPlan.__init__()"
        check_array(func_name, "'xpointing' argument", xpointing, 3, np.float32, gpu=False)
        
        if xpointing.shape[0] != 3:
            raise RuntimeError(f"{func_name}: {xpointing_arr.shape=}, expected (3,ndet,nt)")

        ndet, nt = xpointing.shape[1:]
        check_dims(func_name, ndet, nt, ndec, nra)
        
        # _construct_cpu_plan1(out, xpointing, ndet, nt, ndec, nra, verbose)
        out_step1 = np.zeros(4, dtype=int)
        _construct_cpu_plan1(out_step1.ctypes.data, xpointing.ctypes.data, ndet, nt, ndec, nra, verbose)

        cookie, self.ncl_uninflated, self.ncl_inflated, self.num_quadruples = out_step1
        self.cltod_list = np.zeros(self.ncl_inflated, dtype=np.int32)
        self.quadruples = np.zeros((self.num_quadruples,4), dtype=np.int32)
        
        # _construct_cpu_plan2(plan_cltod_list, plan_quadruples, cookie, ncl_inflated, num_quadruples)
        _construct_cpu_plan2(self.cltod_list.ctypes.data, self.quadruples.ctypes.data, cookie, self.ncl_inflated, self.num_quadruples)

        self.cltod_list = cp.asarray(self.cltod_list)   # copy CPU -> GPU
        self.quadruples = cp.asarray(self.quadruples)   # copy CPU -> GPU
        self.ndec = ndec
        self.nra = nra
