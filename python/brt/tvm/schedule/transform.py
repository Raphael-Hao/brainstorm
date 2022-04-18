# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.


"""CUDA implementations of transforms, adopted from TVM"""
import tvm
from tvm import te
from tvm.target import Target
from tvm.topi.utils import traverse_inline


def schedule_transpose(outs, input_dimx, input_dimy, sm_count=80):
    """Schedule a unfused transpose"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    warp_size = int(Target.current(allow_none=False).thread_warp_size)
    block_dim_x = input_dimx // warp_size
    block_dim_y = input_dimy // warp_size
    least_blocks = sm_count * 2
    while block_dim_y * block_dim_x < least_blocks:
        block_dim_y << 1
    schedule_transpose_from_existing(s, outs[0], block_dim_x)
    return s


def schedule_transpose_from_existing(s, out, block_dim_x):
    """Schedule for transpose on the gpu.

    Roughly follows this:
    https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/, but
    without the padding for shared memory. For better performance, we could
    rewrite it in tir to add the padding. Also, rewriting in tir would allow
    use to use warp shuffles instead of shared memory (see
    https://github.com/bryancatanzaro/trove).
    """

    def _callback(op):
        # pylint: disable=invalid-name
        m, n = s[op].op.axis
        warp_size = int(Target.current(allow_none=False).thread_warp_size)
        no, ni = s[op].split(n, factor=warp_size)
        print(no, ni)
        mo, mi = s[op].split(m, nparts=block_dim_x)
        s[op].reorder(mo, no, mi, ni)
        s[op].bind(mo, te.thread_axis("blockIdx.x"))
        s[op].bind(no, te.thread_axis("blockIdx.y"))
        c = s.cache_read(op.input_tensors[0], "shared", op)
        s[c].compute_at(s[op], no)
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")
        s[op].bind(ni, thread_x)
        # This is a hack to make the scheduling language realize that this axis
        # can be scheduled.
        a, _ = s[c].split(s[c].op.axis[1], factor=1)
        s[c].bind(a, thread_x)
        # Use 4 warps per block. Slightly faster than 1 warp per block
        ao, _ = s[op].split(mi, nparts=4)
        s[op].bind(ao, thread_y)
        ao, _ = s[c].split(s[c].op.axis[0], nparts=4)
        s[c].bind(ao, thread_y)

    traverse_inline(s, out.op, _callback)


def _invert_permutation_ir(data, out):
    """Low level IR to get invert_permutation.

    Parameters
    ----------
    data : Buffer
        Input data. 1-D Buffer with shape [elem_num].

    out : Buffer
        1D buffer for invert permutation result with the same shape with data.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    elem_num = data.shape[0]

    irb = tvm.tir.ir_builder.create()
    data = irb.buffer_ptr(data)
    out = irb.buffer_ptr(out)

    max_threads = int(Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = elem_num // max_threads + 1
    thread_x = te.thread_axis("threadIdx.x")
    block_x = te.thread_axis("blockIdx.x")
    irb.scope_attr(thread_x, "thread_extent", nthread_tx)
    irb.scope_attr(block_x, "thread_extent", nthread_bx)
    tid = block_x * max_threads + thread_x

    with irb.if_scope(tid < elem_num):
        r_ind = data[tid]
        out[r_ind] = tid
    return irb.get()


def invert_permutation(data):
    """Compute definition of invert_permutation.
    For an output tensor y and an input tensor x, this operation computes the following:

       y[x[i]] = i for i in [0, 1, ..., len(x) - 1]

    Parameters
    ----------
    data : tvm.te.Tensor
        1-D tensor

    Returns
    -------
    out : tvm.te.Tensor
    """
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "out_buf", data_alignment=8)

    out = te.extern(
        [data.shape],
        [data],
        lambda ins, outs: _invert_permutation_ir(ins[0], outs[0]),
        in_buffers=[
            data_buf,
        ],
        out_buffers=[
            out_buf,
        ],
        name="invert_permutation",
        tag="invert_permutation_gpu",
    )
    return out
