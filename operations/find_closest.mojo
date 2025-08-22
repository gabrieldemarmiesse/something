
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import (
    InputTensor,
    OutputTensor,
)
from layout import Layout, LayoutTensor
from gpu.host import DeviceContext
from gpu.id import block_idx, thread_idx, block_dim
from gpu.memory import AddressSpace
from gpu.sync import barrier


@compiler.register("find_closest")
struct FindClosest:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor[dtype = DType.int64, rank = 2],
        x: InputTensor[dtype = DType.float32, rank = 3],
        w: InputTensor[dtype = DType.float32, rank = 3],
        ctx: DeviceContextPtr,
    ) raises:
        # Convert to LayoutTensor for easier manipulation
        X = x.to_layout_tensor()
        W = w.to_layout_tensor()
        O = outp.to_layout_tensor()
        
        if target == "cpu":
            find_closest_cpu(X, W, O)
        else:
            dev_ctx = ctx.get_device_context()
            find_closest_gpu(dev_ctx, X, W, O)


@always_inline
fn find_closest_cpu(
    X: LayoutTensor, W: LayoutTensor, mut O: LayoutTensor
):
    alias H = X.shape[0]()
    alias N = X.shape[1]()
    alias D = X.shape[2]()
    alias C = W.shape[1]()
    
    for h in range(H):
        for n in range(N):
            var min_dist = Float32(1e10)
            var min_idx = 0
            
            for c in range(C):
                var dist = Float32(0)
                
                for d in range(D):
                    var x_val = rebind[Float32](X[h, n, d].cast[DType.float32]())
                    var w_val = rebind[Float32](W[h, c, d].cast[DType.float32]())
                    var diff = x_val - w_val
                    dist += diff * diff
                
                if dist < min_dist:
                    min_dist = dist
                    min_idx = c
            
            O[h, n] = min_idx


fn find_closest_kernel[
    x_dtype: DType,
    x_layout: Layout,
    w_dtype: DType,
    w_layout: Layout,
    o_dtype: DType,
    o_layout: Layout,
](
    X: LayoutTensor[x_dtype, x_layout, MutableAnyOrigin],
    W: LayoutTensor[w_dtype, w_layout, MutableAnyOrigin],
    O: LayoutTensor[o_dtype, o_layout, MutableAnyOrigin],
):
    alias H = X.shape[0]()
    alias N = X.shape[1]()
    alias D = X.shape[2]()
    alias C = W.shape[1]()
    
    # Each thread processes one element in the output
    var h = block_idx.y
    var n = block_idx.x * block_dim.x + thread_idx.x
    
    if n < N:
        var min_dist = Float32(1e10)
        var min_idx = 0
        
        # We could do fancy reductions with shared memory here
        for c in range(C):
            var dist = Float32(0)
            
            for d in range(D):
                var x_val = rebind[Float32](X[h, n, d].cast[DType.float32]())
                var w_val = rebind[Float32](W[h, c, d].cast[DType.float32]())
                var diff = x_val - w_val
                dist += diff * diff
            
            if dist < min_dist:
                min_dist = dist
                min_idx = c
        
        O[h, n] = min_idx


def find_closest_gpu(
    ctx: DeviceContext,
    X: LayoutTensor,
    W: LayoutTensor,
    mut O: LayoutTensor,
):
    alias kernel_func = find_closest_kernel[
        X.dtype,
        X.layout,
        W.dtype,
        W.layout,
        O.dtype,
        O.layout,
    ]
    
    alias H = X.shape[0]()
    alias N = X.shape[1]()
    alias threads_per_block = 256
    alias blocks_x = (N + threads_per_block - 1) // threads_per_block
    
    ctx.enqueue_function[kernel_func](
        X,
        W,
        O,
        grid_dim=(blocks_x, H),
        block_dim=(threads_per_block),
    )
        
