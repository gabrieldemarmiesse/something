
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import (
    InputTensor,
    OutputTensor,
    foreach,
)
from utils.index import IndexList


@compiler.register("find_closest")
struct FindClosest:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor,
        x: InputTensor[dtype = outp.dtype, rank = 3], # fix dtype, should be float32
        w: InputTensor[dtype = outp.dtype, rank = 3], # fix dtype should be float32
        ctx: DeviceContextPtr,
    ) raises:
        pass
        
