
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import (
    InputTensor,
    OutputTensor,
    foreach,
)
from utils.index import IndexList


@compiler.register("add_constant_custom")
struct AddConstantCustom[value: Int]:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        outp: OutputTensor,
        x: InputTensor[dtype = outp.dtype, rank = 3],
        w: InputTensor[dtype = outp.dtype, rank = 3],
        ctx: DeviceContextPtr,
    ) raises:
        pass
        
