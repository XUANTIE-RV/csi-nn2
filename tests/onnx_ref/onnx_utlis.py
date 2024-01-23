import onnx
# from onnx import helper, save_model
# from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime 
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from onnx.onnx_pb import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
)

# import onnx_tool 



def _extract_value_info(
    input: Union[List[Any], np.ndarray, None],
    name: str,
    type_proto: Optional[TypeProto] = None,
) -> onnx.ValueInfoProto:
    if type_proto is None:
        if input is None:
            raise NotImplementedError(
                "_extract_value_info: both input and type_proto arguments cannot be None."
            )
        elif isinstance(input, list):
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input[0].dtype]
            shape = None
            tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
            type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        elif isinstance(input, TensorProto):
            elem_type = input.data_type
            shape = tuple(input.dims)
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)
        else:
            elem_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input.dtype]
            shape = input.shape
            type_proto = onnx.helper.make_tensor_type_proto(elem_type, shape)

    return onnx.helper.make_value_info(name, type_proto)


def run(
    node: onnx.NodeProto,
    inputs: Sequence[np.ndarray],
    outputs: onnx.ValueInfoProto,
    name: str,
    **kwargs: Any,
) -> None:
    # Builds the model
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs["input_type_protos"]
        del kwargs["input_type_protos"]
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs["output_type_protos"]
        del kwargs["output_type_protos"]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = outputs
    graph = onnx.helper.make_graph(
        nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    kwargs["producer_name"] = "backend-test"

    if "opset_imports" not in kwargs:
        # To make sure the model will be produced with the same opset_version after opset changes
        # By default, it uses since_version as opset_version for produced models
        produce_opset_version = onnx.defs.get_schema(
            node.op_type, domain=node.domain
        ).since_version
        kwargs["opset_imports"] = [
            onnx.helper.make_operatorsetid(node.domain, produce_opset_version)
        ]

    model = onnx.helper.make_model_gen_version(graph, **kwargs)

    # Checking the produces are the expected ones.
    sess = onnxruntime.InferenceSession(model.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
    # onnx_tool.model_profile(model)
    feeds = {name: value for name, value in zip(node.input, inputs)}

    results = sess.run(None, feeds)




    return results
    