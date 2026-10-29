import os
import onnx
from onnx import numpy_helper
import numpy as np

cur_dir = "D:\\Git-Project\\mjlab\\src\\mjlab\\tasks\\velocity\\config\\balance_car_v2\\sim2real"
file_name = "2026-10-27_11-58-34.onnx"
weight_file = "model_weights.h"

model = onnx.load(os.path.join(cur_dir, file_name))
graph = model.graph
# 生成C头文件
with open(os.path.join(cur_dir, weight_file), "w", encoding="utf-8") as f:
    f.write("// Auto-generated ONNX weights to C code\n")
    f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    
    for tensor in graph.initializer:
        # 读取权重原始数据
        data = np.frombuffer(tensor.raw_data, dtype=np.float32)
        shape = tuple(d for d in tensor.dims)
        var_name = tensor.name.replace(".", "_").replace("/", "_")
        
        # 生成C数组定义
        f.write(f"// Tensor shape: {shape}\n")
        f.write(f"const float {var_name}[{data.size}] = {{\n")
        
        # 按行格式化输出数值
        for i, val in enumerate(data):
            if i % 12 == 0:
                f.write("    ")
            f.write(f"{val:.4e}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("};\n\n")
    
    f.write("#endif // MODEL_WEIGHTS_H\n")