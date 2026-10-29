import torch
from tensordict import TensorDict
import numpy as np
import onnx
import onnxruntime as ort
from onnx import checker
from rsl_rl.models import MLPModel

def verify_onnx_consistency(pytorch_model, onnx_path, dummy_input):
    try:
        onnx_model = onnx.load(onnx_path)
        checker.check_model(onnx_model)
        print("[PASS] ONNX 结构检查通过")
    except Exception as e:
        print(f"[FAIL] ONNX 结构检查失败: {e}")
        return False
    
    pytorch_model.eval()  # 确保处于评估模式，关闭 Dropout/BN 训练行为
    with torch.no_grad():
        pt_input = dummy_input.clone()
        pt_output = pytorch_model(pt_input)
        
        if isinstance(pt_output, tuple):
            pt_output = pt_output[0]
        pt_output_np = pt_output.cpu().numpy()
        print(pt_output_np)

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: dummy_input['actor'].cpu().numpy()})
    ort_output_np = ort_outputs[0]
    print(ort_output_np)

if __name__ == "__main__":
    device = 'cpu'
    obs = TensorDict(
        {
            "actor": torch.zeros(1, 12, device=device),
        },
        batch_size=[1],
        device=device
    )
    obs_groups = {
        'actor': ['actor',],
    }
    distribution_cfg = {
        'class_name': 'GaussianDistribution',
        'init_std': 1.0,
        'std_type': 'scalar'
    }
    model_path = '/home/yun/data/git_projects/mjlab/src/mjlab/tasks/velocity/config/balance_car_v2/model_999.pt'
    model = MLPModel(obs=obs, obs_groups=obs_groups, obs_set='actor', output_dim=2, hidden_dims=(24,12), obs_normalization=True, distribution_cfg=distribution_cfg)
    loaded_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(loaded_dict['actor_state_dict'], strict=True)
    
    # 2. 准备虚拟输入 (需与导出时的 dummy_input 形状一致)
    # tensor([[ 0.5355, -0.5736]], device='cuda:0')
    dummy = TensorDict(
        {
            "actor": torch.tensor([[-1.0321e-02, -1.7328e-01,  3.8765e-01,  2.6002e-02,  4.7311e-05,
                                    -9.9966e-01,  8.3297e+00,  6.8149e+00,  5.3278e-01, -5.7659e-01,
                                    -4.4091e-01,  3.7685e-01]]),
            # "actor": torch.randn(1, 12, device=device)
        },
        batch_size=[1],
        device=device
    )
    
    # 3. 执行验证
    onnx_model_path = "/home/yun/data/git_projects/mjlab/src/mjlab/tasks/velocity/config/balance_car_v2/2026-10-27_11-58-34.onnx"
    is_consistent = verify_onnx_consistency(model, onnx_model_path, dummy)