import torch

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    GPUが使用可能かどうかを判定し、適切なtorch.deviceを返す。

    Parameters:
        prefer_gpu (bool): Trueの場合、GPUが使えるならGPUを、使えないならCPUを返す。
                           Falseなら常にCPUを使う。

    Returns:
        torch.device: 使用すべきデバイスオブジェクト（cuda or cpu）
    """
    if prefer_gpu and torch.cuda.is_available():
        print(f"✅ GPU利用可能: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ GPU未使用（CPUにフォールバック）")
        return torch.device("cpu")
