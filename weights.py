import torch

path = "__weights/tmodel_00.pt"   # change to your file
obj = torch.load(path, map_location="cpu")  # load on CPU safely

print(type(obj))

if isinstance(obj, dict):
    print("Top-level keys:", list(obj.keys())[:20])

    # Support common checkpoint layouts.
    if "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    elif "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj

    tensor_items = [(name, tensor) for name, tensor in state_dict.items() if torch.is_tensor(tensor)]
    total_params = sum(tensor.numel() for _, tensor in tensor_items)

    print(f"Number of tensors: {len(tensor_items)}")
    print("Tensor shapes:")
    for name, tensor in tensor_items:
        print(f"- {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, params={tensor.numel()}")

    print(f"Total parameters: {total_params}")
else:
    print("Loaded object is not a dict checkpoint, cannot extract state_dict.")