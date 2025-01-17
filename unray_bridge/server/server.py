import torch
print(torch.cuda.get_device_name(0))
print(f"Memoria disponible: {torch.cuda.memory_allocated(0) / 1e9} GB")
print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
