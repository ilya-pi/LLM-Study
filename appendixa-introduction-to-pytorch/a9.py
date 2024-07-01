import torch

print(f"CUDA is available == {torch.cuda.is_available()}")

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
print(tensor_1 + tensor_2)

# I am using mps (apple gpu support) instead of cuda
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

tensor_1 = tensor_1.to("mps")
tensor_2 = tensor_2.to("mps")

print(tensor_1 + tensor_2)

# The below should fail as tensors are on different devices
# tensor_1 = tensor_1.to("cpu")
# print(tensor_1 + tensor_2)

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
