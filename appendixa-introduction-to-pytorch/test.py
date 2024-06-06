import torch

print(f"PyTorch version {torch.__version__}, CUDA support == {torch.cuda.is_available()}, is Apple Silicon chip acceleration supported {torch.backends.mps.is_available()}")

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1, 2, 3])

tensor2d = torch.tensor([[1, 2], [3, 4]])

tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [8, 8]]])

print(f"tensor0d = {tensor0d}, tensor1d = {tensor1d}, tensor2d = {tensor2d}, tensor3d = {tensor3d}")

tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)

floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)

floatvec = tensor1d.to(torch.float32)
print(f"After changing tensor's precision to float {floatvec.dtype}")


print("\n\n\n")


tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"tensor {tensor2d} of shape {tensor2d.shape}")

print(f"after reshaping it is {tensor2d.reshape(3, 2)}")

print(f"but a more common view is {tensor2d.view(3, 2)}")

print(f"here it is transposed with .T {tensor2d.T}")

print(f"Multiplying on itself transposed is {tensor2d.matmul(tensor2d.T)}")

print(f"or more shortly with @ operator {tensor2d @ tensor2d.T}")
