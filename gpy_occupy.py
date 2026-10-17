import torch
import time

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create large tensors to occupy memory (adjust size as needed)
n = 10000
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)

# Infinite loop to keep GPU computing/occupied
try:
    while True:
        # Perform continuous matrix multiplication
        c = torch.matmul(a, b)
        # Optional: sync to ensure execution finishes on device
        torch.cuda.synchronize()
except KeyboardInterrupt:
    print("Stopped by user.")
