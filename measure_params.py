import numpy as np
import torch
import time
from model import CNN_network
from thop import profile
from ptflops import get_model_complexity_info

device = torch.device("cuda")
cnn = CNN_network.CNNet(150, batch_size=8, num_class=5, epochs=100)
cnn.to(device)
cnn.eval()

dummy_input = torch.randn(1, 3, 150, 150).to(device)

macs, params = profile(cnn, inputs=(dummy_input, ), verbose=False)
flops = 2 * macs
print(f"MACs:  {macs/1e6:.2f} M")
print(f"FLOPs: {flops/1e9:.2f} G")
print(f"Params:{params/1e6:.2f} M")

with torch.cuda.device(0):
    macs, params = get_model_complexity_info(cnn, (3, 150, 150), as_strings=False,
                                           print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# CPU latency
device = torch.device("cpu")
cnn.to(device)

dummy_input = torch.randn(1, 3, 150, 150).to(device)

# Warm up
with torch.no_grad():
    for _ in range(20):
        _ = cnn(dummy_input)

n_runs = 100
times = []

with torch.no_grad():
    for _ in range(n_runs):
        start_time = time.time()
        _ = cnn(dummy_input)
        end_time = time.time()
        times.append(end_time - start_time)

times = np.array(times)
print(f"Mean latency: {times.mean()*1000:.2f} ms")
print(f"Std:          {times.std()*1000:.2f} ms")
print(f"Min:          {times.min()*1000:.2f} ms")