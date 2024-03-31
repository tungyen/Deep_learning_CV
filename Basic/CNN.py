import torch
inputC = 5
outputC = 10
w, h = 100, 100
kernelSize = 3
batchSize = 1

input = torch.randn(batchSize, inputC, w, h)
conv_layer = torch.nn.Conv2d(inputC, outputC, kernel_size=kernelSize)
output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)