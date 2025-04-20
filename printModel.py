from torchsummary import summary
from Model import HTNet,HTNet_Enhanced
import sys

model = HTNet_Enhanced(
    image_size=28,
    patch_size=7,
    dim=256,
    heads=3,
    num_hierarchies=3,
    block_repeats=(2, 2, 10),
    num_classes=3
).to('cuda')  # 如果使用 GPU

summary(model, input_size=(3, 28, 28))  # 输入形状为 (channels, height, width)

# with open('model_struct.txt', 'w') as f:
#     sys.stdout = f
#     print(model)
#     summary(model, input_size=(3, 28, 28))  # 输入形状为 (channels, height, width)
# sys.stdout = sys.__stdout__