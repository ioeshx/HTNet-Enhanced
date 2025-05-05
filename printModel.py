from torchsummary import summary
from Model import *
import sys

# model = HTNet_Enhanced_v4(
#     image_size=28,
#     patch_size=7,
#     dim=256,  # 256,--96, 56-, 192
#     heads=3,  # 3 ---- , 6-
#     num_hierarchies=3,  # 3----number of hierarchies
#     block_repeats=(2, 2, 10),#(2, 2, 8),------
#     # the number of transformer blocks at each heirarchy, starting from the bottom(2,2,20) -
#     num_classes=3
# ).to('cuda')  # 如果使用 GPU

model = HTNet(
    image_size=28,
    patch_size=7,
    dim=256,  # 256,--96, 56-, 192
    heads=3,  # 3 ---- , 6-
    num_hierarchies=3,  # 3----number of hierarchies
    block_repeats=(2, 2, 10),#(2, 2, 8),------
    # the number of transformer blocks at each heirarchy, starting from the bottom(2,2,20) -
    num_classes=3
)

summary(model, input_size=(3, 28, 28))  # 输入形状为 (channels, height, width)

# with open('model_struct.txt', 'w') as f:
#     sys.stdout = f
#     print(model)
#     summary(model, input_size=(3, 28, 28))  # 输入形状为 (channels, height, width)
# sys.stdout = sys.__stdout__