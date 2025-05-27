import torch
print("PyTorch 버전:", torch.__version__)
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print("CUDA 버전:", torch.version.cuda)

import numpy as np
print(np.__file__)

import numpy as np
print("✅ numpy version:", np.__version__)
print("✅ type check:", type(np.ndarray))

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.new('RGB', (100, 100), color='blue')
arr = np.array(img)
plt.imshow(arr)
plt.title("Test Image")
plt.show()
