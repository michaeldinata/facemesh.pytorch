import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
import numpy as np
import torch
import cv2
from network import FaceMesh

# gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# net = FaceMesh().to(gpu)
net = FaceMesh()
# net.load_weights("../trial_facemesh_10.pth")
net.load_weights("../facemesh.pth")

# img = cv2.imread("../dataset/training_data/images/thermal_image0.jpg")
img = cv2.imread("../test.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (192, 192))

detections = net.predict_on_image(img).numpy()
# detections *= 192
print(detections.shape)
print(detections)

plt.imshow(img, zorder=1)
x, y = detections[:, 0], detections[:, 1]
plt.scatter(x, y, zorder=2, s=1.0)
plt.savefig("test.png")
# plt.show()