import torch
from networks.models import DeepAutoEncoder, Encoder
import cv2


device = torch.device("cuda")

# AutoEncoder inference #

# Image to tensor
image = cv2.imread('dataset/f8/89.jpg', 0)
image = cv2.resize(image, (128,128))
#image = image.transpose(2,0,1)
image = image[None,None,:,:]
image = torch.from_numpy(image).cuda().float()

# Load model
autoencoder = DeepAutoEncoder()
autoencoder.load_state_dict(torch.load("AutoEncoder.pth"))
autoencoder.to(device)

# Inference
reconstructed = autoencoder(image)[0][0].detach().cpu().numpy()
cv2.imwrite("results/reconstructed.png", reconstructed * 255)


# Deep SVDD inference #

# if score > 0: Anomaly, if score < 0: Normal
# if your encoder loss was really low (<0.3) that should work if it was 0.5, 0.6 or higher
# then you probably need to verify whats the highest score between your good samples,
# and take it as threshold (usually it will be like 0.1, 0.01, but thats higher than 0)
# in my trainings I often see 0.0x as the highest
print("Good sample inference:")

# Load model
model_dict = torch.load("DeepSVD_f8.tar")
center = model_dict["center"]
radius = model_dict["radius"]
weights = model_dict["encoder_weights"]
model = Encoder()
model.load_state_dict(weights)
model.to(device)

# Inference
outputs = model(image)[0]
distance = torch.sum((outputs - center) ** 2)
final_scores = distance - (radius ** 2)
print(f"Score: {final_scores}")
# ------------------------- #

import os

for subdir, dirs, files in os.walk("test"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        image = cv2.imread(filepath, 0)
        image = cv2.resize(image, (128,128))
        image = image[None,None,:,:]
        image = torch.from_numpy(image).cuda().float()
        outputs = model(image)[0]

        distance = torch.sum((outputs - center) ** 2)
        final_scores = distance - (radius ** 2)
        print(f"Score for {file}: {final_scores}")
