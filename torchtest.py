import torch
from torchvision import transforms
from PIL import Image

# Step 1: Load the custom YOLOv5 model
#model_path = './yolo/mbari-mb-benthic-33k.pt'
model = torch.load('yolo/mbari-mb-benthic-33k.pt')
model.eval()

# Step 2: Load and preprocess the image
image_path = 'fishLL.png'
img = Image.open(image_path)
img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
])
img = img_transform(img)
img = img.unsqueeze(0)  # Add batch dimension

# Step 3: Perform inference
with torch.no_grad():
    output = model(img)

print(output)