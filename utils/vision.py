from torchvision import transforms
import torch
vision_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def preprocess_obs_dict(obs):
    image = obs["image"] if "image" in obs else obs["rgb"]
    proprio = obs["proprio"] if "proprio" in obs else obs["agent"]  # depends on env config
    image_tensor = vision_tfm(image).float()
    proprio_tensor = torch.tensor(proprio, dtype=torch.float32)
    return image_tensor, proprio_tensor