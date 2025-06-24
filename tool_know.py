from tool_model import ToolNet,ResNet50Mine
import torch
from PIL import Image
from torchvision import transforms

num_classes=15
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=ResNet50Mine(num_classes=num_classes).to(device)
pretrained_dict = torch.load(f"model/TOOL/{model.__class__.__name__}best.pth")
model.load_state_dict(pretrained_dict)
class_names=['10_keyboard', '11_mobile_phone', '12_mouse', '13_headphones', '14_monitor', '15_speaker', '1_wrench', '2_soldering_iron', '3_electrodrill', '4_tape_measure', '5_screwdriver', '6_pliers', '7_oscillograph', '8_multimeter', '9_printer']
def recognize_tool(image):
    image=Image.open(image).convert("RGB")
    input_tensor=transform(image).unsqueeze(0)
    input_tensor=input_tensor.to(device)
    model.eval()
    output=model(input_tensor)
    _, preds = torch.max(output, 1)
    pred_name=class_names[preds.item()]


    return pred_name




if __name__=="__main__":
    file="data/STRUCT/val/tool/mouse_005.jpg"
    # file="data/STRUCT/val/tool/soldering_iron_008.jpg"
    recognize_name=recognize_tool(file)
    print(recognize_name)

