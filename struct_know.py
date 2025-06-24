import torch
from PIL import Image
from torchvision import transforms
from struct_model import MyCNN
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=MyCNN().to(device)
class_name=["DIGIT","TOOL"]

pred_train_dict=torch.load("model/STRUCT/tool_vs_digit3.pth")
model.load_state_dict(pred_train_dict)

def recognize_tool_or_digit(image):
    model.eval()
    image=Image.open(image).convert("RGB")
    image=transform(image).unsqueeze(0)
    image=image.to(device)
    output=model(image)
    _,pred=torch.max(output,1)
    recognize_name=class_name[pred.item()]
    return recognize_name

if __name__=="__main__":
    # file="data/STRUCT/val/tool/mouse_005.jpg"
    # file = "data/STRUCT/val/digit/img_0010.png"
    file = "single_pic/test2.png"
    recognize_name=recognize_tool_or_digit(file)
    print(recognize_name)

