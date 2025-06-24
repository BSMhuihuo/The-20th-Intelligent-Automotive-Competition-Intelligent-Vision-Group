import os
import shutil
from sklearn.model_selection import train_test_split

digit_dir = r"D:\Desktop\CV\code\output_digits"
tool_root_dir = r"D:\Desktop\CV\code\data\TOOL"

output_root = r"data/STRUCT"
os.makedirs(output_root, exist_ok=True)

def collect_images(src_dir, exts):
    return [os.path.join(root, f)
            for root, _, files in os.walk(src_dir)
            for f in files if f.lower().endswith(exts)]

# 收集图片路径
digit_images = collect_images(digit_dir, (".png",))
tool_images = collect_images(tool_root_dir, (".jpg",))

# 分割训练验证
digit_train, digit_val = train_test_split(digit_images, test_size=0.2, random_state=42)
tool_train, tool_val = train_test_split(tool_images, test_size=0.2, random_state=42)

def copy_images(file_list, label, phase):
    target_dir = os.path.join(output_root, phase, label)
    os.makedirs(target_dir, exist_ok=True)
    for file in file_list:
        shutil.copy(file, os.path.join(target_dir, os.path.basename(file)))

# 拷贝图片
copy_images(digit_train, "digit", "train")
copy_images(digit_val, "digit", "val")
copy_images(tool_train, "tool", "train")
copy_images(tool_val, "tool", "val")
