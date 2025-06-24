from num_know import recognize_image as recog_num
from tool_know import recognize_tool as recog_tool
from struct_know import recognize_tool_or_digit as recog_str
import os
import pandas as pd
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix,f1_score,precision_score,recall_score
from tqdm import tqdm
import time
def recognize_more():


    # 原有路径   数字数据集没有用STRUCT/val/number  是为了保证训练过程中所使用的训练集与测试阶段使用的增强图像数据源彼此独立
    digit_image_dir_1 = 'output_digits/aug_all'
    digit_image_dir_2 = 'output_digits/output_digits_white'
    digit_image_dir_3 = 'output_digits/output_digits_black'

    tool_image_dir = 'data/STRUCT/val/tool'

    # 获取图像路径
    digit_images_1 = glob(os.path.join(digit_image_dir_1, '*.png'))
    digit_images_2 = glob(os.path.join(digit_image_dir_2, '*.png'))
    digit_images_3 = glob(os.path.join(digit_image_dir_3, '*.png'))
    tool_images = glob(os.path.join(tool_image_dir, '*.jpg'))

    # 合并所有图像路径
    digit_images = digit_images_1 + digit_images_2 + digit_images_3
    all_images = digit_images + tool_images

    # 构造 label map for digit_images_1（原逻辑）
    digit_label_map_1 = {
        os.path.basename(path): int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
        for path in digit_images_1
    }

    # 构造 label map for digit_images_2（读取 white_labels.csv）
    white_csv_path = os.path.join(digit_image_dir_2, 'white_labels.csv')
    df_white = pd.read_csv(white_csv_path)

    # 提取文件名（去掉路径）作为 key
    df_white['filename'] = df_white['filename'].apply(os.path.basename)
    digit_label_map_2 = dict(zip(df_white['filename'], df_white['label']))

    # 构造 label map for digit_images_2（读取 black_labels.csv）
    black_csv_path = os.path.join(digit_image_dir_3, 'black_labels.csv')
    df_black = pd.read_csv(black_csv_path)

    # 提取文件名（去掉路径）作为 key
    df_black['filename'] = df_black['filename'].apply(os.path.basename)
    digit_label_map_3 = dict(zip(df_black['filename'], df_black['label']))


    # 合并两个标签字典
    digit_label_map = {**digit_label_map_1, **digit_label_map_2, **digit_label_map_3}

    # 查看结果
    print(f"总共数字图像数：{len(digit_images)},工具图像数：{len(tool_images)}")


    # 保存预测结果和真实标签
    y_true = []
    y_pred = []


    # 遍历所有图像
    num=0
    time_total=0
    for img_path in tqdm(all_images,desc=num/len(all_images)):
        # 获取预测标签
        timer = time.time()
        pred_label = recognize(img_path)
        time_total=time_total+time.time()-timer

        # 获取真实标签
        filename = os.path.basename(img_path)

        if filename.endswith('.png'):  # 数字图像
            true_label = digit_label_map.get(filename, None)

        elif filename.endswith('.jpg'):  # 工具图像
            true_label = '_'.join(filename.split('_')[:-1])
        else:
            continue

        if true_label is None:
            print(f"找不到标签: {filename}")
            continue

        y_true.append(str(true_label))
        y_pred.append(str(pred_label))
        num+=1

    # 打印评估结果
    print("分类报告:")
    conf_matrix = classification_report(y_true, y_pred,output_dict=True, digits=4)

    print(f"平均耗时      :{time_total/num:.4f}s")
    weighted = conf_matrix['weighted avg']
    accuracy = conf_matrix['accuracy']
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Weighted Avg : Precision: {weighted['precision']:.4f}, Recall: {weighted['recall']:.4f}, F1: {weighted['f1-score']:.4f}")

def recognize(image):
    str_name=recog_str(image)
    if str_name=="TOOL":
        name=recog_tool(image)
        name=name.split("_",1)[-1]
    else:
        try:
            name=recog_num(image)
            if name>=100:
                s = str(name)
                name = int(s[0] + s[-1])

        except:
            name=0
    return name

if __name__=="__main__":

    # file = "single_pic/test2.png"
    # file = "data/STRUCT/val/tool/soldering_iron_008.jpg"
    # file = "output_digits/output_digits_black/black_img_0000.png"
    # name=recognize(file)
    # print(name)
    recognize_more()

