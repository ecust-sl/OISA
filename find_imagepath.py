import json

def find_image_path_by_id(json_data, target_id):
    # 假设 json_data 是一个字典，里面有一个 'test' 键，值是一个包含多个字典的列表
    for item in json_data.get('test', []):
        if item.get('id') == target_id:
            return item.get('image_path')
    return None

# 读取 JSON 文件
json_path = "/ssd/shilei/dataset/annotations/mimic_annotation.json"
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


# 设置你想要查找的 id
target_id = "018a20b6-6f0efbba-f043405f-e1af115c-a30fa5ed"

# 查找并输出结果
image_path = find_image_path_by_id(data, target_id)
if image_path:
    print(f"Image path for id {target_id}: {image_path}")
else:
    print(f"No image found for id {target_id}")
