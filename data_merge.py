import os
import pickle

# 文件夹路径
FOLDER_PATH = 'data/'
OUTPUT_FILE = 'results/merged_data_mass.txt'  # 新的数据集保存路径

def load_data(file_path):
    """
    从pickle文件中加载数据。
    
    :param file_path: 数据集文件路径
    :return: 数据列表
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return []
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return []

def save_data(data, file_path):
    """
    将合并后的数据保存为pickle文件。
    
    :param data: 要保存的数据
    :param file_path: 文件路径
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已保存到 {file_path}")
    except Exception as e:
        print(f"保存数据时发生错误: {e}")

def main():
    # 获取文件夹中的所有文件
    all_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.txt')]  # 根据需要调整文件类型
    all_files.sort()  # 如果需要按字母顺序处理文件

    # 初始化合并后的数据
    merged_data = []

    # 初始化合并的文件数量和总数据量
    total_files = 0
    total_data_points = 0

    # 加载每个文件并合并
    for file_name in all_files:
        file_path = os.path.join(FOLDER_PATH, file_name)
        data = load_data(file_path)
        
        # 获取当前文件的数据量
        current_file_data_points = len(data)
        total_data_points += current_file_data_points
        
        # 将当前数据追加到合并数据中
        merged_data.extend(data)  
        total_files += 1  # 每处理一个文件，合并的文件数量加1
        
        # 打印当前文件的数据量和合并后的数据数量
        print(f"合并文件: {file_name}，数据量: {current_file_data_points}，当前合并总数据数量: {total_data_points}")

    # 打印合并后的总数据量和文件数量
    print(f"\n合并后的文件数量: {total_files}")
    print(f"合并后的总数据量: {total_data_points}")

    # 保存合并后的数据到新文件
    save_data(merged_data, OUTPUT_FILE)

if __name__ == "__main__":
    main()
