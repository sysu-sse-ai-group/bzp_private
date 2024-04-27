import os

def find_sources_directories(root):
    # 这个列表用来存储名为 "sources" 的目录的路径
    sources_directories = []
    # os.walk 通过在目录树中游走（从顶部向下或从底部向上），生成目录中的文件名。
    for dirpath, dirnames, filenames in os.walk(root):
        # 检查 'sources' 是否是 dirpath 的直接子目录
        if 'sources' in dirnames:
            # 构建到 'sources' 目录的完整路径
            full_path = os.path.join(dirpath, 'sources')
            sources_directories.append(full_path)

    return sources_directories

# 示例使用：
root_directory = 'G:\data'  # 根据需要更新此路径
sources_dirs = find_sources_directories(root_directory)
print(len(sources_dirs))
for path in sources_dirs:
    print(path)

