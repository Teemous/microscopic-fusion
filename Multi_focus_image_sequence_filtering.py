import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import time

def brenner_score(img):
    img = img.astype(np.float32)
    diff = img[:, 2:] - img[:, :-2]
    score = np.sum(diff**2)
    return score

def crop_image(path_list,crop_num=3):
    all_files = sorted(os.listdir(path_list))
    valid_paths = []
    for f in all_files:
        if f.lower().endswith(".jpg") and '_fo' not in f.lower():
            valid_paths.append(os.path.join(path_list, f)) # 去除不符条件图像的地址列表

    img_stack = []

    first_path = os.path.join(path_list,os.listdir(path_list)[0])
    first_img = cv.imread(first_path,cv.IMREAD_GRAYSCALE)
    H,W = first_img.shape

    for p in valid_paths:
        img_ = cv.imread(p, cv.IMREAD_GRAYSCALE)
        if img_ is None: continue
        assert img_.shape == (H, W), f"尺寸不符: {p}"
        img_stack.append(img_)
    #print(len(img_stack))
    N = len(img_stack)
    # 进行图像裁剪
    block_H = (H // crop_num) + 1
    block_W = (W // crop_num) + 1

    all_blocks = [[[ [] for _ in range(N)] for _ in range(crop_num)] for _ in range(crop_num)]
    sharpness_matrix = [[[0.0 for _ in range(N)] for _ in range(crop_num)] for _ in range(crop_num)]
    block = []
    for n in range(N): # 层
        current_img = img_stack[n]
        for i in range(crop_num): # 行
            for j in range(crop_num): # 列
                # 计算切片范围
                if j < crop_num - 1:
                    x_start = j * block_W
                    x_end = (j+1) * block_W
                else:
                    x_start = W - block_W
                    x_end = W
                if i < crop_num - 1:
                    y_start = i * block_H
                    y_end = (i+1) * block_H
                else:
                    y_start = H - block_H
                    y_end = H
                crop_part = current_img[y_start:y_end,x_start:x_end]
                # 存储切片
                all_blocks[i][j][n] = crop_part
                # 计算清晰度
                sharpness_matrix[i][j][n] = brenner_score(crop_part)

    return all_blocks,sharpness_matrix,valid_paths


def selection(scores_block):
    # 获取行列数
    rows = len(scores_block)  # 3
    cols = len(scores_block[0])  # 3

    # 显式地创建一个新的 Figure 对象，不要使用全局 plt
    fig = plt.figure(figsize=(12, 10))

    for i in range(rows):  # 行
        for j in range(cols):  # 列
            idx = i * cols + j + 1

            ax = fig.add_subplot(rows, cols, idx)  # 使用 ax 对象绘图
            ax.plot(scores_block[i][j])

            ax.set_title(f"Block [{i},{j}]")
            ax.set_xlabel("Index")
            ax.set_ylabel("Score")

    fig.tight_layout()  # 自动调整布局，防止标题重叠

    # 返回创建好的画布对象
    return fig

def extract_best_indices(scores_block, top_k=3):
    rows = len(scores_block)
    cols = len(scores_block[0])
    num_images = len(scores_block[0][0])
    scores_array = np.array(scores_block)

    local_peaks = [] # 存储最清晰帧
    for i in range(rows):
        for j in range(cols):
            local_peaks.append(np.argmax(scores_array[i, j, :])) # 找到最清晰帧

    #看看哪些索引被各区域选中的次数最多
    unique_peaks, counts = np.unique(local_peaks, return_counts=True) # 找出数组唯一元素和各元素出现次数

    sorted_unique_peaks = unique_peaks[np.argsort(counts)[::-1]] # 按名次取回对应索引号
    print(unique_peaks,counts)
    #return sorted_unique_peaks[:top_k]
    return sorted_unique_peaks

def extract_best_indices_cluster(scores_block,distance_threshold=1,vote_threshold=2):
    rows = len(scores_block)
    cols = len(scores_block[0])
    num_images = len(scores_block[0][0])
    scores_array = np.array(scores_block)

    local_peaks = [] # 存储最清晰帧
    for i in range(rows):
        for j in range(cols):
            local_peaks.append(np.argmax(scores_array[i, j, :])) # 找到最清晰帧

    #看看哪些索引被各区域选中的次数最多
    unique_peaks, counts = np.unique(local_peaks, return_counts=True) # 找出数组唯一元素和各元素出现次数

    sort_idx = np.argsort(unique_peaks) # 返回排序的索引
    unique_peaks = unique_peaks[sort_idx] # 按大小排序
    counts = counts[sort_idx] # 按排序后的顺序重排

    clusters = [] # 簇
    current_cluster = [(unique_peaks[0],counts[0])] # 当前簇存入第一个元素
    for i in range(1,len(unique_peaks)):
        if unique_peaks[i] - unique_peaks[i-1] <= distance_threshold: # 距离小于阈值则判定为同一簇
            current_cluster.append((unique_peaks[i],counts[i]))
        else: # 距离大于阈值则视为不同簇
            clusters.append(current_cluster) # 把同一簇的压入簇中
            current_cluster = [(unique_peaks[i],counts[i])] # 刷新当前簇
    clusters.append(current_cluster)

    final_indices = []
    for cluster in clusters:
        # 筛选出该簇中达到或超过投票阈值的成员
        high_vote_members = [idx for idx, count in cluster if count >= vote_threshold]

        if high_vote_members:
            # 逻辑：如果簇里有超过阈值的，全部保存
            final_indices.extend(high_vote_members)
        else:
            # 逻辑：如果没有超过阈值的，找该簇中投票数最大的那一个代表
            best_in_cluster = max(cluster, key=lambda x: x[1])[0]
            final_indices.append(best_in_cluster)
    #print(final_indices)

    # 去重并排序返回
    return sorted(list(set(final_indices)))

def pipeline(path_list,output_path):
    for folder_path in path_list:
        if os.path.isdir(folder_path):
            start_time = time.time()

            result_blocks,scores, valid_paths = crop_image(folder_path, crop_num=3) # 切片矩阵 层行列
            stack_len = len(valid_paths)
            best_indices = extract_best_indices_cluster(scores,distance_threshold=1)
            # 创建输出文件夹
            folder_name = os.path.basename(folder_path)
            # 000_image
            save_dir = os.path.join(output_path, folder_name)
            # out\000_image
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            fig_plot = selection(scores)
            plot_save_path = os.path.join(save_dir, "focus_curves.png")
            fig_plot.savefig(plot_save_path)
            #print(f"已保存清晰度曲线: {plot_save_path}")
            plt.close(fig_plot)

            for idx in best_indices:
                # 找对应的真实文件路径
                real_file_path = valid_paths[idx]
                #print(real_file_path)
                file_name = os.path.basename(real_file_path)

                # 读取原图并保存
                img_to_save = cv.imread(real_file_path)
                save_name = os.path.join(save_dir, file_name)
                cv.imwrite(save_name, img_to_save)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"文件{folder_path}处理完成，共{stack_len}张图像,处理时间{execution_time}秒")


if __name__ == '__main__':
    root_dir = "Image"
    output_path = 'out'
    img_seq_path = [os.path.join(root_dir,f) for f in os.listdir(root_dir)] # 文件目录
    pipeline(img_seq_path,output_path)
    print("全部处理完成")