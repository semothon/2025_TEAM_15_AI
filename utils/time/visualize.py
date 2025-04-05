import matplotlib.pyplot as plt
import numpy as np
import platform
import matplotlib.patches as mpatches
DAYS = ['월', '화', '수', '목', '금']
HOURS = ['9', '10', '11', '12', '1', '2', '3', '4', '5', '6', '7', '8']

# 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def visualize_free_mask(free_mask: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 9))  

    cax = ax.imshow(free_mask.astype(int), cmap='Greens', aspect='auto')

    ax.set_xticks(np.arange(len(DAYS)))
    ax.set_yticks(np.arange(len(HOURS)))
    ax.set_xticklabels(DAYS)
    ax.set_yticklabels(HOURS)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # 연한 격자선 추가
    ax.set_xticks(np.arange(-0.5, len(DAYS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(HOURS), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.4)

    legend_patch = mpatches.Patch(color='green', label='공통 공강')
    ax.legend(handles=[legend_patch], loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
