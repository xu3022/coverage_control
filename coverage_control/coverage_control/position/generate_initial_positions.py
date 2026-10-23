#!/usr/bin/env python3
"""
初始位置生成器
用于手动生成各个智能体的初始位置，并保存到position目录中
"""

import argparse
import os
import random
import numpy as np
from pathlib import Path

# 配置参数
NUM_ROBOTS = 20  # 智能体数量
AREA_WIDTH = 10.0  # 区域宽度
AREA_HEIGHT = 10.0  # 区域高度
MIN_DISTANCE = 0.1  # 智能体间最小距离
OUTPUT_PATH = str(Path(__file__).resolve().parent / "initial_positions.txt")

# 工作区域边界（确保位置在(0,0)到(10,10)之间）
MIN_X = 0.0
MAX_X = 10.0
MIN_Y = 0.0
MAX_Y = 10.0

def generate_random_positions(num_robots, area_width, area_height, min_distance,
                              boundary_margin=0.0):
    """生成随机初始位置，并可在工作区域边界预留安全余量。"""
    positions = []
    
    for i in range(num_robots):
        max_attempts = 100  # 最大尝试次数
        for attempt in range(max_attempts):
            # 生成随机位置（在(0,0)到(10,10)之间）
            x = random.uniform(MIN_X + boundary_margin,
                               MAX_X - boundary_margin)
            y = random.uniform(MIN_Y + boundary_margin,
                               MAX_Y - boundary_margin)
            
            # 检查是否满足最小距离要求
            valid_position = True
            for existing_pos in positions:
                distance = np.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if distance < min_distance:
                    valid_position = False
                    break
            
            if valid_position:
                positions.append((x, y))
                break
            
            if attempt == max_attempts - 1:
                print(f"警告: 无法为机器人 {i+1} 找到满足最小距离要求的位置")
                positions.append((x, y))  # 使用最后生成的位置
    
    return positions

def generate_grid_positions(num_robots, area_width, area_height):
    """生成网格状初始位置"""
    positions = []
    
    # 计算网格的行列数
    grid_size = int(np.ceil(np.sqrt(num_robots)))
    
    # 计算网格间距（在(0,0)到(10,10)之间）
    spacing_x = area_width / (grid_size + 1)
    spacing_y = area_height / (grid_size + 1)
    
    # 生成网格位置
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= num_robots:
                break
            x = MIN_X + (j + 1) * spacing_x
            y = MIN_Y + (i + 1) * spacing_y
            positions.append((x, y))
            count += 1
        if count >= num_robots:
            break
    
    return positions

def generate_circle_positions(num_robots, radius=4.0):
    """生成圆形初始位置"""
    positions = []
    
    # 圆心在(5,5)，确保所有点在(0,0)到(10,10)之间
    center_x = 5.0
    center_y = 5.0
    
    for i in range(num_robots):
        angle = 2 * np.pi * i / num_robots
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # 确保位置在边界内
        x = max(MIN_X, min(MAX_X, x))
        y = max(MIN_Y, min(MAX_Y, y))
        
        positions.append((x, y))
    
    return positions

def save_positions_to_file(positions, output_path, seed=None,
                           boundary_margin=0.0):
    """将位置信息保存到文件"""
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# 智能体初始位置文件\n")
        f.write("# 格式: 机器人ID X坐标 Y坐标\n")
        f.write(f"# 总机器人数量: {len(positions)}\n")
        if seed is not None:
            f.write(f"# random_seed: {seed}\n")
        if boundary_margin:
            f.write(f"# boundary_margin: {boundary_margin}\n")
        f.write("\n")
        
        for i, (x, y) in enumerate(positions):
            f.write(f"robot{i+1} {x:.3f} {y:.3f}\n")
    
    print(f"初始位置已保存到: {output_path}")
    print(f"共生成 {len(positions)} 个智能体的位置")

def main(argv=None):
    """主函数"""
    parser = argparse.ArgumentParser(description='生成机器人初始位置')
    parser.add_argument('--mode', choices=('random', 'grid', 'circle'))
    parser.add_argument('--seed', type=int, help='随机布局种子')
    parser.add_argument('--output', default=OUTPUT_PATH, help='位置文件路径')
    parser.add_argument('--boundary-margin', type=float, default=0.0,
                        help='随机位置距离边界的最小余量，单位 m')
    args = parser.parse_args(argv)
    if not 0 <= args.boundary_margin < min(AREA_WIDTH, AREA_HEIGHT) / 2:
        parser.error('--boundary-margin 必须在 [0, 5) m 内')

    if args.seed is not None:
        random.seed(args.seed)

    if args.mode is None:
        print("=== 智能体初始位置生成器 ===")
        print("请选择生成模式:")
        print("1. 随机分布 (确保最小距离)")
        print("2. 网格分布")
        print("3. 圆形分布")

        try:
            choice = int(input("请输入选择 (1-3): ").strip())
        except ValueError:
            print("输入无效，使用默认模式: 随机分布")
            choice = 1
    else:
        choice = {'random': 1, 'grid': 2, 'circle': 3}[args.mode]
    
    if choice == 1:
        print("生成随机分布位置...")
        positions = generate_random_positions(
            NUM_ROBOTS, AREA_WIDTH, AREA_HEIGHT, MIN_DISTANCE,
            args.boundary_margin,
        )
    elif choice == 2:
        print("生成网格分布位置...")
        positions = generate_grid_positions(NUM_ROBOTS, AREA_WIDTH, AREA_HEIGHT)
    elif choice == 3:
        print("生成圆形分布位置...")
        positions = generate_circle_positions(NUM_ROBOTS)
    else:
        print("无效选择，使用默认模式: 随机分布")
        positions = generate_random_positions(
            NUM_ROBOTS, AREA_WIDTH, AREA_HEIGHT, MIN_DISTANCE,
            args.boundary_margin,
        )
    
    # 显示生成的位置信息
    print("\n生成的初始位置:")
    for i, (x, y) in enumerate(positions):
        print(f"robot{i+1}: ({x:.3f}, {y:.3f})")
    
    # 保存到文件
    save_positions_to_file(positions, args.output, seed=args.seed,
                           boundary_margin=args.boundary_margin)
    
    print("\n生成完成！")
    print(f"输出文件: {args.output}")
    print("这样可以实现手动控制初始位置的生成")

if __name__ == "__main__":
    main()
