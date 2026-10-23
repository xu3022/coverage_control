# Coverage Control

基于**已知敏感度场**的多机器人 Voronoi 覆盖控制仿真。

项目直接读取配置中给定的真实敏感度函数，计算各机器人 Voronoi 区域的加权质心，
再通过质心跟踪控制和六自由度四旋翼动力学更新机器人状态。

## 主要功能

- 有界二维工作区域上的多机器人 Voronoi 分区；
- 基于已知解析敏感度场或已知离散网格的加权质心计算；
- 一阶 Lloyd 速度控制，可选二阶速度动态；
- 四旋翼速度/高度外环、SO(3) 姿态内环、电机分配和六自由度动力学；
- 支持静态热点与按控制周期移动的热点；
- 输出覆盖代价、机器人到质心的距离和可选 SVG 动画帧；

## 控制原理

已知敏感度场写作：

```text
phi(q) = phi_base + sum_k w_k * phi_k(q)
```

对于机器人 `i` 的 Voronoi 区域 `V_i`，加权质心为：

```text
c_i = integral(V_i, q * phi(q) dq) / integral(V_i, phi(q) dq)
```

水平覆盖控制采用：

```text
v_i = k_m * (c_i - p_i)
```

速度经过限幅后送入四旋翼闭环模型。程序记录的覆盖目标为：

```text
J = -sum_i integral(V_i, ||p_i - q||^2 * phi(q) dq)
```

因此该项目输出的 `coverage_cost` 为负数；控制效果改善时，数值通常向 `0` 增大。

## 运行环境

推荐环境：

- Ubuntu 22.04/24.04 或其他常见 Linux 发行版；
- Python 3.10 及以上，当前验证版本为 Python 3.12；
- 无界面服务器运行时建议使用 Matplotlib `Agg` 后端。

核心依赖：NumPy、SciPy、Matplotlib、Shapely 和 GeoVoronoi。测试依赖 pytest。

## 安装

### 使用独立虚拟环境

```bash
git clone <your-repository-url>
cd coverage_control

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

也可以使用依赖文件：

```bash
python -m pip install -r requirements.txt
```

### 使用已有的 `dl-env`

如果已经有可用的 `dl-env` 虚拟环境，可以直接复用：

```bash
source /path/to/dl-env/bin/activate
cd /path/to/coverage_control
python -m pip install -e ".[dev]"
```

## 快速运行

安装后可直接运行：

```bash
coverage-control
```

源码方式运行：

```bash
python coverage_control/coverage_control.py
```

无界面运行完整 200 个控制周期：

```bash
MPLBACKEND=Agg \
COVERAGE_ENABLE_PLOTTING=0 \
COVERAGE_MAX_CONTROL_TIMESTEPS=200 \
coverage-control
```

快速检查 5 个周期：

```bash
MPLBACKEND=Agg \
COVERAGE_ENABLE_PLOTTING=0 \
COVERAGE_MAX_CONTROL_TIMESTEPS=5 \
coverage-control
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `COVERAGE_MAX_CONTROL_TIMESTEPS` | `200` | 最大控制周期数，必须为正整数 |
| `COVERAGE_ENABLE_PLOTTING` | `1` | `0` 表示关闭逐周期 SVG 绘图 |
| `COVERAGE_SCRIPTS_DIR` | `<项目目录>/scripts` | 覆盖代价和距离 CSV 的输出根目录 |
| `COVERAGE_INITIAL_POSITIONS_FILE` | 包内默认文件 | 自定义机器人初始位置文件 |
| `MPLBACKEND` | 自动选择 | 无显示器时建议设为 `Agg` |

## 主要配置

全部运行参数集中在
[`coverage_control/config.py`](coverage_control/config.py) 的 `CoverageConfig` 中。

| 配置项 | 作用 |
| --- | --- |
| `ROBOT_CONFIG` | 机器人数量和名称 |
| `CONTROL_CONFIG` | 控制周期、覆盖增益、速度上限及二阶动态开关 |
| `QUADROTOR_CONFIG` | 质量、惯量、电机、姿态环和动力学参数 |
| `WORK_AREA_CONFIG` | 工作区域边界及积分分辨率 |
| `DENSITY_FUNCTION_CONFIG` | 已知敏感度分量、权重和热点运动参数 |
| `GRID_SENSITIVITY_CONFIG` | 已知离散敏感度网格开关与分辨率 |
| `VORONOI_CONFIG` | Voronoi 分区及数值积分配置 |
| `EXPERIMENT_CONFIG` | 控制周期数、绘图和输出设置 |
| `RANDOM_CONFIG` | 随机种子设置 |

默认场由三个已知分量组成：截断高斯、环形分布和普通高斯。第三个热点在第
`50` 至 `55` 个控制周期从 `(8, 2)` 线性移动到 `(2, 8)`。

### 初始位置文件

默认文件为
[`coverage_control/position/initial_positions.txt`](coverage_control/position/initial_positions.txt)。
每行格式如下：

```text
robot1 1.25 3.50
robot2 4.10 7.20
```

自定义文件必须包含配置中的全部机器人名称。也可以重新生成：

```bash
python coverage_control/position/generate_initial_positions.py
```

## 输出文件

程序会在运行时自动创建下列目录，因此它们不会提交到 Git：

```text
scripts/
├── cost function/known-sensitivity.csv
└── distance/known-sensitivity-distance.csv

pic/
└── voronoi_coverage_<运行时间>_t<周期>.svg
```

关闭绘图不会影响 CSV 记录。每次启动控制器都会重建本次运行对应的 CSV 文件。

## 测试

```bash
pytest -q
```

如果本机 Python 环境安装了会干扰 pytest 的第三方插件，可以关闭插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

## 项目结构

```text
coverage_control/
├── coverage_control/
│   ├── assets/                  # UAV SVG 标记
│   ├── position/                # 默认位置及位置生成器
│   ├── config.py                # 全部运行配置
│   ├── control_module.py        # 覆盖速度控制
│   ├── coverage_control.py      # 主控制循环和程序入口
│   ├── plotting_module.py       # Voronoi 与敏感度场绘图
│   ├── robot_model.py           # 四旋翼控制与六自由度动力学
│   └── voronoi_partition.py     # 分区和已知场加权质心
├── tests/                       # 当前功能回归测试
├── pyproject.toml               # 安装和命令行入口配置
├── requirements.txt             # 运行及测试依赖
└── README.md
```

## 使用边界

- 本项目假设敏感度场完全已知，不进行在线场估计；
- 当前机器人交互发生在同一仿真进程内，不建模通信丢包或时延；
- 当前版本是数值仿真，不包含 ROS、Gazebo 和真实飞控接口；
- 默认参数面向 `10 m x 10 m` 区域和 20 个机器人，修改规模后应重新检查积分分辨率与运行时间。

## 许可证

当前仓库尚未附带开源许可证。公开发布前，请根据用途选择并添加许可证文件，
例如 MIT、BSD-3-Clause 或 Apache-2.0；未添加许可证时，其他人默认无权复制、修改或分发代码。
