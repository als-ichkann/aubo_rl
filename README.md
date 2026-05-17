# aubo_rl

## Diffusion Policy 训练流程

本工程使用 MuJoCo 中的 AUBO 机械臂仿真采集专家演示数据，并使用官方 `real-stanford/diffusion_policy` 训练视觉模仿学习策略。

当前数据定义：

- 观测：末端相机 `ego_camera` 的 RGB 图像
- 动作：末端 6 维位姿增量 `[dx, dy, dz, droll, dpitch, dyaw]`
- 数据格式：zarr
- 默认数据集路径：`expert_demos/aubo_ego_rgb_delta_pose.zarr`

### 1. 激活环境并检查 DP 安装

```bash
conda activate aubo_rl
INSTALL_DP_EDITABLE=0 ./scripts/setup_diffusion_policy.sh
```

如果你希望把本地源码以 editable 方式重新安装：

```bash
./scripts/setup_diffusion_policy.sh
```

脚本默认使用本地源码目录：

```bash
third_party/diffusion_policy
```

### 2. 采集专家演示数据

运行采集脚本：

```bash
python collect_dp_demo.py --preview
```

常用控制：

- `g`：开始/结束一条 episode
- `x`：丢弃当前 episode
- `space`：将目标位姿重置为当前末端位姿
- `q`：退出

遥控器按钮默认：

- 按钮 `2`：开始/结束一条 episode
- 按钮 `4`：丢弃当前 episode
- 按钮 `3`：重置目标位姿
- 按钮 `1`：退出

采集参数示例：

```bash
python collect_dp_demo.py \
  --output expert_demos/aubo_ego_rgb_delta_pose.zarr \
  --image-size 96 \
  --sample-rate 20 \
  --preview
```

### 3. 检查数据集

```bash
python scripts/inspect_dp_dataset.py expert_demos/aubo_ego_rgb_delta_pose.zarr
```

该脚本会打印：

- episode 数量
- 总步数
- 图像 shape
- action shape
- action 的均值、方差、最小值、最大值

并默认保存若干预览帧到：

```bash
expert_demos/preview_frames
```

### 4. 运行 smoke test

如果只检查本工程数据链路：

```bash
python scripts/smoke_test_dp_pipeline.py
```

如果要同时检查 `diffusion_policy` 是否可导入：

```bash
python scripts/smoke_test_dp_pipeline.py --require-diffusion-policy
```

### 5. 启动训练

训练配置文件：

```bash
configs/diffusion_policy/aubo_ego_rgb_delta_pose.yaml
```

启动训练：

```bash
./scripts/train_diffusion_policy_aubo.sh
```

如果需要覆盖 diffusion_policy 的 hydra 参数，可以追加在命令末尾，例如：

```bash
./scripts/train_diffusion_policy_aubo.sh training.device=cuda:0 training.num_epochs=200
```

训练完成后，权重默认保存在 **`train.py` 的工作目录**（脚本会 `cd` 到 `third_party/diffusion_policy`），因此路径为：

```bash
third_party/diffusion_policy/data/outputs/<YYYY.MM.DD>/<HH.MM.SS>_train_diffusion_unet_image_aubo_ego_rgb_delta_pose_aubo_ego_rgb_delta_pose/checkpoints/
```

（若在别处改过 Hydra `hydra.run.dir`，则以实际输出目录为准。）

### 6. 仿真闭环推理（MuJoCo）

在仓库根目录加载 `latest.ckpt`（或某个 `epoch=*.ckpt`），在仿真里按与采集一致的节拍执行策略输出的末端位姿增量。

```bash
conda activate aubo_rl
python run_dp_policy_sim.py \
  --checkpoint third_party/diffusion_policy/data/outputs/2026.05.17/15.18.08_train_diffusion_unet_image_aubo_ego_rgb_delta_pose_aubo_ego_rgb_delta_pose/checkpoints/latest.ckpt \
  --inference-steps 20
```

```bash
conda activate aubo_rl
python run_dp_policy_sim.py \
  --checkpoint third_party/diffusion_policy/data/outputs/2026.05.17/15.14.17_train_diffusion_unet_image_aubo_ego_rgb_delta_pose_aubo_ego_rgb_delta_pose/checkpoints/latest.ckpt \
  --inference-steps 20
```

也可使用相对仓库根目录的短路径（脚本会自动在 `third_party/diffusion_policy/` 下查找）：

```bash
python run_dp_policy_sim.py \
  --checkpoint data/outputs/<日期>/<运行目录>/checkpoints/latest.ckpt \
  --inference-steps 20
```

操作说明：

- 默认策略关闭；按 **`c`**（MuJoCo 窗口或 OpenCV 预览窗口处于焦点时）打开/关闭闭环策略。
- **`space`**：将目标位姿重置为当前 FK（与采集脚本一致）。
- **`q`**：退出。
- **`--image-size`** 必须与训练 `shape_meta` 一致（默认 96）。
- 若推理跟不上 `sample-rate`，可降低 **`--inference-steps`**（例如 16）或改用 GPU **`--device cuda:0`**。

可选：在未开策略时用遥控器微调起始姿态：`--joystick`（参数与 `collect_dp_demo.py` 中轴映射一致）。

### 7. 训练前建议

- 至少先采集多条短 episode，确认数据集能被 `inspect_dp_dataset.py` 正常读取。
- 确认 `action` 不是全 0，否则说明采集时没有实际移动目标位姿。
- 如果图像训练显存不足，可以先降低配置中的 `batch_size`。
- 如果没有 GPU，可把配置里的 `training.device` 改为 `cpu`，但训练会很慢。
