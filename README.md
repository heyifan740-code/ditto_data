# DITTO 数据特征分析

任务 **uncap（拧开瓶盖）**，对比两种采集模式的数据特性：
- **ITW**（In-The-Wild）：人手持 leader 外骨骼直接操作，TrakSTAR 磁追踪记录手位姿
- **Teleop**：人戴 leader 远程操控 follower 机器人手，follower 真实接触物体

核心问题：两种数据的特性差异如何影响 policy 学习（参考 *Diffusion Beats Autoregressive* 的数据分析方法论）。

---

## 运行环境

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate ditto_analysis
cd /home/fa_team/roamlab/finger_aloha/software/ditto_data/analysis/scripts
python 01_data_inventory.py            # 例：跑数据摸底
python 03_visualize_demo.py teleop 0   # 例：渲染 teleop demo_0 视频
```

依赖：`zarr numpy pandas matplotlib seaborn scipy scikit-learn opencv-python python-docx`

---

## 目录结构

```
analysis/
├── README.md                    # 本文件
├── scripts/                     # 全部分析脚本（01–11）
└── outputs/                     # 全部产出，按主题分类
    ├── reports/                 # docx 综合报告
    ├── videos/                  # demo 可视化 mp4
    ├── vee_verification/        # v_ee 验证图
    ├── phase_segmentation/      # phase 切分时序图（v_ee 初版）
    ├── force_analysis/          # force×运动学机制研究图
    └── closure/                 # Step 2–4 闭环三图
```

脚本输出路径已改为相对 `scripts/`，重跑会自动写入对应 `outputs/` 子目录。

---

## 脚本清单（按分析阶段）

### A. 数据摸底 — Step 0 Inventory

| 脚本 | 作用 | 产出 |
|---|---|---|
| `01_data_inventory.py` | 列出两侧 channel/shape/dtype，统计 demo 数/时长/频率，检查力信号 | 终端打印 |
| `02_full_analysis.py` | 全量特征提取，生成完整 docx 报告 | `outputs/reports/DITTO_Data_Analysis_Report.docx` |

**关键结论**：ITW 50 + Teleop 50 demos，均 20Hz；ITW 的 efforts 全零、follower≡leader 镜像；arm pose 是 6 维 `[x,y,z,rx,ry,rz]` 非 quat；相机 ITW=leader_rgb+side_rgb，Teleop=palm_rgb+side_rgb。

### B. 可视化工具

| 脚本 | 作用 | 产出 |
|---|---|---|
| `03_visualize_demo.py <mode> <id>` | 单 demo 渲染为 mp4：相机画面 + 力 + 动作同步滚动 | `outputs/videos/viz_<mode>_demo<id>.mp4` |

### C. v_ee 验证 — Step 0 收尾

| 脚本 | 作用 | 产出 |
|---|---|---|
| `04_verify_vee.py` | 验证 ITW 的 arm_obs 可当米制位姿算末端速度 v_ee | `outputs/vee_verification/verify_vee_histogram.png` |

**关键结论**：源码确认 arm_obs 单位是米（相对 episode 起点位移，base 固定）；ITW 速度峰值 0.18m/s、Teleop max 0.275m/s 卡在 xArm 限速 0.35m/s 下（硬交叉验证）。两侧 v_ee 分布可比。

### D. Phase 切分初探 — Step 1（已被 q-classifier 取代）

| 脚本 | 作用 | 产出 |
|---|---|---|
| `05_phase_segmentation.py <mode> <id>` | v_ee 状态机切 free/contact，时序图 + 抽帧核对 | `outputs/phase_segmentation/phase_*.png` |

**结论**：单用 v_ee 切 contact 不干净（contact 段末端有微调，碎切）。→ 转向用 force 真值客观评判（见 E）。

### E. Force × 运动学 机制研究 — Step 1 深化（核心）

| 脚本 | 作用 | 产出 |
|---|---|---|
| `06_segmenter_validation.py` | 用 force 双峰谷底 θ_e 定 contact 真值，评判 a_hand/v_ee segmenter F1 | `outputs/force_analysis/efforts_histogram.png`, `ahand_f1_heatmap.png` |
| `07_force_predictability.py` | 滑窗 [q,Δq,v_ee] 监督学习预测 force-contact，AUC（**episode-split**）+ 泄漏对照 | `outputs/force_analysis/coverage_per_episode.png`, `force_microstructure.png` |
| `08_feature_ablation.py` | 消融：force 信息藏在哪个特征（q vs 速度类） | 终端打印 |
| `09_mechanism_test.py` | 机制判定：follower_q vs leader_q + tracking_err | 终端打印 |

**关键结论（认识经三轮修正）**：
1. 速度范数标量 a_hand/v_ee 预测 force AUC≈0.5（范数丢分关节信息，**勿据此说"运动学⊥force"**）。
2. 滑窗 [q,Δq,v_ee] 监督学习 AUC=0.98；消融定位 **单帧 q 即 0.96**，force 由手构型决定。
3. follower_q(0.964)≈leader_q(0.972) → **机制B 构型代理主导**；但 tracking_err 与 force ρ=0.73、AUC=0.91 → **机制A 接触偏移真实存在**，且 ITW 恒为 0。

### F. 闭环 — Step 2–4

| 脚本 | 作用 | 产出 |
|---|---|---|
| `10_closure.py` | q→contact LR 切两侧 phase + ITW spot check + per-phase Δq gap + tracking_err 缺失 | `outputs/closure/itw_contact_spotcheck.png`, `gap_by_phase.png`, `tracking_err_missing.png` |
| `11_gap_robustness.py` | gap by phase 的下采样 + bootstrap 稳健性 | 终端打印 |

**关键结论（三重交集叙事）**：
- phase 切分：q→contact LR，val AUC=0.964；teleop contact 31%（≈force 真值），ITW contact 仅 7.3%。
- **gap by phase**：contact W1=0.611 [CI 0.574–0.656] vs free W1=0.250 [CI 0.242–0.258]，**contact 段动作 gap 是 free 的 2.4×**（bootstrap 确认非样本量 artifact）。
- **缺失通道**：tracking_err teleop contact 0.066/free 0.031（编码接触），ITW 恒 0（结构性）。
- → **contact phase = 动作 gap 最大 ∩ teleop 有接触信号 ∩ ITW 结构性缺失** → 对应 DITTO Fig6 ITW-only 在 cap lift 失败。

---

## 输出图速查

| 图 | 位置 | 说明 |
|---|---|---|
| verify_vee_histogram.png | vee_verification/ | 两侧末端速度分布可比（ITW 中高速尾巴更厚=人手更连续） |
| phase_teleop_demo0.png | phase_segmentation/ | v_ee/a_hand 时序 + phase 着色（初版切分） |
| efforts_histogram.png | force_analysis/ | ‖efforts‖ 双峰 + θ_e≈253 谷底（contact 真值阈值） |
| ahand_f1_heatmap.png | force_analysis/ | a_hand segmenter 参数 grid 的 F1（均低→运动学切不准） |
| force_microstructure.png | force_analysis/ | manipulation 段内 force 的间歇 micro-episode |
| coverage_per_episode.png | force_analysis/ | a_hand 段对 force-contact 的覆盖率分布 |
| itw_contact_spotcheck.png | closure/ | ITW 判 contact 的 15 帧 side_rgb 人工核对 |
| **gap_by_phase.png** | closure/ | **核心：contact vs free 的 ITW-teleop 动作 gap** |
| **tracking_err_missing.png** | closure/ | **核心：tracking_err teleop 有/ITW 恒 0** |
| viz_*_demo0.mp4 | videos/ | demo 同步可视化视频 |

---

## 关键名词

- **v_ee**：末端线速度 `‖Δ(arm_obs[:3])‖/dt`
- **a_hand**：手指关节角速度范数 `‖Δq‖/dt`
- **tracking_err**：`‖follower_q − leader_q‖`，teleop 编码接触力、ITW 恒 0（镜像）
- **θ_e**：force ‖efforts‖ 双峰谷底，定 contact 真值的阈值（≈253）
- **机制 A / B**：A=接触偏移（position 控制跟踪误差编码 force）；B=构型代理（拧盖构型本身预测 force）
