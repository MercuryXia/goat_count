# goat_count — 桥头石山羊进出栏视觉计数（论文配套工程）

对应开题报告《低照度环境下桥头石山羊进出栏视觉计数方法研究与系统实现》第 3–6 章的
可运行实现：**数据与退化建模 → 低照度检测（FiLM 联合增强） → 自由流跟踪与干扰物过滤 →
双向过栏计数与守恒校验 → 边缘端系统**。

工程按"先能跑、再能训、最后能上机"三层组织：跟踪与计数是纯 numpy 实现，**不依赖
ultralytics，也不依赖任何权重**，因此拿到代码就能端到端验证计数逻辑；检测部分需要
ultralytics + 权重才进入真实精度阶段。

---

## 1. 环境

已在 `A:\MiniConda\envs\yolo` 下验证（Python 3.10.22 / torch 2.5.1+cu121 / ultralytics 8.4.211 /
opencv 5.0.0 / numpy 2.2.6 / RTX 3060）。

```bat
A:\MiniConda\envs\yolo\python.exe -m pip install -r requirements.txt
```

该环境没有 scipy / sklearn / pandas / pyyaml / pytest，工程据此做了降级（见 §7），
不需要额外安装。

---

## 2. 五分钟跑通

```bat
cd C:\Users\Mercury\Documents\deepseek-harness\default-workspace\goat_count

:: 1) 单元测试（25 个用例：几何、迟滞判决、往返、合并、跟踪、步态、指标、标签）
A:\MiniConda\envs\yolo\python.exe -m unittest discover -s tests -t .

:: 2) 端到端 demo：合成 12 只出栏 + 12 只归栏 + 狗 + 人，不需要任何权重
A:\MiniConda\envs\yolo\python.exe -m goatcount --config configs\scene_gate.yaml demo --out out\demo
::   -> out/demo/{demo.mp4, demo_annotated.mp4, events.jsonl, bev.png, summary.json, gt_events.json}
::   -> 打印 pred / gt、CA/MAE/MAPE、方向准确率

:: 3) 造一个合成 OBB 数据集（只为把训练链路跑通）
A:\MiniConda\envs\yolo\python.exe scripts\make_synthetic_dataset.py --out data\synthetic --frames 120

:: 4) 1 epoch 冒烟训练（验证数据/标签/增强/损失/FiLM 都能走通）
A:\MiniConda\envs\yolo\python.exe -m goatcount train --stage det --smoke ^
    --data data\synthetic\data.yaml --device 0 --project runs\smoke --name film

:: 5) 真实视频推理（换成自己的权重与视频）
A:\MiniConda\envs\yolo\python.exe -m goatcount --config configs\scene_gate.yaml infer ^
    --video data\clips\evening.mp4 --weights runs\obb\goat_gate\weights\best.pt --save-video
```

注意 `--config` 是**全局参数，必须写在子命令前面**。

参考结果（合成场景，仅证明链路正确，不代表真实精度）：出栏 11/12、归栏 10/12、
方向判断 100%、无误报；漏计主要来自两只羊几乎同时过线被合并规则吃掉。

---

## 3. 目录与章节对应

| 路径 | 对应章节 | 职责 |
| --- | --- | --- |
| `goatcount/config.py` | 3.1 | 场景配置（标定四点、计数线、迟滞、确认帧、过滤阈值）与 YAML 读写 |
| `goatcount/geometry.py` | 3.3.2 | 单应求解、BEV 投影、着地点、有向距离、OBB 工具 |
| `goatcount/data.py` | 3.1 | OBB 多边形标签读写、按天划分、低照度退化合成、数据集体检 |
| `goatcount/models.py` | 3.3.1 | 照度估计头 + FiLM 调制 + 联合增强挂载（`attach_joint_enhancement`） |
| `goatcount/train.py` | 3.3.1 / 3.4 | 三阶段训练：`enh`（低光配对自监督）/ `det`（FiLM 联合）/ `all` |
| `goatcount/tracking.py` | 4.x | 轻量跟踪器（两阶段关联、贪心匹配、丢跟踪再关联）+ 步态频谱特征 |
| `goatcount/counting.py` | 5.x | 迟滞两段式判决、往返判定、重复合并、守恒账本、事件 JSONL |
| `goatcount/metrics.py` | 3.4.2 | CA / MAE / MAPE / bias、方向 P-R-F1、MOTA / IDSW / IDF1（近似） |
| `goatcount/pipeline.py` | 4–5 | 检测 → 跟踪 → 过滤 → 计数的装配层 |
| `goatcount/infer.py` | 6.x | 三种检测器：ultralytics OBB、颜色桩（demo 用）、JSON 回放 |
| `goatcount/viz.py` | 6.x | 标注帧、BEV 画布（计数线、迟滞带、轨迹尾迹） |
| `goatcount/demo.py` | 3.4.5 | 合成场景生成器（含真值事件，用于无权重验收） |
| `goatcount/cli.py` | 6.x | 命令行：`demo / infer / check / train / selftest` |
| `configs/scene_gate.yaml` | 3.1 | 一个可直接用的示例场景 |
| `tests/test_core.py` | — | 25 个用例，回归基线 |

---

## 4. 数据格式

与 ultralytics OBB 一致：每张图一个 `.txt`，每行

```
class x1 y1 x2 y2 x3 y3 x4 y4      # 四点多边形，归一化到 [0,1]，顺序无所谓
```

目录：

```
data/goat_gate/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt
  data.yaml            # path / train / val / names
```

**划分必须按天/场次，而不是按帧**——同一天相邻帧几乎一样，按帧划分会虚高。用
`goatcount.data.prepare_dataset(src, out)` 可以按"日期目录名"自动切分并生成 `data.yaml`：

```bat
A:\MiniConda\envs\yolo\python.exe -m goatcount check --data data\goat_gate
```

会打印类别分布、空标签、每类实例数与划分统计。

---

## 5. 关键设计（与论文一致）

**单应 + 着地点**：相机俯角约 60°，检测框四角投影到 BEV 后取"最低边中点"作着地点，
避免用框中心导致近处目标系统性偏移；BEV 下体长近似恒定，可把像素位移换算成体长倍数。

**迟滞两段式判决**（`counting.GateCounter`）：进带记侧别与帧数 → 出带且侧别翻转才登记
"候选过栏" → 候选期内要同时满足 `|s| ≥ 0.5×体长` 与确认帧数（白天 5 / 夜间 8）才落事件；
候选期内回到原侧即撤销。这样既不会被线上抖动反复触发，也不会因为"出带那一帧余量不够"
把真实过栏永久丢掉。

**双向非对称**：出栏与归栏分别标定迟滞、余量、确认帧数，不共享阈值——晨出是密集遮挡
主导，暮归是低光漏检 + 重复触发主导，误差机理相反。

**守恒校验**（`ConservationLedger`）：`S_t = S_{t-1} - E_t + R_t`，日尺度上
`|R_t - E_t|` 超过 `max(5, 3%·E_t)` 即标记异常，回溯事件表输出候选片段。

**干扰物过滤**（`TrackFilter`）：检测类别 → 轨迹投票（≥70% 帧判羊）→ 业务规则（是否稳定
穿带）。步态频谱作为**软证据**：BEV 下体长/体宽/质心高时序去趋势后做 rFFT，狗的主频
（≥2.6 Hz）明显高于羊（≤1.6 Hz）；只在检测置信度低且投票含糊时才用它二次裁决，避免
引入新的误判源。

**FiLM 联合增强**：照度估计头（1/8 分辨率）+ 每级 `(γ, β)` 调制颈部特征，零初始化保证
起步是恒等映射；增强解码器只在训练期存在，推理时裁掉，参数与延迟增量 <5%。

---

## 6. 配置项（`configs/scene_gate.yaml`）

| 键 | 含义 |
| --- | --- |
| `homography.image_points / bev_points` | 地面 4 点标定（图像 ↔ BEV，顺序一一对应） |
| `counting.line` | 计数线两端点（BEV，米）；`in_side: positive` 表示线左侧算栏内 |
| `hysteresis_m` | 迟滞带半宽（默认 0.30 m） |
| `body_length_m` / `margin_body` | 体长先验与位移余量倍数（0.5 体长） |
| `confirm_frames.day / night` | 白天/夜间确认帧数（5 / 8） |
| `reentry_window_s` | 多长时间内的反向过线判为往返 |
| `merge_window_s` / `merge_radius_m` | 同一只羊被拆成两条轨迹时的合并窗口（别开太大，否则前后脚的两只羊会被并掉） |
| `tracking.gate_m / max_age_s / n_init` | 关联门限、丢失保留时长、确认命中数 |
| `filter.goat_vote` / `gait.*` | 轨迹投票阈值与步态频率判据 |

标定方法：在地上量一个矩形（或直接用门口四角），记录四个点的像素坐标与对应的
BEV 坐标（米），填进 yaml 即可；改完用 `demo` 或 `check` 验证投影是否落在合理范围。

---

## 7. 环境降级说明（重要）

| 缺失依赖 | 工程里的替代 | 影响 |
| --- | --- | --- |
| scipy | 贪心关联（`tracking.LiteTracker`），不用匈牙利算法 | 密集场景关联略差于线性分配 |
| scipy / motmetrics | IDF1 用"ID 对共现"近似，MOTA/IDSW 为精确实现 | IDF1 仅用于横向比较 |
| sklearn | 步态判别改成规则化阈值 + 可解释特征 | 无学习型分类器，需现场标定 |
| pyyaml | 用 `ultralytics.utils.YAML` | 纯 numpy 部分不读 yaml 也能跑 |
| pytest | `unittest`（`selftest` 子命令） | 无 |
| onnx / onnxruntime | 未实现导出脚本 | 第 6 章端侧部署需另装 onnx + TensorRT |

另外，受限账户/沙箱下 `multiprocessing.pool.ThreadPool` 会因命名管道报
`PermissionError: [WinError 5]`，`train.py` 里的 `patch_ultralytics_pool()` 已把它替换成
串行 map（只影响标签缓存速度）。

---

## 8. 已知限制 / 下一步

1. **没有真实数据**：现在所有精度数字都来自合成场景，只能证明链路正确；换真实数据后
   必须重跑 `check` → 训练 → `infer`，并按照度/密度分层报告。
2. **FiLM 的辅助损失未接入 ultralytics 的 criterion**：目前 `L_ill` 只作为模块方法存在，
   FiLM 参数通过检测损失端到端学习；若要严格复现论文的 `L = L_det + λ1L_rec + λ2L_ill + λ3L_con`，
   需要自定义 `v8OBBLoss` 并在 `model.loss()` 里叠加 `adapter.extra_loss()`。
3. **未实现导出**：TensorRT/INT8 校准、双模态（可见光 + 850 nm 红外）切换调度还缺代码。
4. **合并规则偏保守**：合成场景里两只羊前后脚过线会被并成一次，真机上应按"同框重叠率 +
   轨迹相似度"而不是纯距离判重。

---

## 9. 复现论文实验的最小路径

```bat
:: a. 整理数据（按日期目录放原图与标签）
python -m goatcount.data  :: 或直接调用 prepare_dataset(src, out)
python -m goatcount check --data data\goat_gate

:: b. 低光配对自监督（可选，产出照度估计器）
python -m goatcount train --stage enh --pairs data\pairs\pairs.jsonl

:: c. 联合训练（FiLM 挂在 trainer 里，权重会随 checkpoint 保存）
python -m goatcount train --stage det --data data\goat_gate\data.yaml ^
    --model yolo11-obb.yaml --epochs 100 --imgsz 640 --batch 8 --device 0 ^
    --enh-weights runs\enh\best.pt

:: d. 现场视频推理 + 指标
python -m goatcount --config configs\scene_gate.yaml infer --video xxx.mp4 ^
    --weights runs\obb\goat_gate\weights\best.pt --save-video
```

`events.jsonl` 每行一个过栏事件（时刻、方向、轨迹 ID、置信度、BEV 坐标、速度、是否往返、
照度），可直接喂给第 5 章的守恒校验与第 6 章的可视化。
