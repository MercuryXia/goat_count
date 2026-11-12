# -*- coding: utf-8 -*-
"""goatcount —— 桥头石山羊进出栏视觉计数（开题报告第 3–6 章的工程实现）。

模块分层（与论文章节的对应关系）：

    第 3 章 低照度检测   data.py（物理退化合成）  models.py（YOLO11-OBB / FiLM 调制）  train.py
    第 4 章 跟踪与过滤   tracking.py（BEV 跟踪、步态频谱、羊/狗/人轨迹判别）
    第 5 章 双向计数     counting.py（迟滞状态机、事件判决、守恒校验）  metrics.py
    第 6 章 系统实现     pipeline.py  infer.py  viz.py  cli.py

设计约束：
* 计数与跟踪只用 numpy，不依赖 ultralytics，因此没有模型权重也能端到端跑通（见 cli demo）。
* 训练与推理才需要 ultralytics / torch，且只在这两个模块里 import。
"""

__version__ = "0.1.0"
