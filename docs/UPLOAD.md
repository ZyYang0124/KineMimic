# 拍摄 → 上传 → 可视化：你的蛛蚁视频工作流

这是 MOTIONSCAPE 的主产品路径：拍摄蜘蛛与蚂蚁同框活动 → 一条命令 →
Movement Galaxy 可视化 + Siler↔Ant 比较。

## 1. 拍摄建议（让管线可靠）

- **固定机位三脚架**，俯拍或近似俯拍；拍摄期间不要移动、不要变焦。
- 背景（树干/叶面/地面）保持稳定：本版检测器基于背景差分，
  剧烈摇晃的树叶会引入噪声。
- **画面里放一把尺子**（或已知长度的物体）至少出现一帧，用于
  `px_per_cm` 标定——速度、加速度才有真实单位。
- 分辨率 ≥1080p、帧率 ≥30fps（120fps 更好，系统会自动抽帧到 30fps 分析，
  片段切回仍用原始帧）。
- 单段 ≥1 分钟为宜；动物越小越近越好（目标 >15 像素）。
- 光照均匀；避免频闪光源。

## 2. 上传处理

```bash
# 单个视频
python -m motionscape ingest D:/field/video01.mp4 --id siteA_2026-09-17_01

# 多个视频一次处理（同一研究地点/条件）
python -m motionscape ingest vid1.mp4 vid2.mp4 vid3.mp4 \
    --id siteA_01 siteA_02 siteA_03 \
    --store motionscape_runs/mysite
```

输出：
- `runs/<run_id>/episodes.jsonl` —— 提取的全部运动片段（含轨迹、特征、
  启发式物种标注、完整溯源链）
- `runs/<run_id>/summary.json` —— 拟态指纹（逐维度 Siler↔Ant 重叠度）
- `runs/<run_id>/atlas/` —— Movement Galaxy（`python -m http.server`
  在该目录内启动后浏览器打开）

## 3. 调参（按需）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--min-episode-s` | 3.0 | 短于此的观察丢弃；调小可保留更多碎片 |
| `--target-fps` | 30 | 分析帧率；高帧率视频自动跳帧提速 |
| `--n-motifs` | 8 | Motion Dictionary 的模式数 |

检测器内部参数（阈值/目标大小范围）在 `motionscape/detect.py` 顶部，
若漏检/误检可按场地调整；每段 episode 的处理历史会记录所用模型版本。

## 4. 标注修正（可选但推荐）

启发式标注（体型伸长度 × 间歇性）会有误差。人工修正不触碰观察数据，
只追加标注记录：

```python
from motionscape.store import EpisodeStore
EpisodeStore.add_annotation("motionscape_runs/mysite",
    episode_id="ep_xxx", bio_label="siler", confidence=0.95)
```

之后重新 `ingest` 或 `atlas` 重建时标注自动按时间顺序生效。

## 5. 比较

- **Galaxy 内**：Reveal species → 点 Siler 粒子 → Compare with nearest ant
  （分屏同步对照真实片段）。
- **Motion Dictionary**：看哪些 motif 蚂蚁多、Siler 混入、非拟态蛛稀少
  ——候选拟态行为。
- **summary.json**：模块化拟态指纹（速度/间歇性/节律/转向/轨迹形状）。

## 已实测

真实场地视频（GoPro 1280×720@120fps，蛛蚁同场）：57 秒视频自动提取
83 段 episode，全链路（检测→跟踪→片段→特征→标注→银河→指纹）约 10
分钟，片段直接切自原始视频。
