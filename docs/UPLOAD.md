# 拍摄 → 上传 → 标注 → 图谱:你的蛛蚁视频工作流

这是 KineMimic 的主产品路径:拍摄蜘蛛与蚂蚁同框活动 → 一条命令 →
人工 QC + 标注 → 可探索的 Movement Atlas。

## 1. 拍摄建议(让管线可靠)

- **固定机位三脚架**,俯拍或近似俯拍;拍摄期间不要移动、不要变焦。
- 背景(树干/叶面/地面)保持稳定:本版检测器基于背景差分,
  剧烈摇晃的树叶会引入噪声。
- **画面里放一把尺子**(或已知长度的物体)至少出现一帧,用于
  `px_per_cm` 标定——速度、加速度才有真实单位。
- 分辨率 ≥1080p、帧率 ≥30fps(120fps 更好,系统会自动抽帧到 30fps 分析,
  片段切回仍用原始帧)。
- 单段 ≥1 分钟为宜;动物越小越近越好(目标 >15 像素)。
- 光照均匀;避免频闪光源。

## 2. 上传处理

```bash
# 单个视频(--site 记入采样层级: Site→Session→Video→Episode)
python -m kinemimic ingest D:/field/video01.mp4 --id siteA_2026-09-17_01 --site siteA

# 多个视频一次处理(同一研究地点/条件)
python -m kinemimic ingest vid1.mp4 vid2.mp4 vid3.mp4 \
    --id siteA_01 siteA_02 siteA_03 --site siteA --session 2026-09-17 \
    --store kinemimic_runs/mysite
```

输出:
- `runs/<run_id>/episodes.jsonl` —— 提取的全部运动片段(轨迹、特征、
  machine 预分类、完整溯源链;观测与标注严格分离)
- `runs/<run_id>/summary.json` —— 标注进度、采样层级、逐维拟态指纹
- `runs/<run_id>/atlas/` —— The Murmur 图谱(见下一步)

## 3. 人工 QC + 标注(推荐,构成可信科学的分界线)

机器预分类与运动分析共享特征,不能作为真值(原理见
`SCIENTIFIC_ASSUMPTIONS.md` §3):

```bash
python -m kinemimic annotate kinemimic_runs/mysite/runs/<run_id>/episodes.jsonl
# → http://127.0.0.1:8692   键盘:S/A/O/R/U 标注,X/F/H/E/T/B QC,每键即存
```

界面左看片段(裁剪放大)、右看元数据与机器预测、底部快捷键。
逐条即时持久化到 append-only 标注日志,重开自动续审。

## 4. 打开图谱

```bash
python -m kinemimic serve kinemimic_runs/mysite/runs/<run_id>/atlas
# → http://127.0.0.1:8694  片段按需从原视频生成
```

先只看到运动结构(blind);点 **Reveal species** 才显示物种身份。
带视频源时点击粒子播放真实片段;最近邻、对比、Motion Dictionary、
指纹见 `docs/ATLAS.md`。

## 5. 调参(按需)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--min-episode-s` | 3.0 | 短于此的观察丢弃;调小可保留更多碎片 |
| `--target-fps` | 30 | 分析帧率;高帧率视频自动跳帧提速 |
| `--n-motifs` | 8 | Motion Dictionary 的模式数 |
| `--site` / `--session` | — | 采样层级标签(统计去伪重复用) |

检测器内部参数(阈值/目标大小范围)在 `kinemimic/detect.py` 顶部,
若漏检/误检可按场地调整;每段 episode 的处理历史会记录所用模型版本。

## 6. 比较(标注完成后才有生物学意义)

- **图谱内**:Reveal species → 选一段 Siler → compare(与最近邻分屏同步
  对照真实轨迹与速度序列);相似度来自 17 维特征空间,非屏幕距离。
- **Motion Dictionary**:机器发现 motif,人工观看后命名;Reveal 后看各
  物种占比——候选拟态行为。
- **指纹面板**:逐维 Siler↔Ant 重叠度(速度/间歇性/节律/转向/轨迹形状),
  有意不合并为单一分数。

## 已实测

真实场地视频(GoPro 1280×720@120fps,蛛蚁同场):57 秒视频自动提取
83 段 episode,全链路(检测→跟踪→片段→特征→标注→图谱→指纹)约 10
分钟,片段直接切自原始视频。
