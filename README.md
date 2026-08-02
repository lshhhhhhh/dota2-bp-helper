# Dota 2 BP Helper

一个面向 Dota 2 天梯三轮同时盲选规则的离线 BP 助手：识别当前屏幕上的双方阵容，
并使用分段模型同时给天辉和夜魇推荐下一手英雄。正常使用完全离线，不需要 API Key。

## 下载与运行

普通用户请在 GitHub Releases 下载 `Dota2BPHelper-0.1.0-win64.zip`，解压到任意目录，
双击 `Dota2BPHelper.exe`。不要直接在压缩包内运行。

源码运行需要 Python 3.11+：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\run_desktop.cmd
```

应用不会读取 Dota 2 进程内存、注入游戏或自动操作鼠标键盘；屏幕识别只处理桌面截图。

它把系统拆成四层：

1. `DraftState`：只描述当前公开阵容与轮次。
2. `PolicyModel`：估计高分玩家在当前状态下会选择哪些英雄。
3. `ValueModel`：估计完整 5v5 阵容的胜率。
4. `RolloutRecommender`：对同轮隐藏选人和后续轮次做 Monte Carlo rollout。

当前版本已经包含离线推荐模型、截图头像识别和 Windows 桌面 UI。模型、视觉与界面彼此独立，后续可以单独替换或升级。

## 数据假设

OpenDota 的逐场详情目前提供 `picks_bans`。剔除同轮撞英雄后没有进入最终阵容的尝试记录，可以还原天梯三轮：

- 第一轮：双方各选两个；
- 第二轮：在公开 2v2 状态下双方各选两个；
- 第三轮：在公开 4v4 状态下双方各选一个。

采集器同时保存压缩原始响应和规范化 SQLite 数据，避免为同一批比赛重复支付 API 费用。

## 第一步：先审计数据

不要先假定数据量和字段可用性。运行：

```powershell
python -m d2draft.audit --days 3 --detail-sample 30 --min-avg-rank-tier 75
```

脚本会保存：

- 每个 `avg_rank_tier` 的逐日有效局数；
- 最终 5v5 阵容完整率；
- `picks_bans` 可用率；
- 剔除撞车英雄后能否还原 `2+2 / 2+2 / 1+1` 三轮；
- 每局已知个人段位数、Immortal 人数及严格全员高分的保留率；
- 失败和异常记录，而不是静默丢弃。

脚本默认不使用 `.env` 中的密钥。只有显式加入 `--use-env-key` 时才会使用 `OPENDOTA_API_KEY`，且永远不会输出密钥内容。请求间隔默认仍为 1.05 秒，可通过 `OPENDOTA_MIN_INTERVAL` 调整。

## 免费额度内的 MVP 采集

采集器兼容 `.env` 中的 `OPENDOTA_API_KEY` 或 `open_dota_api`，默认最多请求 2,000 次：

```powershell
python -m d2draft.collect --env-file .env --data-dir data/collection
```

它具有硬请求上限、限速、断点续传、去重和失败记录。提高 `--max-attempts` 可能产生费用，必须显式指定。

训练并比较“不看当前阵容、只按常见选择排序”的简单方法与小型神经网络：

```powershell
python -m d2draft.experiment `
  --database data/collection/draft_matches.sqlite3 `
  --output-dir artifacts/mvp
```

## 后续快速开始

使用任意 Python 3.11+ 即可运行训练和研究命令：

当前样本和实验结果见 `data/collection/manifest.json` 与
`artifacts/mvp/report.json`。数据库现有 66,515 场可还原的 7.41 有序 BP；桌面端
三个现役模型仍使用通过晋升验证的 56,625 场版本。66,515 场候选在最新时序
测试上没有稳定超过现役模型，因此保留在 `artifacts/candidates/base_66515/`
而未发布。第一轮使用热度，第二、第三轮使用神经网络
Policy。旧的 23,892 场模型保存在
`artifacts/archive/base_23892_before_38717/`，晋升比较见
`artifacts/candidates/base_38717/BASE_MODEL_REPORT.md`；38,717 场模型保存在
`artifacts/archive/base_38717_before_56625/`，最新晋升比较见
`artifacts/candidates/base_56625/BASE_MODEL_REPORT.md`。

数据库同时保存上游原始版本编号与统一版本名：`data_source`、
`source_patch_id`、`canonical_patch`。同步版本常量并重新回填：

```powershell
python -m d2draft.patches --sync
```

训练和回测应明确指定补丁，避免未来版本混入当前模型：

```powershell
python -m d2draft.experiment --patch 7.41 --output-dir artifacts/checkpoints/patch_7.41
python -m d2draft.backtest --patch 7.41 --model artifacts/checkpoints/patch_7.41/hybrid_model.npz
```

也可以使用 `avg_rank_tier` 训练分段模型。传奇及以上使用下限 50，统帅及以下使用
排他的上限 50；段位未知的对局会自动排除：

```powershell
python -m d2draft.experiment --patch 7.41 --min-rank-tier 50 --output-dir artifacts/models/legend_plus
python -m d2draft.backtest --patch 7.41 --min-rank-tier 50 --model artifacts/models/legend_plus/hybrid_model.npz --output artifacts/models/legend_plus/backtest.json

python -m d2draft.experiment --patch 7.41 --max-rank-tier-exclusive 50 --output-dir artifacts/models/archon_below
python -m d2draft.backtest --patch 7.41 --max-rank-tier-exclusive 50 --model artifacts/models/archon_below/hybrid_model.npz --output artifacts/models/archon_below/backtest.json
```

用固定的最新 20% 测试集画学习曲线，检查增加训练对局后模型是否趋于平台期：

```powershell
python -m d2draft.learning_curve `
  --patch 7.41 `
  --seeds 3 `
  --output artifacts/learning_curve/learning_curve.json
```

脚本分别评估传奇及以上、统帅及以下模型。默认把旧 80% 中的训练样本按
1,000、2,000、4,000……逐步扩大，每个点训练三个随机种子；只有连续两个
至少相隔 1,000 场的数据增量都小于每 2,000 场 0.5 个百分点，且随机种子
区间包含零，才标记为实际平台期。第一次运行会把测试对局 ID 和训练顺序保存
到 `artifacts/learning_curve/fixed_split.json`；之后采集的新对局只追加到训练池，
不会偷偷改变测试题或早期检查点。

离线回测各阶段的 Policy/Value 混合权重：

```powershell
python -m d2draft.backtest `
  --database data/collection/draft_matches.sqlite3 `
  --model artifacts/mvp/hybrid_model.npz
```

运行一个 4v4 的第三轮推荐（英雄可使用英文名或 OpenDota ID）：

```powershell
python -m d2draft.recommend_cli `
  --model artifacts/mvp/hybrid_model.npz `
  --phase 3 `
  --allies "Axe,Crystal Maiden,Juggernaut,Pudge" `
  --enemies "Anti-Mage,Lion,Invoker,Sniper" `
  --top-k 10
```

默认使用验证集为每个阶段独立选择的轻量 Value 权重。加入
`--value-blend 0` 可以只看 Policy；`--json` 可输出供桌面 UI 调用的结构化结果。
`data/heroes.json` 保存英雄名称映射。

## 桌面端与截图识别

在 Windows 中双击：

```text
run_desktop.cmd
```

默认启动器会在后台运行，不保留 CMD/PowerShell 窗口。如果 App 无法启动，需要查看
报错时，双击 `run_desktop_debug.cmd` 使用带控制台的调试启动器。

窗口支持：

- “模型”界面展示当前模型 ID、Dota 版本、训练规模、原理和最终测试指标；
- “基准对比”展示历史对局中选择模型前列英雄时的胜率关联、样本量与不确定区间；
- 默认使用传奇及以上模型，并可在传奇+、统帅-和全段位三个内置模型间切换；
- 一键识别当前屏幕，支持自动检查所有显示器或固定选择某一块屏幕；
- 自动定位显示器内带黑边的 16:9 窗口化直播/录像游戏画面；
- 读取已有 BP 截图；
- 自动按双方已公开英雄数推断第 1/2/3 轮；
- 天辉、夜魇各五个固定头像槽，并在头像旁显示识别置信度；
- 点击头像移除，分别向天辉或夜魇手动添加英雄；
- 手动输入支持官方中英文名、拼音、内部名、常用缩写和国服绰号；
- 同时给天辉和夜魇推荐下一手，夜魇候选也可视为天辉需要防范的英雄；
- 每两秒轮询和窗口置顶；
- 推荐结果显示 Policy 选择概率与 Value 胜率倾向。

### 模型包与 App 解耦

桌面端通过各模型目录的 `model_manifest.json` 发现并加载模型，而不是直接假定某个
模型文件。manifest 声明模型包格式、模型 ID、适用 Dota 版本与分段、英雄数、输入
输出契约及模型文件 SHA-256。启动时会同时校验模型参数、文件完整性和英雄表兼容性。

训练命令会为每个新的输出目录自动生成 manifest；`report.json` 保存训练报告，
`backtest.json` 保存独立回测和各轮混合权重。当前 UI 可以切换项目内置且经过校验
的模型包，但不接受任意外部 `.npz` 文件。
完整的三套模型对比也保存在 `artifacts/models/BENCHMARK.md`。

默认头像区域来自一张 2560x1440、7.41 版本的真实天梯 BP 截图，并按当前
分辨率等比例缩放。实际截图的 10 个已选英雄均被正确识别。本机 Dota VPK 中的
横版头像和身心/至宝变体也已加入模板库。

当前视觉验证：

- “双方已选满”截图识别为 10/10；“双方 0 选择”截图识别为 0/10，十个空槽均无误报；
- Dota UI 缩放、自定义宽高比或多显示器布局可能需要重新标定坐标；
- 独占全屏下系统截图可能返回黑屏，建议使用无边框窗口模式；
- 工具只读取屏幕，不读取内存、不注入 Dota 进程。

## 评估指标

- Value：AUC、Log Loss、Brier Score、Accuracy、校准误差 ECE。
- Policy：遮挡英雄的 Hit@5、Hit@10 和 MRR。
- 所有切分按时间进行，较新的比赛只用于测试。
- 报告同时给出 Pairwise baseline 与 Neural model，避免只看神经网络自己的绝对数字。

## 已知限制

- 极少数比赛的 `picks_bans` 不完整，采集器会将其标记为无效。
- 公开比赛只能提供事实阵容，不能提供“换成另一个英雄会怎样”的反事实结果。
- 首版 rollout 将同一英雄视为全局不可重复，没有模拟双方同轮撞英雄后的重选细节。
- 没有位置、玩家熟练度和专家标签。
- MVP 的“胜率”首先用于候选排序；校准不足时不应作为精确百分比展示给用户。

## 许可证与声明

原创源代码采用 MIT License。Dota 2、英雄名称和英雄图像归 Valve Corporation
所有；相关图像只用于这个非官方、非商业同人项目的界面显示与截图识别，不包含在
MIT 授权中。本项目与 Valve、OpenDota、STRATZ 无隶属或背书关系。

发布物不包含训练数据库、原始 API 响应、玩家资料或 API Key。详见
`THIRD_PARTY_NOTICES.md` 与 `DATA_SOURCES.md`。
