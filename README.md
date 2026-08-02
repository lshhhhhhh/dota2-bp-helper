# Dota 2 BP Helper

[中文](#中文) | [English](#english)

## 中文

一个面向 Dota 2 天梯三轮同时盲选规则的离线 BP 助手。它可以识别当前屏幕或已有截图中的双方阵容，并同时为天辉和夜魇推荐下一手英雄。正常使用完全离线，不需要 API Key。

### 下载与运行

普通用户请从 [GitHub Releases](https://github.com/lshhhhhhh/dota2-bp-helper/releases/latest) 下载 `Dota2BPHelper-0.2.0-win64.zip`：

1. 将压缩包完整解压到任意目录；不要直接在压缩包内运行。
2. 双击 `Dota2BPHelper.exe`。
3. 在窗口右上角选择 `中文` 或 `English`，界面会立即切换语言。

这是便携版，不需要安装 Python。应用不会读取 Dota 2 进程内存、注入游戏或自动操作鼠标键盘；屏幕识别只处理桌面截图。

### 功能

- 中英文界面即时切换，包括主界面、模型页、Benchmark、状态信息和错误提示；
- 一键扫描所有显示器，也可以固定选择某一块屏幕；
- 自动定位显示器内带黑边的 16:9 窗口化直播或录像画面；
- 读取本地 BP 截图；
- 自动按双方已公开英雄数推断第 1、2、3 轮；
- 天辉和夜魇各五个固定头像槽，显示识别置信度；
- 暗色的“建议选择”头像不会被当作已经锁定的英雄；
- 点击头像移除，或手动向任一方添加英雄；
- 输入提示支持官方中英文名、拼音、内部名、常用缩写与国服绰号，例如 `主宰`、`剑圣`、`jugg`；
- 同时给两边推荐下一手；对方推荐也代表己方需要防范的候选；
- 推荐列表显示英雄头像、选择概率和胜率倾向；
- 内置传奇及以上、统帅及以下、全段位三套模型。

独占全屏模式有时会让系统截图返回黑屏，建议使用无边框窗口模式。直播窗口可以不最大化，但画面过小、被遮挡或使用非 16:9 比例时可能降低识别率。

### 模型与 App 解耦

系统分成四层：

1. `DraftState`：只描述公开阵容和轮次；
2. `PolicyModel`：根据当前状态估计下一手的选择分布；
3. `ValueModel`：估计完整阵容的胜率倾向；
4. 桌面端：负责截图识别、状态管理和界面展示。

桌面端通过每个模型目录内的 `model_manifest.json` 发现并加载模型。Manifest 声明模型 ID、Dota 版本、适用分段、英雄表、输入输出契约和文件 SHA-256。应用只允许切换随项目发布且通过兼容性校验的模型包，不直接载入任意 `.npz` 文件。

第一轮双方没有公开阵容信息，模型按当前版本的选择频率排序。第二、三轮把双方已选英雄和轮次输入一个小型单隐藏层神经网络；Value 模块按独立回测选出的权重加入胜率倾向。当前模型没有使用位置、分路、玩家熟练度，或“先手”“幻象处理”等专家标签。

### Benchmark 应该怎样理解

当前发布模型在按时间留出的测试集上，第三轮的前 10 覆盖率如下：

| 适用人群 | 阵容模型前 10 | 完全不看阵容的常见选择榜 | 每 100 次多覆盖 |
| --- | ---: | ---: | ---: |
| 传奇及以上 | 40.57% | 29.09% | +11.48 |
| 统帅及以下 | 40.83% | 33.76% | +7.07 |
| 全段位 | 41.65% | 31.59% | +10.06 |

“前 10 覆盖率”表示真实玩家下一手出现在推荐前十中的比例。第一轮没有公开阵容信息，所以阵容模型不应比常见选择榜更强；第三轮已有公开 4v4 阵容，模型优势最明显。

历史对局中，选择模型前五推荐的组别有约 `+0.8` 至 `+1.5` 个百分点的观察胜率差，前十为约 `+0.9` 至 `+2.2` 个百分点，但粗略 95% 区间仍跨过零。玩家并非随机分配到英雄，因此这只是历史关联，不能宣传成模型已被证明可以因果性提高相同数值的胜率。完整数字见 [`artifacts/models/BENCHMARK.md`](artifacts/models/BENCHMARK.md)。

### 从源码运行

需要 Python 3.11+：

```powershell
python -m pip install -e .
python dota2_bp_helper.py
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

构建 Windows 便携版：

```powershell
python -m pip install -e ".[build]"
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1 -Version 0.2.0
```

构建结果位于 `dist/Dota2BPHelper-0.2.0-win64.zip`。项目没有自动 GitHub Actions 工作流；测试和发布由维护者手动执行。

### 数据与训练

训练数据来自 OpenDota 的公开逐场比赛数据。采集器保存规范化 SQLite 数据并支持按 Dota 版本隔离、限速、断点续传、去重和硬请求上限。公开仓库及便携版不包含原始 API 响应、训练数据库、玩家资料或 API Key。

仓库内的三个现役 7.41 模型使用 56,625 场可重建 BP 的对局训练。研究命令和数据结构保留在 `d2draft/`；若需要重新采集，应自行在 `.env` 设置 `OPENDOTA_API_KEY`，并显式指定请求预算。不要提交 `.env`。

主要离线指标：

- Policy：Hit@5、Hit@10、MRR 和中位排名；
- Value：AUC、Log Loss、Brier Score、Accuracy 和 ECE；
- 数据按时间切分，较新的比赛只用于最终测试；
- 报告同时给出不看阵容的常见选择榜，避免只看神经网络的绝对数字。

### 已知限制

- 公开比赛无法告诉我们“同一局改选另一个英雄会怎样”，因此不能直接得到反事实胜率；
- 极少数比赛的 `picks_bans` 不完整，采集器会将其标记为无效；
- MVP 将同一英雄视为全局不可重复，没有完整模拟双方同轮撞英雄后的重选过程；
- 视觉模块可能需要针对新的 Dota UI、缩放方式或宽高比重新标定；
- 胜率倾向主要用于候选排序，不应当作精确的个人胜率预测。

### 许可证与声明

原创源代码采用 MIT License。Dota 2、英雄名称和英雄图像归 Valve Corporation 所有；相关图像只用于这个非官方、非商业同人项目的界面显示与截图识别，不包含在 MIT 授权中。本项目与 Valve、OpenDota、STRATZ 无隶属或背书关系。

详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)、[`DATA_SOURCES.md`](DATA_SOURCES.md) 和 [`data/ASSET_NOTICE.md`](data/ASSET_NOTICE.md)。

---

## English

An offline draft assistant for Dota 2 ranked All Pick's three simultaneous blind-pick rounds. It recognizes both lineups from the current screen or an existing screenshot and recommends the next hero for both Radiant and Dire. Normal use is fully offline and requires no API key.

### Download and run

Download `Dota2BPHelper-0.2.0-win64.zip` from [GitHub Releases](https://github.com/lshhhhhhh/dota2-bp-helper/releases/latest):

1. Extract the whole archive to any folder; do not run it from inside the ZIP.
2. Double-click `Dota2BPHelper.exe`.
3. Choose `中文` or `English` in the upper-right corner. The UI switches immediately.

The portable build does not require Python. The app does not read Dota 2 process memory, inject into the game, or control the mouse or keyboard. Screen recognition only processes desktop screenshots.

### Features

- Instant Chinese/English switching for the main window, model page, benchmark, status messages, and errors;
- Scan all displays automatically or select one display explicitly;
- Locate a letterboxed 16:9 Dota viewport inside a windowed stream or recording;
- Open an existing draft screenshot;
- Infer round 1, 2, or 3 from the number of revealed heroes on both sides;
- Five fixed portrait slots per side with recognition confidence;
- Ignore dim “suggested pick” portraits that have not been locked in;
- Click a portrait to remove it, or add a hero manually to either side;
- Type-ahead search across official Chinese/English names, pinyin, internal names, abbreviations, and common Chinese nicknames—for example `主宰`, `剑圣`, or `jugg`;
- Recommend for both sides at once; the opposing recommendation is also a hero your side should be ready for;
- Show hero portraits, pick probabilities, and win-rate tendency in the recommendation list;
- Three built-in models: Legend and above, Archon and below, and all ranks.

Exclusive fullscreen can cause Windows screen capture to return a black frame, so borderless windowed mode is recommended. A stream window does not need to be maximized, but a very small, occluded, or non-16:9 viewport may reduce recognition quality.

### Model/app separation

The system has four layers:

1. `DraftState` describes only the public lineup and round;
2. `PolicyModel` estimates the next-pick distribution for the current state;
3. `ValueModel` estimates the win-rate tendency of a completed lineup;
4. The desktop app handles screen recognition, state management, and presentation.

The app discovers each model through its `model_manifest.json`. The manifest declares the model ID, Dota patch, skill bracket, hero catalog, input/output contract, and SHA-256 file hash. Only bundled models that pass compatibility checks can be selected; arbitrary `.npz` files cannot be loaded directly.

In round 1, neither side has public lineup information, so heroes are ranked by pick frequency for the patch. In rounds 2 and 3, both revealed lineups and the round are fed into a small one-hidden-layer neural network. A Value module blends in win-rate tendency using a weight selected by a held-out backtest. The current model does not use roles, lanes, player proficiency, or expert tags such as initiative and illusion clear.

### How to read the benchmark

On the chronologically held-out test set, round-3 Hit@10 is:

| Population | Lineup-aware model | Pick-frequency list with no lineup | Extra hits per 100 decisions |
| --- | ---: | ---: | ---: |
| Legend and above | 40.57% | 29.09% | +11.48 |
| Archon and below | 40.83% | 33.76% | +7.07 |
| All ranks | 41.65% | 31.59% | +10.06 |

Hit@10 is the share of actual next picks found in the top ten recommendations. With no public lineup in round 1, a lineup-aware model should not beat the pick-frequency list. Its largest advantage appears in round 3, where a public 4v4 lineup is available.

Historically, groups whose actual pick fell inside the model's top five had an observed win-rate difference of roughly `+0.8` to `+1.5` percentage points; top-ten differences were about `+0.9` to `+2.2` points. However, the approximate 95% intervals still cross zero, and players were not randomly assigned to picks. These are associations, not proof that the model causally increases win rate by the same amount. See [`artifacts/models/BENCHMARK.md`](artifacts/models/BENCHMARK.md) for the full results.

### Run from source

Python 3.11+ is required:

```powershell
python -m pip install -e .
python dota2_bp_helper.py
```

Run the test suite:

```powershell
python -m unittest discover -s tests -v
```

Build the Windows portable archive:

```powershell
python -m pip install -e ".[build]"
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1 -Version 0.2.0
```

The output is `dist/Dota2BPHelper-0.2.0-win64.zip`. This repository has no automatic GitHub Actions workflow; maintainers run tests and releases manually.

### Data and training

Training data comes from OpenDota's public per-match data. The collector stores normalized SQLite records and supports patch isolation, rate limiting, resume, deduplication, and a hard request cap. Neither the public repository nor the portable build includes raw API responses, the training database, player profiles, or API keys.

The three bundled 7.41 models were trained on 56,625 reconstructable drafts. Research commands and data structures remain under `d2draft/`. To collect new data, set `OPENDOTA_API_KEY` in your own `.env` and specify an explicit request budget. Never commit `.env`.

Primary offline metrics:

- Policy: Hit@5, Hit@10, MRR, and median rank;
- Value: AUC, Log Loss, Brier Score, Accuracy, and ECE;
- Data is split chronologically, with newer matches reserved for final testing;
- Reports include the no-lineup pick-frequency list so the neural network is not judged only by absolute numbers.

### Known limitations

- Public matches cannot tell us what would have happened if the same player had chosen a different hero, so counterfactual win rate is not directly observable;
- A small number of matches have incomplete `picks_bans`; the collector marks them invalid;
- The MVP treats every hero as globally unique and does not fully simulate repicks after simultaneous duplicate attempts;
- New Dota UI layouts, scaling, or aspect ratios may require vision recalibration;
- Win-rate tendency is primarily a ranking signal, not a precise personal win-probability forecast.

### License and notices

Original source code is licensed under the MIT License. Dota 2, hero names, and hero images belong to Valve Corporation. Those images are used only for UI display and screenshot recognition in this unofficial, non-commercial fan project and are not covered by the MIT license. This project is not affiliated with or endorsed by Valve, OpenDota, or STRATZ.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`DATA_SOURCES.md`](DATA_SOURCES.md), and [`data/ASSET_NOTICE.md`](data/ASSET_NOTICE.md).
