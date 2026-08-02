from __future__ import annotations

import argparse
import json
import sqlite3
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageGrab, ImageTk

from .benchmark import approximate_ab_test_matches
from .model_bundle import ModelBundle
from .recommender import HeroCatalog, HeroInfo, HybridRecommender, normalize_hero_name
from .screen_capture import MonitorInfo, enumerate_monitors, rectangles_intersect
from .state import DraftState
from .vision import (
    CaptureConfig,
    PortraitMatcher,
    ViewportCandidate,
    VisionMatch,
    locate_windowed_viewports,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "mvp"
MODEL_COLLECTION_DIR = ROOT / "artifacts" / "models"
BG = "#10151b"
PANEL = "#19212a"
CARD = "#222c36"
TEXT = "#e8edf2"
MUTED = "#91a0ad"
RADIANT = "#72b043"
DIRE = "#d05252"
ACCENT = "#d6a756"


class DraftDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Dota 2 BP Helper")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width, height = min(1240, screen_width), min(880, screen_height)
        window_x = max(0, screen_width - width - 24)
        window_y = max(0, (screen_height - height) // 2)
        self.root.geometry(f"{width}x{height}+{window_x}+{window_y}")
        self.root.minsize(1000, 720)
        self.root.configure(bg=BG)

        self.catalog = HeroCatalog(ROOT / "data" / "heroes.json")
        self.model_choices = self._discover_model_bundles()
        self.model_bundle = next(
            (
                bundle
                for bundle in self.model_choices.values()
                if bundle.rank_bracket_id == "legend_plus"
            ),
            next(iter(self.model_choices.values())),
        )
        self.recommender = HybridRecommender(self.model_bundle.artifact_path, self.catalog)
        portrait_dirs = [ROOT / "data" / "hero_portraits"]
        self.game_portraits = (
            ROOT / "data" / "game_hero_images" / "panorama" / "images" / "heroes"
        )
        if self.game_portraits.exists():
            portrait_dirs.append(self.game_portraits)
        self.matcher = PortraitMatcher(portrait_dirs, self.catalog)
        self.config_path = ROOT / "data" / "screen_config.json"

        self.model_patch = self.model_bundle.patch_label
        self.latest_data_patch = self._read_latest_data_patch()
        self._model_window: tk.Toplevel | None = None

        self.phase_var = tk.StringVar(value="自动")
        self.model_choice_var = tk.StringVar(value=self._model_choice_label(self.model_bundle))
        self.model_header_var = tk.StringVar(value=self._model_header_text())
        self.screen_var = tk.StringVar(value="自动（所有屏幕）")
        self.monitors: list[MonitorInfo] = []
        self.monitor_choices: dict[str, MonitorInfo | None] = {}
        self.monitor_combo: ttk.Combobox | None = None
        self._refresh_monitors(screen_width, screen_height)
        patch_status = f"模型 {self.model_bundle.rank_bracket_label} · 版本 {self.model_patch}"
        if self.latest_data_patch and self.latest_data_patch != self.model_patch:
            patch_status += f"；数据最新版本 {self.latest_data_patch}，需要重新训练"
        self.status_var = tk.StringVar(value=f"就绪：读取截图或识别当前屏幕 · {patch_status}")
        self.topmost_var = tk.BooleanVar(value=True)
        self.auto_var = tk.BooleanVar(value=False)
        self.team_ids: dict[str, list[int | None]] = {
            "radiant": [None] * 5,
            "dire": [None] * 5,
        }
        self.team_confidence: dict[str, list[float | None]] = {
            "radiant": [None] * 5,
            "dire": [None] * 5,
        }
        self.input_vars = {side: tk.StringVar() for side in self.team_ids}
        self.slot_buttons: dict[str, list[tk.Button]] = {"radiant": [], "dire": []}
        self.count_labels: dict[str, ttk.Label] = {}
        self.suggestion_ids: dict[str, dict[str, int]] = {"radiant": {}, "dire": {}}
        self.photo_cache: dict[int | str, ImageTk.PhotoImage] = {}
        self.result_titles = {
            "radiant": tk.StringVar(value="天辉下一手推荐 · 夜魇需防范"),
            "dire": tk.StringVar(value="夜魇下一手推荐 · 天辉需防范"),
        }
        self.trees: dict[str, ttk.Treeview] = {}

        self._configure_styles()
        self._build_ui()
        self.root.attributes("-topmost", True)

    @staticmethod
    def _model_choice_label(bundle: ModelBundle) -> str:
        return f"{bundle.rank_bracket_label} · Dota {bundle.patch_label}"

    def _discover_model_bundles(self) -> dict[str, ModelBundle]:
        directories = [MODEL_DIR]
        if MODEL_COLLECTION_DIR.exists():
            directories.extend(
                path.parent
                for path in sorted(MODEL_COLLECTION_DIR.glob("*/model_manifest.json"))
            )
        bundles: list[ModelBundle] = []
        seen: set[Path] = set()
        for directory in directories:
            resolved = directory.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            bundles.append(
                ModelBundle.load(directory, expected_hero_ids=self.catalog.by_id)
            )
        priority = {"legend_plus": 0, "archon_below": 1, "all": 2}
        bundles.sort(
            key=lambda bundle: (
                priority.get(bundle.rank_bracket_id, 9),
                bundle.patch_label,
                bundle.model_id,
            )
        )
        return {self._model_choice_label(bundle): bundle for bundle in bundles}

    def _model_header_text(self) -> str:
        return (
            f"  图像识别 · 双向阵容推荐 · "
            f"{self.model_bundle.rank_bracket_label}模型 {self.model_patch}"
        )

    def _read_latest_data_patch(self) -> str | None:
        database = ROOT / "data" / "collection" / "draft_matches.sqlite3"
        if not database.exists():
            return None
        try:
            connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
            row = connection.execute(
                "SELECT canonical_patch FROM candidates WHERE canonical_patch IS NOT NULL "
                "ORDER BY start_time DESC LIMIT 1"
            ).fetchone()
            connection.close()
            return str(row[0]) if row else None
        except sqlite3.Error:
            return None

    def _configure_styles(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=CARD)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Microsoft YaHei UI", 17, "bold"))
        style.configure("Team.TLabel", background=PANEL, foreground=TEXT, font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("TButton", padding=(10, 7), background=CARD, foreground=TEXT)
        style.map("TButton", background=[("active", "#303d49")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#121212", font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#e5bd73")])
        style.configure("TCheckbutton", background=BG, foreground=TEXT)
        style.configure(
            "Dark.TCombobox",
            fieldbackground=CARD,
            background=CARD,
            foreground=TEXT,
            arrowcolor=TEXT,
            selectbackground="#34424e",
            selectforeground=TEXT,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", CARD), ("focus", CARD), ("!disabled", CARD)],
            foreground=[("readonly", TEXT), ("focus", TEXT), ("!disabled", TEXT)],
            selectbackground=[("readonly", CARD), ("focus", "#34424e")],
            selectforeground=[("readonly", TEXT), ("focus", TEXT)],
        )
        # The dropdown list is a classic Tk Listbox and is not controlled by ttk styles.
        self.root.option_add("*TCombobox*Listbox.background", CARD)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#34424e")
        self.root.option_add("*TCombobox*Listbox.selectForeground", TEXT)
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=38, borderwidth=0)
        style.map("Treeview", background=[("selected", "#3b4b59")])
        style.configure("Treeview.Heading", background=CARD, foreground=TEXT, relief="flat", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(18, 14, 18, 10))
        header.pack(fill="x")
        ttk.Label(header, text="DOTA 2  BP HELPER", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            textvariable=self.model_header_var,
            style="Muted.TLabel",
        ).pack(side="left", pady=(6, 0))

        controls = ttk.Frame(header)
        controls.pack(side="right")
        ttk.Button(controls, text="模型", command=self.show_model_info).pack(side="left", padx=3)
        ttk.Button(controls, text="识别屏幕", command=self.capture_screen).pack(side="left", padx=3)
        ttk.Button(controls, text="读取截图", command=self.open_screenshot).pack(side="left", padx=3)

        options = ttk.Frame(self.root, padding=(18, 0, 18, 10))
        options.pack(fill="x")
        ttk.Label(options, text="BP轮次").pack(side="left")
        phase_combo = ttk.Combobox(
            options,
            textvariable=self.phase_var,
            values=("自动", "1", "2", "3"),
            width=6,
            state="readonly",
            style="Dark.TCombobox",
        )
        phase_combo.pack(side="left", padx=(6, 18))
        phase_combo.bind("<<ComboboxSelected>>", lambda _event: self._maybe_recommend())
        ttk.Label(options, text="识别来源").pack(side="left")
        self.monitor_combo = ttk.Combobox(
            options,
            textvariable=self.screen_var,
            values=tuple(self.monitor_choices),
            width=25,
            state="readonly",
            style="Dark.TCombobox",
        )
        self.monitor_combo.pack(side="left", padx=(6, 18))
        ttk.Checkbutton(options, text="窗口置顶", variable=self.topmost_var, command=self.toggle_topmost).pack(side="left")
        ttk.Checkbutton(options, text="每2秒自动识别", variable=self.auto_var, command=self.toggle_auto).pack(side="left", padx=(14, 0))
        ttk.Label(options, text="点击已选英雄即可移除；手动输入支持中文、英文、拼音、缩写与常用绰号", style="Muted.TLabel").pack(side="right")

        teams = ttk.Frame(self.root, padding=(18, 0, 18, 8))
        teams.pack(fill="x")
        self._build_team_row(teams, "radiant", "天辉", RADIANT)
        self._build_team_row(teams, "dire", "夜魇", DIRE)

        results = ttk.Frame(self.root, padding=(18, 4, 18, 10))
        results.pack(fill="both", expand=True)
        results.columnconfigure(0, weight=1)
        results.columnconfigure(1, weight=1)
        results.rowconfigure(0, weight=1)
        self._build_result_panel(results, "radiant", 0)
        self._build_result_panel(results, "dire", 1)

        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#0b0f13",
            fg=MUTED,
            anchor="w",
            padx=18,
            pady=7,
            font=("Microsoft YaHei UI", 9),
        )
        status.pack(fill="x", side="bottom")

    @staticmethod
    def _metric(value: object, *, percent: bool = False) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{number:.1%}" if percent else f"{number:g}"

    def _model_overview_text(self) -> str:
        manifest = self.model_bundle.manifest
        report = self.model_bundle.report
        policy = report.get("policy", {})
        matches = int(policy.get("train_matches", 0)) + int(policy.get("test_matches", 0))
        examples = int(policy.get("train_examples", 0)) + int(policy.get("test_examples", 0))
        created = str(manifest.get("created_at_utc", "未知"))
        return (
            f"{self.model_bundle.display_name}\n\n"
            f"状态　　　　　已加载 · 兼容性与文件完整性校验通过\n"
            f"模型 ID　　　 {self.model_bundle.model_id}\n"
            f"适用版本　　　Dota 2 {self.model_patch}\n"
            f"适用分段　　　{self.model_bundle.rank_bracket_label}\n"
            f"生成时间　　　{created} (UTC)\n"
            f"英雄表　　　　{len(self.model_bundle.hero_ids)} 个英雄\n"
            f"训练对局　　　{matches:,} 场可重建 BP\n"
            f"训练决策样本　{examples:,} 个下一手选择\n"
            f"模型包格式　　v{manifest.get('format_version', '—')}\n"
            f"模型文件　　　{self.model_bundle.artifact_path.name}\n"
            f"SHA-256　　　{self.model_bundle.short_hash}…\n\n"
            "这里只允许切换项目内置且通过校验的模型包，不接受任意 .npz 文件。"
            "这样可以防止模型与英雄表、输入维度或游戏版本不兼容。"
        )

    def _model_principle_text(self) -> str:
        blend = self.model_bundle.backtest.get("selected_value_blend", {})
        return (
            "模型与 App 的边界\n\n"
            "App 负责截图识别、BP 状态管理和界面展示；模型只接收 BP 轮次、"
            "天辉英雄 ID、夜魇英雄 ID，输出所有合法英雄的排序及分数。模型包"
            "通过 manifest 声明版本、英雄表规模、输入输出契约和文件哈希。\n\n"
            "当前推荐原理\n\n"
            "第一轮：双方没有阵容信息，使用该版本的英雄选择频率作为先验。\n\n"
            "第二、三轮：把双方已出现的英雄编码成两个 127 维向量，加上 BP "
            "轮次，输入单隐藏层神经网络，学习真实比赛中下一手通常会选择谁。\n\n"
            "Value 模块：估计候选英雄带来的胜率倾向，并按独立回测选择的权重"
            f"混入排序。目前第一/二/三轮权重分别为 "
            f"{blend.get('phase_1', 0):g} / {blend.get('phase_2', 0):g} / "
            f"{blend.get('phase_3', 0.1):g}。\n\n"
            "当前没有加入位置、分路、玩家熟练度或“先手/幻象处理”等专家标签；"
            "这些可以作为以后可替换模型的输入扩展，但不属于当前模型契约。"
        )

    def _model_metrics_text(self) -> str:
        backtest = self.model_bundle.backtest
        selected = backtest.get("final_test_selected", {})
        baseline = backtest.get("final_test_policy_baseline", {})
        lines = [
            "按时间切分的最终测试集",
            "",
            "轮次    样本数    前5覆盖率  前10覆盖率  中位排名  不看阵容榜前10",
        ]
        for phase in (1, 2, 3):
            values = selected.get(f"phase_{phase}", {})
            base = baseline.get(f"phase_{phase}", {})
            lines.append(
                f"第 {phase} 轮   {int(values.get('examples', 0)):>5,}    "
                f"{self._metric(values.get('hit_at_5'), percent=True):>6}   "
                f"{self._metric(values.get('hit_at_10'), percent=True):>6}   "
                f"{self._metric(values.get('median_rank')):>6}       "
                f"{self._metric(base.get('hit_at_10'), percent=True):>6}"
            )
        value = self.model_bundle.report.get("value", {})
        baseline_value = value.get("baseline", {})
        lines.extend(
            [
                "",
                f"胜率倾向模块：AUC {self._metric(baseline_value.get('auc'))}，"
                f"LogLoss {self._metric(baseline_value.get('log_loss'))}，"
                f"Accuracy {self._metric(baseline_value.get('accuracy'), percent=True)}。",
                "",
                "解释：“前10覆盖率”表示实际下一手出现在推荐前 10 名中的比例。它衡量模型"
                "与真实选择的一致性，不等于推荐一定最优，也不能证明反事实胜率提升。",
            ]
        )
        return "\n".join(lines)

    def _model_benchmark_text(self) -> str:
        report = self.model_bundle.advantage_benchmark
        if not report:
            return (
                "这套模型还没有生成历史胜率关联报告。\n\n"
                "请先运行 d2draft.advantage_benchmark。"
            )
        top_one = report["groups"]["top_1"]
        top_five = report["groups"]["top_5"]
        low, high = top_five["approximate_95_ci_points"]
        lines = [
            f"{self.model_bundle.rank_bracket_label}模型 · 历史对局中的胜率关联",
            "",
            "我们要回答的问题：当真实玩家最后一手与模型推荐一致时，胜率是否更高？",
            "",
            "把“采纳推荐”定义为选择模型前 5 名内的英雄：",
            "",
            f"选择前 5 名内英雄：胜率 {float(top_five['followed_win_rate']):.1%} "
            f"（{int(top_five['followed_decisions']):,} 次）",
            f"选择前 5 名外英雄：胜率 {float(top_five['other_win_rate']):.1%} "
            f"（{int(top_five['other_decisions']):,} 次）",
            f"观察到的胜率差：　 {float(top_five['observed_difference_points']):+.1f} 个百分点",
            "",
            "更严格地只看模型第 1 推荐：",
            f"选择第 1 推荐时胜率 {float(top_one['followed_win_rate']):.1%}，"
            f"其他选择 {float(top_one['other_win_rate']):.1%}，"
            f"观察差值 {float(top_one['observed_difference_points']):+.1f} 个百分点"
            f"（仅 {int(top_one['followed_decisions']):,} 次匹配）。",
            "",
            "证据强度",
            f"前 5 差值的粗略 95% 区间为 {float(low):+.1f} 到 {float(high):+.1f} 个百分点。",
        ]
        required_matches = approximate_ab_test_matches()
        lines.extend(
            [
                "",
                "当前结论",
                "历史数据呈现正相关，但区间仍包含 0；而且玩家选择并非随机分配，"
                "所以现在可以说“历史测试中观察到胜率优势”，还不能说模型导致了"
                "这些胜率提升。",
                "",
                f"若要验证从 50% 到 53% 的实际胜率提升，在理想的 1:1 随机对照、"
                f"95% 显著性和 80% 检验功效下，约需 {required_matches:,} 场完成对局；"
                "用户不采纳推荐时还需要更多样本。",
            ]
        )
        return "\n".join(lines)

    def show_model_info(self) -> None:
        if self._model_window is not None and self._model_window.winfo_exists():
            self._model_window.lift()
            self._model_window.focus_force()
            return

        window = tk.Toplevel(self.root)
        self._model_window = window
        window.title("模型信息")
        window.geometry("760x620")
        window.minsize(680, 520)
        window.configure(bg=BG)
        window.transient(self.root)
        window.attributes("-topmost", bool(self.topmost_var.get()))

        header = ttk.Frame(window, padding=(18, 16, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text="模型", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text=f"  {self.model_bundle.display_name} · Dota 2 {self.model_patch}",
            style="Muted.TLabel",
        ).pack(side="left", pady=(6, 0))

        notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True, padx=18, pady=(4, 10))

        def add_tab(title: str, content: str) -> None:
            frame = ttk.Frame(notebook, style="Panel.TFrame", padding=8)
            text_widget = tk.Text(
                frame,
                wrap="word",
                bg=PANEL,
                fg=TEXT,
                insertbackground=TEXT,
                relief="flat",
                padx=14,
                pady=12,
                font=("Microsoft YaHei UI", 10),
                spacing1=2,
                spacing3=4,
            )
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            text_widget.insert("1.0", content)
            text_widget.configure(state="disabled")
            notebook.add(frame, text=title)

        add_tab("总览", self._model_overview_text())
        add_tab("原理", self._model_principle_text())
        add_tab("验证指标", self._model_metrics_text())
        add_tab("基准对比", self._model_benchmark_text())

        footer = ttk.Frame(window, padding=(18, 0, 18, 14))
        footer.pack(fill="x")
        ttk.Label(footer, text="推荐分段").pack(side="left")
        model_combo = ttk.Combobox(
            footer,
            textvariable=self.model_choice_var,
            values=tuple(self.model_choices),
            width=28,
            state="readonly",
            style="Dark.TCombobox",
        )
        model_combo.pack(side="left", padx=(8, 8))

        def apply_model() -> None:
            self._activate_model(self.model_choice_var.get())
            close_window()
            self.show_model_info()

        ttk.Button(footer, text="应用模型", command=apply_model).pack(side="left")
        ttk.Label(
            footer,
            text="仅列出通过兼容性校验的内置模型",
            style="Muted.TLabel",
        ).pack(side="right")

        def close_window() -> None:
            self._model_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    def _activate_model(self, choice: str) -> None:
        bundle = self.model_choices.get(choice)
        if bundle is None or bundle.model_id == self.model_bundle.model_id:
            return
        self.model_bundle = bundle
        self.model_patch = bundle.patch_label
        self.recommender = HybridRecommender(bundle.artifact_path, self.catalog)
        self.model_header_var.set(self._model_header_text())
        self.status_var.set(
            f"已切换到 {bundle.rank_bracket_label}模型 · Dota {bundle.patch_label}"
        )
        self._maybe_recommend()

    def _build_team_row(self, parent: ttk.Frame, side: str, label: str, color: str) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=(12, 10))
        panel.pack(fill="x", pady=4)
        tk.Frame(panel, width=5, bg=color).pack(side="left", fill="y", padx=(0, 12))
        title = ttk.Frame(panel, style="Panel.TFrame", width=86)
        title.pack(side="left", fill="y", padx=(0, 8))
        title.pack_propagate(False)
        ttk.Label(title, text=label, style="Team.TLabel").pack(anchor="w", pady=(11, 0))
        count_label = ttk.Label(title, text="0 / 5", style="Panel.TLabel", foreground=MUTED)
        count_label.pack(anchor="w", pady=(4, 0))
        self.count_labels[side] = count_label

        slots = ttk.Frame(panel, style="Panel.TFrame")
        slots.pack(side="left", fill="x", expand=True)
        empty = self._empty_photo()
        for index in range(5):
            button = tk.Button(
                slots,
                image=empty,
                text=f"空位 {index + 1}",
                compound="top",
                command=lambda s=side, i=index: self.remove_slot(s, i),
                bg=CARD,
                fg=MUTED,
                activebackground="#303d49",
                activeforeground=TEXT,
                relief="flat",
                bd=0,
                padx=4,
                pady=4,
                width=118,
                height=86,
                font=("Microsoft YaHei UI", 8),
                cursor="hand2",
            )
            button.pack(side="left", padx=3)
            self.slot_buttons[side].append(button)

        manual = ttk.Frame(panel, style="Panel.TFrame", width=250)
        manual.pack(side="right", fill="y", padx=(10, 0))
        manual.pack_propagate(False)
        ttk.Label(manual, text=f"手动添加到{label}", style="Panel.TLabel").pack(anchor="w")
        combo = ttk.Combobox(
            manual,
            textvariable=self.input_vars[side],
            width=28,
            style="Dark.TCombobox",
        )
        combo.pack(fill="x", pady=(5, 5))
        combo.bind(
            "<KeyRelease>",
            lambda event, s=side, c=combo: self._update_suggestions(
                s, c, event.keysym
            ),
        )
        combo.bind("<<ComboboxSelected>>", lambda event, s=side: self.add_from_input(s))
        combo.bind("<Return>", lambda event, s=side: self.add_from_input(s))
        combo.bind("<Escape>", lambda event, s=side: self.input_vars[s].set(""))
        ttk.Button(manual, text="添加英雄", command=lambda s=side: self.add_from_input(s)).pack(fill="x")

    def _build_result_panel(self, parent: ttk.Frame, side: str, column: int) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        panel.grid(row=0, column=column, sticky="nsew", padx=(0, 5) if column == 0 else (5, 0))
        ttk.Label(panel, textvariable=self.result_titles[side], style="Team.TLabel").pack(anchor="w", pady=(0, 8))
        columns = ("rank", "hero", "policy", "value")
        tree = ttk.Treeview(panel, columns=columns, show=("tree", "headings"), height=10)
        tree.heading("#0", text="头像")
        tree.column("#0", width=84, minwidth=84, stretch=False, anchor="center")
        for key, title, width, anchor in (
            ("rank", "#", 34, "center"),
            ("hero", "英雄", 230, "w"),
            ("policy", "选择概率", 82, "center"),
            ("value", "胜率倾向", 78, "center"),
        ):
            tree.heading(key, text=title)
            tree.column(key, width=width, minwidth=width, anchor=anchor)
        tree.pack(fill="both", expand=True)
        self.trees[side] = tree

    def _empty_photo(self) -> ImageTk.PhotoImage:
        if "empty" not in self.photo_cache:
            image = Image.new("RGB", (112, 55), "#151c23")
            draw = ImageDraw.Draw(image)
            draw.rectangle((1, 1, 110, 53), outline="#35424e", width=1)
            draw.line((42, 27, 70, 27), fill="#52606c", width=2)
            self.photo_cache["empty"] = ImageTk.PhotoImage(image)
        return self.photo_cache["empty"]

    def _portrait_path(self, hero: HeroInfo) -> Path | None:
        basic = ROOT / "data" / "hero_portraits" / f"{hero.hero_id}.png"
        if basic.exists():
            return basic
        short = hero.internal_name.removeprefix("npc_dota_hero_")
        candidates = sorted(self.game_portraits.glob(f"{short}*.png"))
        return candidates[0] if candidates else None

    def _hero_photo(self, hero_id: int) -> ImageTk.PhotoImage:
        if hero_id not in self.photo_cache:
            hero = self.catalog.info(hero_id)
            path = self._portrait_path(hero)
            if path:
                with Image.open(path) as source:
                    image = source.convert("RGB").resize((112, 55), Image.Resampling.LANCZOS)
            else:
                image = Image.new("RGB", (112, 55), CARD)
            self.photo_cache[hero_id] = ImageTk.PhotoImage(image)
        return self.photo_cache[hero_id]

    def _hero_thumbnail(self, hero_id: int) -> ImageTk.PhotoImage:
        key = f"recommendation:{hero_id}"
        if key not in self.photo_cache:
            hero = self.catalog.info(hero_id)
            path = self._portrait_path(hero)
            if path:
                with Image.open(path) as source:
                    image = source.convert("RGB").resize(
                        (60, 30), Image.Resampling.LANCZOS
                    )
            else:
                image = Image.new("RGB", (60, 30), CARD)
            self.photo_cache[key] = ImageTk.PhotoImage(image)
        return self.photo_cache[key]

    def _refresh_team(self, side: str) -> None:
        for index, button in enumerate(self.slot_buttons[side]):
            hero_id = self.team_ids[side][index]
            confidence = self.team_confidence[side][index]
            if hero_id is None:
                button.configure(image=self._empty_photo(), text=f"空位 {index + 1}", fg=MUTED)
                continue
            hero = self.catalog.info(hero_id)
            name = hero.chinese_name or hero.name
            suffix = f"  {confidence:.0%}" if confidence is not None else "  手动"
            button.configure(image=self._hero_photo(hero_id), text=f"{name}{suffix}", fg=TEXT)
        count = sum(hero_id is not None for hero_id in self.team_ids[side])
        self.count_labels[side].configure(text=f"{count} / 5")

    def _suggestion_label(self, hero: HeroInfo, query: str = "") -> str:
        needle = normalize_hero_name(query)
        candidates = [hero.chinese_name, *hero.aliases, hero.name]
        matched = [
            value
            for value in candidates
            if value and needle and needle in normalize_hero_name(value)
        ]
        matched.sort(
            key=lambda value: (
                not normalize_hero_name(value).startswith(needle),
                len(value),
            )
        )
        primary = hero.display_name
        if matched and normalize_hero_name(matched[0]) not in {
            normalize_hero_name(hero.chinese_name),
            normalize_hero_name(hero.name),
        }:
            primary = f"{matched[0]} · {primary}"
        aliases = [
            alias
            for alias in hero.aliases
            if len(alias) <= 8 and (not matched or alias != matched[0])
        ][:2]
        suffix = f"  ({', '.join(aliases)})" if aliases else ""
        return f"{primary}{suffix}"

    def _update_suggestions(
        self, side: str, combo: ttk.Combobox, keysym: str = ""
    ) -> None:
        query = self.input_vars[side].get().strip()
        heroes = self.catalog.search(query, limit=12) if query else []
        mapping = {
            self._suggestion_label(hero, query): hero.hero_id for hero in heroes
        }
        self.suggestion_ids[side] = mapping
        combo.configure(values=tuple(mapping))
        if query and mapping and keysym not in {
            "Up",
            "Down",
            "Return",
            "Escape",
            "Tab",
        }:
            # Post the dropdown without changing the text or preselecting a hero.
            try:
                combo.tk.call("ttk::combobox::Post", combo._w)
            except tk.TclError:
                combo.event_generate("<Down>")

    def add_from_input(self, side: str) -> None:
        value = self.input_vars[side].get().strip()
        if not value:
            return
        try:
            hero_id = self.suggestion_ids[side].get(value)
            if hero_id is None:
                hero_id = self.catalog.resolve(value)
            if any(hero_id in values for values in self.team_ids.values()):
                raise ValueError(f"{self.catalog.info(hero_id).display_name} 已经在BP中")
            try:
                slot = self.team_ids[side].index(None)
            except ValueError as exc:
                raise ValueError("该阵营五个槽位已满；请先点击头像移除一个英雄") from exc
            self.team_ids[side][slot] = hero_id
            self.team_confidence[side][slot] = None
            self.input_vars[side].set("")
            self._refresh_team(side)
            self.status_var.set(f"已手动添加 {self.catalog.info(hero_id).display_name}")
            self._maybe_recommend()
        except ValueError as exc:
            self.status_var.set(str(exc))
            messagebox.showwarning("无法添加英雄", str(exc))

    def remove_slot(self, side: str, slot: int) -> None:
        hero_id = self.team_ids[side][slot]
        if hero_id is None:
            return
        name = self.catalog.info(hero_id).display_name
        self.team_ids[side][slot] = None
        self.team_confidence[side][slot] = None
        self._refresh_team(side)
        self.status_var.set(f"已移除 {name}")

    def toggle_topmost(self) -> None:
        self.root.attributes("-topmost", bool(self.topmost_var.get()))

    def _refresh_monitors(self, fallback_width: int, fallback_height: int) -> None:
        selected_device = None
        previous = self.monitor_choices.get(self.screen_var.get())
        if previous is not None:
            selected_device = previous.device
        self.monitors = enumerate_monitors(fallback_width, fallback_height)
        choices: dict[str, MonitorInfo | None] = {"自动（所有屏幕）": None}
        for index, monitor in enumerate(self.monitors, 1):
            primary = " · 主屏" if monitor.primary else ""
            position = ""
            primary_monitor = next((item for item in self.monitors if item.primary), None)
            if primary_monitor is not None and not monitor.primary:
                if monitor.left >= primary_monitor.right:
                    position = " · 右侧"
                elif monitor.right <= primary_monitor.left:
                    position = " · 左侧"
                elif monitor.top >= primary_monitor.bottom:
                    position = " · 下方"
                elif monitor.bottom <= primary_monitor.top:
                    position = " · 上方"
            label = f"屏幕 {index}{primary}{position} · {monitor.width}×{monitor.height}"
            choices[label] = monitor
        self.monitor_choices = choices
        restored = next(
            (
                label
                for label, monitor in choices.items()
                if monitor is not None and monitor.device == selected_device
            ),
            "自动（所有屏幕）",
        )
        self.screen_var.set(restored)
        if self.monitor_combo is not None:
            self.monitor_combo.configure(values=tuple(choices))

    def toggle_auto(self) -> None:
        if self.auto_var.get():
            self._auto_tick()

    def _auto_tick(self) -> None:
        if not self.auto_var.get():
            return
        try:
            self.capture_screen(silent=True)
        except Exception as exc:
            self.status_var.set(f"自动识别失败：{type(exc).__name__}")
        self.root.after(2000, self._auto_tick)

    def _config_for(self, image: Image.Image) -> CaptureConfig:
        if self.config_path.exists():
            config = CaptureConfig.load(self.config_path)
            if (config.screen_width, config.screen_height) == image.size:
                return config
        config = CaptureConfig.default_for_screen(*image.size)
        config.save(self.config_path)
        return config

    def _window_overlaps_capture_area(self, monitors: list[MonitorInfo]) -> bool:
        self.root.update_idletasks()
        window = (
            self.root.winfo_rootx(),
            self.root.winfo_rooty(),
            self.root.winfo_rootx() + self.root.winfo_width(),
            self.root.winfo_rooty() + self.root.winfo_height(),
        )
        for monitor in monitors:
            config = CaptureConfig.default_for_screen(monitor.width, monitor.height)
            for box in (config.allies_box, config.enemies_box):
                global_box = (
                    monitor.left + box[0],
                    monitor.top + box[1],
                    monitor.left + box[2],
                    monitor.top + box[3],
                )
                if rectangles_intersect(window, global_box):
                    return True
        return False

    def _capture_monitor(self, monitor: MonitorInfo) -> Image.Image:
        return ImageGrab.grab(bbox=monitor.rect, all_screens=True).convert("RGB")

    def _matches_for(self, screenshot: Image.Image) -> dict[str, list[VisionMatch]]:
        config = self._config_for(screenshot)
        return {
            "radiant": self.matcher.recognize_box(
                screenshot, config.allies_box, orientation=config.orientation
            ),
            "dire": self.matcher.recognize_box(
                screenshot, config.enemies_box, orientation=config.orientation
            ),
        }

    @staticmethod
    def _screen_match_quality(matches: dict[str, list[VisionMatch]]) -> tuple[int, int, float]:
        radiant = sum(match.hero_id is not None for match in matches["radiant"])
        dire = sum(match.hero_id is not None for match in matches["dire"])
        accepted = radiant + dire
        valid_stage = int(radiant == dire and radiant in {0, 2, 4, 5})
        evidence = sum(match.similarity for values in matches.values() for match in values)
        return valid_stage, accepted, evidence

    def capture_screen(self, silent: bool = False) -> None:
        if not silent:
            self.status_var.set("正在扫描显示器并定位 Dota 游戏画面…")
            self.root.update_idletasks()
        self._refresh_monitors(self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        selected = self.monitor_choices.get(self.screen_var.get())
        targets = self.monitors if selected is None else [selected]
        should_hide = self._window_overlaps_capture_area(targets)
        previous_state = self.root.state()
        if should_hide:
            self.root.withdraw()
        try:
            if should_hide:
                self.root.update()
                time.sleep(0.12)
            captures = [(monitor, self._capture_monitor(monitor)) for monitor in targets]
        except Exception as exc:
            self.status_var.set(f"截屏失败：{type(exc).__name__} · 请尝试固定选择一块屏幕")
            if not silent:
                messagebox.showerror(
                    "无法截取屏幕",
                    f"{exc}\n\n请在“识别来源”中固定选择 Dota 所在的屏幕后重试。",
                )
            return
        finally:
            if should_hide:
                self.root.deiconify()
                if previous_state == "zoomed":
                    self.root.state("zoomed")
                self.root.attributes("-topmost", bool(self.topmost_var.get()))
                self.root.lift()

        evaluated = []
        for monitor, image in captures:
            viewport_candidates = [
                ViewportCandidate((0, 0, image.width, image.height), 0.0, "full-screen"),
                *locate_windowed_viewports(image),
            ]
            for viewport in viewport_candidates:
                candidate_image = image.crop(viewport.rect)
                matches = self._matches_for(candidate_image)
                quality = (*self._screen_match_quality(matches), viewport.score)
                evaluated.append(
                    (quality, monitor, candidate_image, matches, viewport.source)
                )
        _, monitor, screenshot, matches, viewport_source = max(
            evaluated, key=lambda item: item[0]
        )
        monitor_number = self.monitors.index(monitor) + 1
        source_suffix = " · 自动定位游戏画面" if viewport_source == "auto" else ""
        self.recognize_image(
            screenshot,
            source=f"屏幕 {monitor_number}{source_suffix}",
            silent=silent,
            matches=matches,
        )

    def open_screenshot(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择 Dota 2 BP 截图",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")],
            initialdir=str(ROOT / "screenshot"),
        )
        if not filename:
            return
        with Image.open(filename) as image:
            screenshot = image.convert("RGB")
        self.recognize_image(screenshot, source=Path(filename).name, silent=False)

    def recognize_image(
        self,
        screenshot: Image.Image,
        *,
        source: str,
        silent: bool = False,
        matches: dict[str, list[VisionMatch]] | None = None,
    ) -> None:
        if matches is None:
            matches = self._matches_for(screenshot)
        accepted_count = sum(match.hero_id is not None for values in matches.values() for match in values)
        # Auto polling outside the BP screen must not erase a manually corrected draft.
        if silent and accepted_count == 0:
            return
        for side, values in matches.items():
            self.team_ids[side] = [match.hero_id for match in values]
            self.team_confidence[side] = [match.similarity if match.hero_id is not None else None for match in values]
            self._refresh_team(side)
        radiant_count = self._team_count("radiant")
        dire_count = self._team_count("dire")
        inferred = self._infer_phase(radiant_count, dire_count, allow_complete=True)
        confidence = [match.similarity for values in matches.values() for match in values if match.hero_id is not None]
        confidence_text = f"，平均置信度 {sum(confidence) / len(confidence):.0%}" if confidence else ""
        if inferred == 4:
            self.status_var.set(f"{source}：识别完整 5v5{confidence_text}，BP 已结束")
            self._clear_results("BP 已结束")
        elif inferred in {1, 2, 3}:
            self.status_var.set(f"{source}：识别 {radiant_count}v{dire_count}{confidence_text}，第 {inferred} 轮")
            self.generate_recommendations(silent=True)
        else:
            self.status_var.set(f"{source}：识别 {radiant_count}v{dire_count}{confidence_text}；请手动修正空缺或指定轮次")

    def _team_count(self, side: str) -> int:
        return sum(hero_id is not None for hero_id in self.team_ids[side])

    def _heroes(self, side: str) -> tuple[int, ...]:
        return tuple(hero_id for hero_id in self.team_ids[side] if hero_id is not None)

    @staticmethod
    def _infer_phase(radiant: int, dire: int, *, allow_complete: bool = False) -> int | None:
        if radiant == dire == 0:
            return 1
        if radiant == dire == 2:
            return 2
        if radiant == dire == 4:
            return 3
        if allow_complete and radiant == dire == 5:
            return 4
        return None

    def _selected_phase(self) -> int:
        if self.phase_var.get() != "自动":
            return int(self.phase_var.get())
        phase = self._infer_phase(self._team_count("radiant"), self._team_count("dire"))
        if phase is None:
            raise ValueError("自动轮次要求天辉和夜魇的英雄数同为 0、2 或 4")
        return phase

    def _blend_for(self, phase: int) -> float:
        defaults = {1: 0.0, 2: 0.1, 3: 0.1}
        report = self.model_bundle.backtest
        if not report:
            return defaults[phase]
        return float(report.get("selected_value_blend", {}).get(f"phase_{phase}", defaults[phase]))

    def _maybe_recommend(self) -> None:
        if self.phase_var.get() != "自动" or self._infer_phase(self._team_count("radiant"), self._team_count("dire")):
            self.generate_recommendations(silent=True)

    def _fill_tree(self, side: str, recommendations: list[object]) -> None:
        tree = self.trees[side]
        for item in tree.get_children():
            tree.delete(item)
        for recommendation in recommendations:
            info = self.catalog.info(recommendation.hero_id)
            tree.insert(
                "",
                "end",
                image=self._hero_thumbnail(recommendation.hero_id),
                values=(
                    recommendation.rank,
                    info.display_name,
                    f"{recommendation.policy_probability:.1%}",
                    f"{recommendation.value_log_odds_delta:+.3f}",
                ),
            )

    def _clear_results(self, label: str = "等待有效BP状态") -> None:
        for side, tree in self.trees.items():
            for item in tree.get_children():
                tree.delete(item)
            self.result_titles[side].set(("天辉" if side == "radiant" else "夜魇") + f"：{label}")

    def generate_recommendations(self, silent: bool = False) -> None:
        try:
            radiant = self._heroes("radiant")
            dire = self._heroes("dire")
            if len(set(radiant + dire)) != len(radiant + dire):
                raise ValueError("同一英雄不能同时出现在两个阵营")
            phase = self._selected_phase()
            blend = self._blend_for(phase)
            radiant_recs, radiant_kind = self.recommender.recommend(
                DraftState(phase=phase, allies=radiant, enemies=dire), top_k=10, value_blend=blend
            )
            dire_recs, dire_kind = self.recommender.recommend(
                DraftState(phase=phase, allies=dire, enemies=radiant), top_k=10, value_blend=blend
            )
            self._fill_tree("radiant", radiant_recs)
            self._fill_tree("dire", dire_recs)
            self.result_titles["radiant"].set("天辉下一手推荐 · 夜魇需防范")
            self.result_titles["dire"].set("夜魇下一手推荐 · 天辉需防范")
            self.status_var.set(
                f"第 {phase} 轮双向推荐完成 · 天辉 {radiant_kind} / 夜魇 {dire_kind} · "
                f"Value 权重 {blend:g} · {self.model_bundle.rank_bracket_label}模型 "
                f"{self.model_patch}"
            )
        except Exception as exc:
            self.status_var.set(f"无法推荐：{exc}")
            if not silent:
                messagebox.showerror("无法生成推荐", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dota 2 BP Helper desktop MVP")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    DraftDesktopApp(root)
    if args.smoke_test:
        root.after(350, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
