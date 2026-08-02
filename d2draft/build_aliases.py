from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Chinese community names that are not part of Valve's official name table.
# Valve's own pinyin/search aliases are merged below, so this table only needs
# to cover genuinely colloquial names and useful English shorthand.
COMMUNITY_ALIASES: dict[str, tuple[str, ...]] = {
    "antimage": ("敌法", "df", "am"),
    "axe": ("斧王",),
    "bane": ("睡魔",),
    "bloodseeker": ("血魔", "bs"),
    "crystal_maiden": ("冰女", "cm"),
    "drow_ranger": ("小黑", "黑弓", "dr"),
    "earthshaker": ("小牛", "神牛", "es"),
    "juggernaut": ("剑圣", "jugg", "jug", "js"),
    "mirana": ("白虎", "月女", "pom"),
    "morphling": ("水人",),
    "nevermore": ("影魔", "sf"),
    "phantom_lancer": ("猴子", "长矛手", "pl"),
    "puck": ("仙女龙",),
    "pudge": ("屠夫",),
    "razor": ("电魂",),
    "sand_king": ("沙王", "sk"),
    "storm_spirit": ("蓝猫",),
    "sven": ("流浪",),
    "tiny": ("小小", "石头人"),
    "vengefulspirit": ("复仇", "vs"),
    "windrunner": ("风行", "风行者", "wr"),
    "zuus": ("宙斯", "zeus"),
    "kunkka": ("船长",),
    "lina": ("火女",),
    "lion": ("莱恩", "恶魔巫师"),
    "shadow_shaman": ("小y", "小歪", "萨满"),
    "slardar": ("大鱼", "大鱼人"),
    "tidehunter": ("潮汐", "西瓜皮"),
    "witch_doctor": ("巫医", "51", "wd"),
    "lich": ("巫妖",),
    "riki": ("隐刺", "sa"),
    "enigma": ("谜团",),
    "tinker": ("修补",),
    "sniper": ("火枪", "矮子"),
    "necrolyte": ("死灵法", "nec"),
    "warlock": ("术士",),
    "beastmaster": ("兽王", "bm"),
    "queenofpain": ("女王", "qop"),
    "venomancer": ("剧毒",),
    "faceless_void": ("虚空", "fv"),
    "skeleton_king": ("骷髅王", "wk"),
    "death_prophet": ("死亡先知", "dp"),
    "phantom_assassin": ("幻刺", "pa"),
    "pugna": ("骨法",),
    "templar_assassin": ("圣堂", "ta"),
    "viper": ("毒龙",),
    "luna": ("月骑",),
    "dragon_knight": ("龙骑", "dk"),
    "dazzle": ("暗牧",),
    "rattletrap": ("发条", "clock"),
    "leshrac": ("老鹿",),
    "furion": ("先知", "np"),
    "life_stealer": ("小狗", "ls"),
    "dark_seer": ("兔子", "ds"),
    "clinkz": ("小骷髅", "骨弓"),
    "omniknight": ("全能", "omni"),
    "enchantress": ("小鹿",),
    "huskar": ("神灵",),
    "night_stalker": ("夜魔",),
    "broodmother": ("蜘蛛",),
    "bounty_hunter": ("赏金", "bh"),
    "weaver": ("蚂蚁",),
    "jakiro": ("双头龙",),
    "batrider": ("蝙蝠",),
    "chen": ("陈",),
    "spectre": ("幽鬼",),
    "doom_bringer": ("末日", "doom"),
    "ancient_apparition": ("冰魂", "aa"),
    "ursa": ("拍拍", "拍拍熊"),
    "spirit_breaker": ("白牛", "sb"),
    "gyrocopter": ("飞机",),
    "alchemist": ("炼金",),
    "invoker": ("卡尔",),
    "silencer": ("沉默",),
    "obsidian_destroyer": ("黑鸟", "od"),
    "lycan": ("狼人",),
    "brewmaster": ("熊猫",),
    "shadow_demon": ("毒狗", "sd"),
    "lone_druid": ("德鲁伊", "熊德", "ld"),
    "chaos_knight": ("混沌", "ck"),
    "meepo": ("地卜",),
    "treant": ("大树",),
    "ogre_magi": ("蓝胖",),
    "undying": ("尸王",),
    "rubick": ("拉比克",),
    "disruptor": ("萨尔",),
    "nyx_assassin": ("小强", "na"),
    "naga_siren": ("小娜迦", "娜迦"),
    "keeper_of_the_light": ("光法", "kotl"),
    "wisp": ("小精灵", "io"),
    "visage": ("死灵龙",),
    "slark": ("小鱼", "小鱼人"),
    "medusa": ("一姐", "美杜莎"),
    "troll_warlord": ("巨魔",),
    "centaur": ("人马",),
    "magnataur": ("猛犸",),
    "shredder": ("伐木机",),
    "bristleback": ("刚背",),
    "tusk": ("海民",),
    "skywrath_mage": ("天怒",),
    "abaddon": ("亚巴顿",),
    "elder_titan": ("大牛", "et"),
    "legion_commander": ("军团", "lc"),
    "techies": ("炸弹", "炸弹人"),
    "ember_spirit": ("火猫",),
    "earth_spirit": ("土猫",),
    "abyssal_underlord": ("大屁股",),
    "terrorblade": ("恐怖利刃", "tb"),
    "phoenix": ("凤凰",),
    "oracle": ("神谕",),
    "winter_wyvern": ("冰龙", "ww"),
    "arc_warden": ("电狗", "arc"),
    "monkey_king": ("大圣", "mk"),
    "dark_willow": ("小仙女", "花仙子"),
    "pangolier": ("滚滚", "穿山甲"),
    "grimstroke": ("墨客",),
    "hoodwink": ("松鼠",),
    "void_spirit": ("紫猫",),
    "snapfire": ("老奶奶", "奶奶"),
    "mars": ("马尔斯",),
    "dawnbreaker": ("破晓", "锤妹"),
    "marci": ("玛西",),
    "primal_beast": ("兽", "原始兽"),
    "muerta": ("穆尔塔",),
    "kez": ("凯", "鸟武士"),
    "largo": ("朗戈", "青蛙", "蛤蟆"),
}


def _tokens(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8-sig")
    return dict(re.findall(r'^\s*"([^"]+)"\s+"([^"]*)"', text, re.MULTILINE))


def build() -> dict[str, dict[str, object]]:
    heroes = json.loads((ROOT / "data" / "heroes.json").read_text(encoding="utf-8-sig"))
    localization = ROOT / "data" / "localization" / "resource" / "localization"
    abilities = _tokens(localization / "abilities_schinese.txt")
    dota = _tokens(localization / "dota_schinese.txt")
    result: dict[str, dict[str, object]] = {}
    for raw in heroes.values():
        hero_id = str(raw["id"])
        internal = str(raw["name"])
        short = internal.removeprefix("npc_dota_hero_")
        chinese = abilities.get(f"{internal}:n", "")
        valve = tuple(filter(None, dota.get(f"{internal}__name_alias", "").split(";")))
        english = str(raw["localized_name"])
        words = re.findall(r"[a-z0-9]+", english.casefold())
        initialism = "".join(word[0] for word in words) if len(words) > 1 else ""
        aliases = {
            *valve,
            *COMMUNITY_ALIASES.get(short, ()),
            *((initialism,) if initialism else ()),
        }
        aliases.discard("")
        aliases.discard(chinese)
        result[hero_id] = {
            "chinese_name": chinese,
            "aliases": sorted(aliases, key=lambda value: (len(value), value.casefold())),
        }
    return result


def main() -> None:
    output = ROOT / "data" / "hero_aliases_zh.json"
    result = build()
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    missing = [hero_id for hero_id, value in result.items() if not value["chinese_name"]]
    print(f"wrote {len(result)} heroes to {output}")
    print(f"missing official Chinese names: {missing}")


if __name__ == "__main__":
    main()
