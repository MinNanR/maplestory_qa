---
id: system-meta
title: 游戏系统机制与规则
type: folder_meta
---

# 游戏系统机制与规则

本文件夹收录冒险岛（MapleStory）游戏系统层面的机制与规则文档：伤害计算公式（属性体系 Pure/Added/Percentage/Final、武器倍率、熟练度、伤害%/BOSS 伤/最终伤害/无视防御、暴击率与暴击伤害、伤害范围、等级优势倍率、星力/神秘之力/真实之力地图修正、普通/BOSS/DOT/灵魂武器最终输出公式）、装备强化系统（星之力 Star Force：星级上限、成功率/失败率/破坏率、属性成长、强化费用与折扣、保护与破坏恢复、强化活动、版本改动时间线、GMS 差异）、附加选项/火花系统（Bonus Stats / Flames / Rebirth Flame：GMS 基础规则与档位体系、重生之焰家族、传承卷轴、各类数值范围与公式、强化指南、版本时间线、KMS/GMS 差异）与 Utility 相关机制（异常状态抗性、击退抗性、攻击速度、绑定/Buff/召唤时长、技能冷却与重置、防御与怪物伤害、闪避率、移动能力、死亡惩罚）。可回答伤害如何计算、属性如何堆叠、强化概率与费用、装备破坏与恢复、火花/附加选项如何评估、异常抗性/攻速/冷却/防御/闪避等 Utility 机制如何运作等系统性问题。

## 文件索引

- id: upgrade-rule
  title: 强化规则
  description: 装备强化速查手册：各装备等级区间 15~30 星主属性/攻击力累计增幅表、12~29 星强化费用表（按 140/150/160/200/250 级）、15~17 星与 18~21 星装备保护规则及费用倍率（3 倍 / 6.5 倍）、七折与降炸等强化活动说明
  keyword: 强化, 星之力, 装备, 属性增幅, 强化费用, 保护, 成功率, 活动, 七折, 降炸
- id: damage-formulas
  title: 伤害计算公式（全量）
  description: 冒险岛伤害计算公式完整推导：属性体系（基础/追加/百分比/最终属性）、武器倍率表、熟练度、Damage%/BOSS 伤害%/最终伤害/无视防御（乘法叠加规则）、暴击率与暴击伤害、伤害范围计算、等级优势倍率、星力/神秘之力/真实之力地图修正表、普通攻击/BOSS/DOT/灵魂武器召唤最终输出公式、加成类型速记
  keyword: 伤害, 公式, 属性, 武器倍率, 熟练度, 暴击, 暴击伤害, 无视防御, 最终伤害, BOSS伤, 地图力, 星力, 神秘之力, 真实之力
- id: starforce
  title: 星之力强化（装备强化 / Star Force Enhancement）
  description: 星之力强化系统详解（KMS 2026 / 30 星版本，附 GMS 差异）：星级上限（普通/卓越装备）、0~30 星成功率/失败率/破坏率全表、卓越装备（Superior）概率与费用、属性成长、强化费用公式与折扣（MVP/PC 房/周日枫叶）、破坏与恢复（传统 12 星恢复 / 确定恢复、备件数与金币表、特殊武器）、星之力猎场伤害表、停靠星级建议、周日枫叶活动、版本改动时间线、成就
  keyword: 星之力, Star Force, 强化, 装备, 成功率, 破坏, 恢复, 卓越装备, Superior, 周日枫叶, 折扣, 30星, 星之力猎场, GMS
- id: flame-additional-options
  title: 附加选项 / 火花（Bonus Stats / Additional Options / Rebirth Flame）
  description: 附加选项（火花）系统全解（GMS 视角为主，MapleStory Wiki；附 KMS NamuWiki 差异）：GMS 基础规则（7 档 tier 体系、最多 4 条不重复、Flame Advantaged/boss flames 与例外、无法附加火花的装备清单）、Rebirth Flame 家族（Powerful/Eternal/Black/Abyssal 各档位范围与 Karma 变体、英雄服 950 万 meso 直购等获取渠道）、KMS 生成逻辑与概率表、附加选项传承卷轴、各类附加选项数值公式（武器攻/魔、单双属性、全属性%、MaxHP/MP）、防具/首饰与武器 추 档评估与分资本目标、历史时间线、KMS/GMS 术语对照表；关键差异要点：用 Meso 直接重置附加选项为 KMS 2026-03-19 专属改动，GMS 未实装、仍只能使用火花道具
  keyword: 附加选项, 火花, Flame, Bonus Stats, Rebirth Flame, 重生之焰, Powerful, Eternal, Black, Abyssal, Karma, bonus stats, 附加潜能, 추옵, chuop, 1추, 阶数, tier, Flame Advantaged, boss flames, 剪刀次数, 传承卷轴, Transfer Scroll, 全属性, BOSS伤害, 보뎀, Meso重置, 概率操纵, 英雄服
- id: utility-related
  title: Utility 相关机制（抗性 / 攻速 / 冷却 / 防御 / 闪避 / 移动 / 死亡惩罚）
  description: StrategyWiki Formulas 页 "Utility Related" 一节整理：异常状态抗性（28×log10(抗性)+1 公式与来源）、击退抗性 Stance（含替换机制）、攻击速度（武器量级、软/硬上限、延迟与频率公式及完整换算表）、绑定技能时长（1% HP = +10% 延长、90 秒抗性、Origin 绝对绑定）、Buff/召唤物时长公式、技能冷却（CDR% 与潜能 CDR 折算规则、5 秒下限、冷却加速/重置）、防御公式、怪物对你伤害的公式（A/B 等级差表、星力/神秘/真实之力地图怪物伤害倍率表、怪物攻击减免、元素抗性）、闪避率公式（90% 上限）、移动速度与跳跃上限、死亡惩罚（0.2 倍经验/掉落、按等级时长表、减免手段）
  keyword: Utility, 异常状态抗性, 状态抗性, 击退抗性, Stance, 攻击速度, Attack Speed, 攻速, Bonus Attack Speed, 施放延迟, 绑定, Bind, 绑定时长, Origin, 绝对绑定, Buff时长, 召唤时长, 技能冷却, Cooldown, CDR, 冷却重置, 冷却加速, 防御, Defense, 怪物伤害, 怪物攻击, A值, B值, 星力地图, 神秘之力地图, 真实之力地图, 闪避, Dodge, 回避, 移动速度, 跳跃, 死亡惩罚, 死降, Death Debuff
