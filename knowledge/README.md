# 冒险岛（MapleStory）知识库

数据源：**maplestorywiki.net**（唯一指定来源）。通过「本机可见浏览器 + Chrome DevTools 协议（CDP）」在用户完成 Cloudflare 人机验证后抓取。

## 目录结构

```
knowledge/
  README.md                # 知识库总览与维护约定
  boss/                    # BOSS 攻略与数据（塞伦_Seren.md 等，按 BOSS 一篇）
  job_skill/               # 职业技能知识（meta.md 索引 + 每职业一篇，共 53 个可玩职业）
  patch_note/              # 版本更新公告知识（meta.md 索引 + 每版本一篇，如 V271_MapleStoryxFrieren_PatchNotes.md）
  system/                  # 游戏系统机制与规则（伤害公式、装备强化等）
_raw/                      # 抓取中间产物（HTML / 纯文本 / 解析 JSON）
scripts/
  fetch_page.mjs           # 通用 HTTP 抓取（对无 Cloudflare 的站可用）
  cdp_extract.mjs          # 通过 CDP 读取浏览器里已加载的页面
  cdp_navigate.mjs         # 通过 CDP 导航新页面并抓取
  cdp_title.mjs            # 轻量 CDP 导航，仅打印页面标题
  class_skill_config.json  # 职业清单配置（slug / 中文名 / 种族 / 关键词，唯一数据源）
  batch_fetch_skills.mjs   # 批量 CDP 抓取职业技能页 → _raw/wiki_<slug>.html/.txt
  parse_wiki_skills.mjs    # 解析技能页 HTML → 结构化 Markdown（第 4 参 slug 必填）
  batch_parse_skills.mjs   # 批量解析 → knowledge/job_skill/<中文>_<slug>.md
  audit_headings.mjs       # 审计页面标题结构（核对转职章节映射）
  audit_parsed.mjs         # 审计解析 JSON 统计（技能数 / 章节分布 / 异常）
  parse_patch_jobskills.mjs# 从版本公告页 HTML 抽取折叠的职业改动明细（供按需查阅，不入知识库）
  gen_meta.mjs             # 重建 job_skill/meta.md 文件索引
launch_edge_debug.bat      # 用户双击：拉起带调试端口(9333)的独立 Edge
```

## 工作流（针对 maplestorywiki.net）

1. **用户**双击 `launch_edge_debug.bat`，在桌面正常环境拉起独立 Edge（端口 9333、专用配置 `.edge-cdp`）。
2. **用户**在弹窗里完成 Cloudflare 人机验证，直到看到页面内容。
3. 新增/更新单个职业：
   - `node scripts/cdp_navigate.mjs https://maplestorywiki.net/w/<职业>/Skills _raw/wiki_<职业> 9333` 抓取页面；
   - 在 `scripts/class_skill_config.json` 补该职业条目（slug、zh、group/groupZh、tags）；
   - `node scripts/parse_wiki_skills.mjs _raw/wiki_<职业>.html knowledge/job_skill/<中文>_<职业>.md _raw/wiki_<职业>_parsed.json <职业slug>`；
   - `node scripts/gen_meta.mjs` 重建 meta.md 索引。
4. 全量抓取/解析：`node scripts/batch_fetch_skills.mjs 9333` → `node scripts/batch_parse_skills.mjs`（沙箱内子进程输出不能走 pipe，批量脚本用 stdio:inherit + 退出码判断）。

## 工作流（版本更新公告 patch_note/）

1. 用户用调试浏览器打开 Nexon 官方更新公告页（`https://www.nexon.com/maplestory/news/update/<id>/...`）。
2. `node scripts/cdp_read_page.mjs 9333 update <前缀>` 抓取 → `_raw/cdp_page.html` / `_raw/cdp_page.txt`。
3. 基于纯文本撰写 `knowledge/patch_note/V<版本>_<主题>_PatchNotes.md`：front-matter 用 `id: patch-<版本号>`、`type: patch_note`、`version`、`published`、`source`；正文按「版本要点速览 / 新增内容 / 各系统改版 / BOSS / 地图 / 道具 / 各活动（时间+条件+奖励） / 影响速记」分节，**数值与日期一律保留原文精度**。
4. 在 `knowledge/patch_note/meta.md` 的「文件索引」补一条 `- id / title / description / keyword`（keyword 决定检索命中，需含版本号、主题 IP、改动关键词的中英文）。
5. 公告里的**职业技能改动**藏在折叠 `<details>` 块中，`innerText` 抓不到；需要时用 `node scripts/parse_patch_jobskills.mjs _raw/cdp_page.html _raw/patch_jobskills.md` 抽取（默认不进知识库，仅作原始素材）。

## 关键结论（踩坑记录）

- **Edge 151**：`DevToolsActivePort` 文件不再写入；连接通道走 HTTP `/json/version`（返回 200）。
- **沙箱内启动 Edge 会崩溃**（unknown software exception），必须由用户在本机正常环境双击脚本拉起。
- 沙箱内 Node 可正常监听本地端口、可连外网 HTTPS（走自带 OpenSSL），问题只在 GUI 浏览器的进程启动。
- `.edge-cdp` 配置已保存 Cloudflare clearance cookie，后续可能免验证（直到过期）。
- **PowerShell 读写会破坏 UTF-8 文件**（GBK 控制台编码），涉及中文文本一律用 read/write/edit 工具或 Node 脚本处理。
- 章节标题有 3 种风格：`<职业> (0)/(I)~(IV)`（正则映射）、主题式标题（如 Hero 的 Beginner's Basics…，按正文前 5 个 h2 序数映射 0~4 转）、特殊页（Zero 的 Alpha/Beta、Dual Blade 的 6 段转职档，用 config 的 jobOrder/noJobMap 控制）。

## 每篇文档约定

- 顶部标注：来源 URL、抓取时间、版本。
- 技能按 0转 / 1~4转 / Hyper / 5转(V) / 6转(HEXA) 分节，含精通等级、描述、各等级数值。
- 文末含「强化核心(Boost Node)」与「精通核心(Mastery Node)」分组。
- **patch_note/**：front-matter 标注 `version` / `published` / `source`；正文含「版本要点速览」表与「版本影响速记」，活动均标注 UTC 时间窗口、参与条件与奖励/限购；同类改动用表格承载数值前后对比。
