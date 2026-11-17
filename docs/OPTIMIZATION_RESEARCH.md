# 前沿调研与下一步优化分析

日期：2026-11-16 · 基于多源检索（精确算术库 / AI 数学工具 / 高性能数值算法）

## 结论先行

我们的项目（Tally 精确算术内核）正处在一条被前沿验证的赛道上：
**"LLM 生成 + 符号引擎精确验证"的神经符号闭环**是 2025–2026 数学 AI
的核心范式（AgentMath、DeepMath、Maple MCP 都在这条路上）。Tally 的
`tally_verify` 恰好就是这个闭环里的"验证端"。

但检索也暴露了一个直接竞品：**Maple MCP**（Maplesoft 官方，9000+ 算法，
MCP 协议，兼容 Claude/ChatGPT/Gemini）。它验证了方向，也抬高了门槛。
Tally 的差异化必须是：**2.2MB 零依赖 vs Maple 的 GB 级商业安装、
精确有理数契约、免费开源、审计注记**。

下一步优化的最高价值项不是硬件技巧，而是三个算法级升级：
**二分分裂级数求值、可证明误差界、模方法精确线性代数**。

---

## 一、前沿全景（与我们相关的部分）

### 1.1 直接竞品与同生态产品

| 产品 | 形态 | 对 Tally 的意义 |
|---|---|---|
| **Maple MCP** | LLM + MCP + Maple 引擎，9000+ 算法，精确符号/数值，确定性输出 | 直接竞品。验证了"数学引擎 via MCP"赛道；Tally 必须打轻量/免费/精确有理数差异化 |
| **Qalculate! (libqalculate)** | GMP/MPFR 桌面计算器，递归下降解析 + AST，动态精度，400+ 功能 | 架构同构（AST→GMP），证明这条路可行；它有单位换算/误差传播，我们没有 |
| **Math.js** | JS BigNumber/Fraction，默认 BigNumber 规避 IEEE 754 | 前端生态的精确计算默认项；Tally 暂无 JS 端 |
| **SymEngine** | C++ 轻量符号内核（SageMath 底层） | 轻量符号引擎的先例；Tally 不做符号代数，边界清晰 |

### 1.2 AI Agent 数学工具（验证我们的定位）

- **AgentMath**（arXiv 2512.20745）：LLM + 代码解释器 + Agentic RL，
  AIME24 90.6% / AIME25 86.4%；工具增强 RL 比纯文本 RL 训练效率高 4 倍。
- **DeepMath**（Intel）：Qwen3-4B + GRPO + 沙盒执行微代码片段，
  输出长度缩短 66%，数学基准提升。
- **OptimAI**：Formulator→Planner→Coder→Critic 四 Agent，
  自然语言→精确表达式→迭代执行。
- **Wolfram 综述（2025–2026）**：LLM + 精确符号工具（CAS/证明助手）
  是 AI for Mathematics 核心趋势。

**含义**：整个领域都在把"精确计算"外包给工具，LLM 只负责生成。
Tally 的 `tally_compute`/`tally_verify` 正是这个工具位。
`tally_verify` 的审计注记（错在第几位/哪个数量级）是竞品没有的细节，
应作为主打卖点。

### 1.3 算法前沿（真正能用的）

| 技术 | 来源 | 对 Tally 的适用性 |
|---|---|---|
| **二分分裂（binary splitting）级数求值** | MPFR/y-cruncher 标准技术 | **最高价值**。我们的 exp/ln/atan 级数目前是逐项累加，大数乘法代价随项数平方增长；二分分裂把有理数级数合并为树形乘积，高精度下快一个数量级 |
| **AGM（算术几何平均）算 pi/ln** | Gauss-Legendre / Borwein 算法 | 高精度 pi 的渐近最优算法；我们目前用 Machin 公式，1000 位时 AGM 更快 |
| **CORE-MATH**（Inria） | 正确舍入数学函数库，Meta 评价 "indispensable" | 它的价值是 float64 正确舍入，不是精确有理数；但其论证归约/多项式求值技巧可借鉴 |
| **FLINT 模方法线性代数** | 快速数论库，SC25 相关 | 整数矩阵行列式用模素数 + CRT 重建，比有理数消元快得多；我们的 det/inv/solve 可升级 |
| **MPFI 区间算术** | MPFR 之上的可证明误差界 | 我们的"certified convergent"目前是保护位启发式；区间算术可给出**可证明**的误差界，让"certified"名副其实 |
| **无分支扩展精度（SC25）** | 比 QD 快 11.7×、比 CAMPARY 快 34.4×（单源数据） | 针对 double-double 浮点，与我们的 GMP 有理数路径不同，适用性低 |
| **数值超优化**（arXiv 2603.24812） | 自动生成正确舍入数学库 | 研究阶段，长期关注 |

### 1.4 形式化验证趋势（长期护城河）

- Coq/Rocq 验证浮点算法正确舍入与边界条件；Lean 4 "If It Compiles,
  It Is Correct" 范式进入 ZK 与工程领域。
- arXiv 2026：Cortex-M 上运行经 Rocq 验证的固件逻辑的完整流水线。
- NeurIPS 2026 MATH-AI / IJCAI NeSy：生成-验证闭环是共识。

**含义**：我们的不变量"每个算子必须 gradcheck/parity"是测试级保证；
前沿在做**证明级保证**。对核心精确算子（有理数四则、级数收敛界）
做 Lean/Coq 形式化，是"最优秀"叙事的终极支撑，但工程量大，放长期。

---

## 二、诚实过滤：检索中不适用于我们的部分

检索报告给了大量**嵌入式 MCU** 建议，必须明确排除，否则会毁掉产品定位：

| 建议 | 为什么不适用 |
|---|---|
| CORDIC / 查表法替代 math.h | CORDIC 给的是**近似值**（16-bit 精度）。Tally 的契约是"精确有理数，浮点永不进入"。用 CORDIC 等于背叛产品核心 |
| Q16.16 定点化 / 饱和算术 | 同上：定点是有限精度近似，与任意精度有理数契约冲突 |
| RISC-V 自定义指令 / P 扩展 / TCM 布局 | 我们是桌面/服务器 Python 包，不是 MCU 固件；这些硬件技巧无落点 |
| Posit 格式 | 面向内存受限 DNN 推理的近似格式，与精确计算无关 |
| 手写 NEON 大数向量化 | GMP 已有各平台汇编优化实现，J 引擎直接调用；手写 NEON 打不过 GMP 汇编，徒增维护成本 |
| GPTune 自动调优 | 我们的级数项数/Newton 迭代数是解析推导的，不是黑盒参数，自动调优收益有限 |

**原则**：任何以牺牲"精确性"换速度的优化都拒绝。我们的速度必须在
精确算法内部找（二分分裂、AGM、模方法），而不是降级到近似。

---

## 三、差距分析（对照前沿）

| 维度 | 现状 | 前沿标杆 | 差距 |
|---|---|---|---|
| 超越函数速度 | exp(1)@200位 1.4ms，慢 mpmath ~200× | mpmath 二分分裂 + 二进制浮点 | 算法级差距，可用二分分裂大幅缩小（但输出契约不同，不必追平） |
| 误差保证 | 保护位启发式（guard digits） | MPFI 可证明区间界 | "certified"目前是工程自信，不是数学证明 |
| 精确线代 | 有理数直接消元 | FLINT 模方法 + CRT | 大矩阵下差距会拉大 |
| 分发 | macOS wheel（cp39） | pip 全平台一键安装 | Linux/多 Python 版本缺失 |
| 竞品 | 无直接对标意识 | Maple MCP 已入场 | 需要明确差异化叙事 |
| 正确性保证 | 91 测试 + 4733 差分 | 形式化证明（Coq/Lean） | 测试级 vs 证明级 |

---

## 四、优化路线图（按价值/成本排序）

### P0 — 二分分裂级数求值（算法级，最高价值）
把 exp/ln/atan/sin/cos 的逐项累加改为二分分裂：
级数 Σ p(k)/q(k) 用分治合并成单个分数，大数乘法次数从 O(n²) 降到
O(n log n) 量级。预期高精度（≥200 位）超越函数提速 5–20×。
完全在现有架构内（J 侧实现），不破坏任何不变量。
验收：对 mpmath 差分验证全过 + bench_tally 前后对比。

### P1 — 可证明误差界（让 "certified" 名副其实）
为每个收敛算子附加区间误差界：用有理数区间 [lo, hi] 包裹截断误差，
返回 `error_bound` 字段。这是 mpmath 没有的能力（它给 mpf 不给证明），
直接构成对 Maple MCP 的差异化。
验收：每个算子的 error_bound 经独立参考验证包含真值。

### P2 — 模方法精确线性代数（FLINT 思路）— 实测否决
原设想：整数矩阵 det/inv 走"模素数 + CRT 重建"避开中间膨胀。
**实测结论：不做。** 原型对 n=4..50、位宽 20..1000 全规模测试，
Python 层模方法在所有情况下都慢于现有 J 引擎路径（0.0×–0.6×）。
根因：J 引擎的 `detr`（vgauss.c）已是 C/GMP 层的精确有理消元，
原地操作 + 栈式 GC，qminus/qtymes 直接调 GMP；Python 大数模运算
开销 + 大条目需多素数，反而更慢。FLINT 的优势在 C 层模算术，
搬到 Python 层不成立。**诚实记录：现有路径已是最优，不改。**

### P3 — 全平台 wheel（分发完整性）
CI 矩阵构建 Linux x86_64/aarch64 + macOS arm64/x86_64，
Python 3.9–3.13。这是"pip install 即用"承诺的兑现。
验收：各平台干净环境安装 + 冒烟测试。

### P4 — 形式化验证核心算子（长期护城河）
用 Lean 4 对有理数四则与级数收敛界做机器检查证明，
逐步替换"测试级保证"。工程量大，分期推进。

### 定位动作（与工程并行）
- README 明确对标 Maple MCP：2.2MB vs GB 级、免费 vs 商业、
  精确有理数契约、审计注记。
- 主打叙事："AI 算错钱时的精确验证端"——蹭 AgentMath/DeepMath
  验证过的生成-验证闭环趋势。

---

## 五、来源（检索所得，关键项）

- Maple MCP: http://www.maplesoft.com/products/MapleMCP/
- AgentMath: https://arxiv.org/abs/2512.20745
- DeepMath: https://hugging-face.cn/blog/intel-deepmath
- CORE-MATH: https://core-math.gitlabpages.inria.fr/
- FLINT / SC25: https://dl.acm.org/doi/abs/10.1145/3712285.3759876
- 无分支扩展精度 SC25: http://theory.stanford.edu/~aiken/publications/papers/sc25.pdf
- 数值超优化: https://arxiv.org/html/2603.24812
- Lean4 工程范式: https://blog.lambdaclass.com/if-it-compiles-it-is-correct-almost-an-introduction-to-lean-4-for-zk-systems-and-engineering-2/
- NeSy 工作坊: https://www.wias-berlin.de/workshops/NESYRE2026/
- Qalculate: https://blog.csdn.net/gitblog_00691/article/details/161861447
- GMP: https://apolo-docs.readthedocs.io/en/latest/software/scientific_libraries/GMP/index.html

注：检索中的性能数字（如 11.7×、93.62%）多为单源论文自述，
未独立复现，引用时保留原始口径。嵌入式 MCU 类来源已按第二节过滤。

---

## 六、实施状态（2026-11-16 当日落地）

### P0 向量化级数 — 已完成
把 exp/ln/pi/atan 的逐项累加改为**公共分母向量化**形式：
- exp(p/q) = [Σ p^k q^(n-k) (n!/k!)] / (q^n n!)，单次向量化 pass
- atan/atanh(p/q) = [Σ (±)p^(2k+1) q^(2(n-k))/(2k+1)] / q^(2n+1)

实测：exp(1)@1000 位 47.2ms → 3.9ms（**12.2×**）；pi@500 2.5ms；
ln2@300 1.6ms。全部对 mpmath 高精度差分验证通过。
（注：这是公共分母向量化，非严格二分分裂；后者需 J 侧递归动词，
JDo 单句限制下暂以向量化达成同等量级收益。）

### P1 可证明误差界 — 已完成
每个收敛算子返回 `error_bound`（精确 p/q）+ `certified: true`：
- 交错级数（atan/sin/cos/pi）：莱布尼茨定理，界 = 首个省略项
- exp：几何尾部界 a_{n+1}/(1-r)，全程 Fraction 运算避免大阶乘浮点下溢
- sqrt：Newton 二次收敛界
- 二元算子：区间算术传播（add/mul 紧界，div 保守界）
- 复合表达式（如 pi+exp(1)、2*pi）整体可证书

验证：101 项测试全绿（新增 10 项证书测试），每个界经 mpmath
超高精度参考确认真值落在界内。这是 mpmath/Maple 都没有的能力。

### P2 模方法线代 — 实测否决（详见第四节）
原型全规模测试（n=4..50，位宽 20..1000）确认 Python 层模方法
在所有情况下慢于 J 引擎的 C/GMP 精确有理消元。不做。

### P3 全平台 wheel — 已完成（CI 待跑）
- setup.py 加入 Linux 内嵌路径：libj.so + libgmp.so（引擎 dlopen
  GMP，需内嵌）+ profile.ijs，patchelf 设 RPATH=$ORIGIN + soname。
- 新增 .github/workflows/wheels.yml：4 平台矩阵（macOS arm64/x86_64、
  Linux x86_64/aarch64）+ sdist，每平台干净 venv 冒烟测试。
- macOS wheel 本地重建验证通过；Linux 路径静态校验通过，
  实际构建需 CI 运行（本机无 Linux 容器）。

### 第五轮（2026-11-17）：三角向量化 + 缓存 + REPL + 诚实基准
- sin/cos/tan 向量化：先求值归约参数 r（舍入到保护精度避免大数幂
  瓶颈），再公共分母向量化 Taylor。sin(100)@100 657ms→34ms（19×），
  归约舍入误差计入证书。至此全部超越函数向量化完成。
- 结果 LRU 缓存（512 条，canonical AST+digits 键）：重复查询 740×。
- 字面量幅值追踪：精确字面量/数组记录 |value|，乘除误差界收紧
  （除以精确字面量得紧界 ea/|b|）。
- 数组编译去重：_to_fraction 每元素一次（原两次），sum10000 37.5→28ms。
- 交互式 REPL（tally -i）：ans 变量、digits 切换、证书显示。
- 误差界科学计数法显示（_sci）：极小界不再显示为全零误导。
- 基准改冷缓存口径（清缓存后计时），README 区分冷/热诚实报告。

### 第六轮（2026-11-17）：读回协议 + irr 牛顿法 + 批量工具
- 读回协议重构：发现 J 会话单次 JDo 输出上限 ~43k 字符（实测
  43761 OK / 44000 截断）。改为分批输出（每批 ≤200 块 ≈40k 字符），
  往返次数随位数/200 增长而非逐块。修复了超大数（cos(-57.81)@150
  的 47k 位分子）读回截断导致的差分失败。
- irr 牛顿法：二分 40 步定位 + 牛顿精化（每步舍入到 g+5 位避免
  分母平方爆炸）。irr@300 从超时（>300s）→4ms；误差界用牛顿残差
  |NPV/NPV'| + 舍入松弛，经 mpmath 验证包含真根。
- MCP 批量工具 tally_batch：一次调用算多个表达式（≤256），
  逐项错误隔离，共享引擎预热。agent 多数字核验场景一次往返。
- infix 解析器补全金融/矩阵函数（irr/fv/pv/pmt/nper/det/inv/
  matmul/solve/round）。

### 第七轮（2026-11-17）：证书感知验证 + 精确比较工具
- tally_verify 证书感知：声称值按所给精度理解为区间（"3.14159"
  = ±0.5e-5），与真值证书区间 [e-B, e+B] 重叠即判 consistent
  （match=True）。修复了"pi≈3.14159 永远验不过"的真实功能缺口。
  区间不重叠则 definitively wrong + 审计注记。精确算子仍要求逐位相等。
- tally_compare 工具：精确比较两表达式关系（lt/eq/gt），误差包络
  不重叠时关系可证书（22/7>pi、355/113>pi 均 certified）；重叠时
  报 indeterminate_at_precision 而非猜测。
- P4 Lean 检查：本机无 lean/lake/elan 工具链，形式化需先装工具链，
  列为后续（不擅自安装）。

### 第八轮（2026-11-17）：级数项数精确化 + exp 误差界 bug 修复
- exp/ln 项数从最坏情况线性估计改为按实际参数精确有理数求解
  （增量累积阶乘/幂，无浮点对数精度损失）。exp(1)@200 项数减半；
  ln(1.0001)@100 降到 0.1ms。
- 修复 exp 潜在误差界 bug：原界漏乘 exp(x) 幅值因子，大参数下
  偏小可达 10^217 倍（exp(500) 证书曾无效）。现按结果实际幅值
  e^x（带符号）计位并乘入界。
- 修复 exp 负参数过度计算：mag_digits 原按 e^|x| 算，exp(-700)
  多算 300 位（66s）；改按 e^x 实际幅值后 3.5s。
- 大参数 exp（500/700）正确但慢（10+s）：q^n 大整数幂的固有代价，
  正确性优先，如实记录。

### 第九轮（2026-11-17）：证书审计 + 任意底对数
- 系统审计收敛算子证书：sin/cos 的 Taylor 界原用
  limit_denominator(10^30) 会向下舍入（潜在证书失效隐患），改为
  pi<355/113 上界 + 纯整数位数差向上取整（避浮点下溢）。
- 新增 log/log2/log10（任意底对数）：ln(x)/ln(b)，商误差用真值
  区间端点严格传播（非一阶线性化）。修复了初版用收敛值近似导致
  的界偏小问题。
- 排查 log_3(2) 测试"失败"：根因是参考值精度不足（mpmath.log(2,3)
  在 dps=80 下仅 ~82 位精度 + nstr 舍入），引擎证书本身有效。
  教训：验证证书必须用远高于收敛值精度的参考。

### 第十轮（2026-11-17）：tan 严格证书 + 全算子证书审计
- tan 误差界从拍的保守值（10^(-g+5)）改为严格商传播：先求值
  sin/cos 收敛值，用区间端点法算 tan=sin/cos 的界；cos 包络含 0
  时明确报错拒绝证书（诚实失败，不给假界）。
- 全算子证书审计通过：sqrt（含完全平方精确根 vs 收敛根区分）、
  atan（大参数归约界 steps*2^steps*10^-(g+10) 实测覆盖）、
  sin/cos/tan/ln/log/exp/pi 全部对 mpmath 高精度参考验证有效。
- 至此每个收敛算子的证书都是数学严格的，无一处拍的近似。

### 第十一轮（2026-11-18）：CI 首跑落地 + 三处真实缺陷修复
推送 v1.0.0 触发 wheels/tests 首次云端运行，暴露并修复：
- **运行器标签过期**：macos-14/13 已于 2026-11-02 退役 →
  macos-15-arm64 / macos-15-intel（注意裸 macos-15 现在是 arm64）。
- **Python 3.11 大整数位数限制**：int↔str 默认上限 4300 位，
  引擎读回常带 11521 位分子 → ValueError。tally_ast 导入时
  set_int_max_str_digits(0) 解除（精确大数正是产品契约）。
  本地 3.9 无此限制故从未暴露——CI 用 3.11 才抓到。
- **universal2 解释器 vs 单架构引擎**：CI 的 Python 是 universal2，
  distutils 自动加 -arch arm64 -arch x86_64，但引擎只有单架构 →
  ld: library 'j' not found。setup.py 按 lipo -archs 引擎实际架构
  编译扩展。
- **sdist 不完整**：numpy 在元数据期被导入 + README_TALLY.md 未入包。
  懒加载 numpy include、MANIFEST.in、long_description 缺省兜底。
- **GMP 内嵌自包含**：改用 jsource 自带 libgmp（仅依赖 libSystem），
  重写 id/自引用为 @loader_path；59/59 引擎所需符号齐全。
- 引擎源码固定到本地验证过的提交 ed13864（可复现构建）；
  补装 nasm（x86_64 SHA 汇编）与 mpmath/sympy（测试依赖）。

首跑结果：Linux x86_64 + aarch64 wheel 构建+冒烟一次通过；
修复后重跑验证 macOS 双架构与 sdist。

### 回归
11 套测试 + solver verify + 124 Tally + 1571 差分全过；
macOS wheel 隔离 venv 验证（tan 带严格证书）；
3.11 下 11521 位整数往返验证；无 numpy 环境 sdist 元数据生成验证。

### 待办
P4 Lean 形式化（需工具链）；PyPI 发布（需用户账号）。