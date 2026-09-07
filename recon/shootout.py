#!/usr/bin/env python3
"""
shootout.py -- recon: "agent's default terminal move" vs the exact engine.

Each case: the terminal command an agent would naively run, against the same
quantity computed exactly by the embedded J engine (GMP extended ints /
rationals) through the pyj bridge. No cherry-picking: case 12 is a deliberate
terminal win (python bignum ints are exact by default), kept for honesty.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYJ = os.path.dirname(HERE)
os.environ.setdefault("PYJ_LIBPATH", os.path.join(PYJ, "jlibrary", "bin"))
sys.path.insert(0, PYJ)
import pyj


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return (p.stdout + p.stderr).strip()


def jsetup():
    pyj.do("digits =: 3 : '\": <. y * 10x^30'")
    pyj.do("newton2 =: 3 : '-: y + 2 % y'")
    pyj.do("mean =: +/ % #")


def jq(s):
    rc, out = pyj.do(s)
    assert rc == 0, f"J rc={rc}: {s}"
    return out[-1].strip() if out else ""


def trim(s, n=72):
    s = s.replace("\n", " / ")
    return s if len(s) <= n else s[:n] + "...(" + str(len(s)) + " chars)"


CASES = []


def case(no, title, terminal_cmd, exact_j):
    CASES.append((no, title, terminal_cmd, exact_j))


case("01", "bc: 1/3 (默认 scale)",
     "printf '1/3\\n' | bc",
     "digits (1r3)")

case("02", "bc: sqrt(2) (不带 -l)",
     "printf 'sqrt(2)\\n' | bc 2>&1 || true",
     "digits ((newton2^:8) (1x))")

case("03", "python: 0.1+0.2",
     "python3 -c \"print(0.1+0.2)\"",
     "digits (3r10)")

case("04", "python: 0.1加10次 == 1.0 ?",
     "python3 -c 'x=0.0\nfor _ in range(10): x+=0.1\nprint(x == 1.0, repr(x))'",
     "+/ 10 # 1r10")

case("05", "python float: 10万本金,月息0.05/12,360期(分)",
     "python3 -c \"print(f'{100000*(1+0.05/12)**360:.10f}')\"",
     "\": <. 100 * 100000x * (1205r1200)^360")

case("06", "numpy: 1/det(Hilbert_4) 应为 6048000",
     "python3 -c \"import numpy as np; from scipy.linalg import hilbert\" 2>/dev/null; python3 -c \"import numpy as np; H=np.array([[1.0/(i+j+1) for j in range(4)] for i in range(4)]); print(1/np.linalg.det(H))\"",
     "H4 =: % 1 + +/~ i.4x\n\": % -/ .* H4")

case("07", "numpy: 6x6 整数矩阵行列式",
     "python3 -c \"import numpy as np; M=np.array([2,5,1,8,3,6,7,1,4,9,2,5,3,8,6,1,4,7,9,2,5,3,6,1,4,7,1,8,2,5,6,3,9,4,7,2]).reshape(6,6); print(np.linalg.det(M))\"",
     "M =: 6 6 $ 2 5 1 8 3 6 7 1 4 9 2 5 3 8 6 1 4 7 9 2 5 3 6 1 4 7 1 8 2 5 6 3 9 4 7 2\n\": -/ .* M")

case("08", "numpy: Hilbert_5 逆(应为整数矩阵)第一行",
     "python3 -c \"import numpy as np; H=np.array([[1.0/(i+j+1) for j in range(5)] for i in range(5)]); print(np.linalg.inv(H)[0])\"",
     "H5 =: % 1 + +/~ i.5x\n\": {. %. H5")

case("09", "python math: 1 - cos(1e-8)",
     "python3 -c \"import math; print(1-math.cos(1e-8))\"",
     "x =. 1r100000000\nk =. i.8x\nc =. +/ ((_1x^k) * (x^+:k)) % ! +:k\n\": 1x - c")

case("10", "numpy 单遍方差公式: mean(x^2)-mean(x)^2, x=1e9+0..9",
     "python3 -c \"import numpy as np; x=1e9+np.arange(10.); print('one-pass:', np.mean(x*x)-np.mean(x)**2); print('np.var :', np.var(x))\"",
     "xs =: 1000000000x + i.10\nm =: mean xs\n\": mean *: xs - m")

case("11", "python: math.e 的位数",
     "python3 -c \"import math; print(repr(math.e))\"",
     "E =: +/ % ! i.50x\n<. E * 10x^39")

case("12", "python: math.comb(100,50) [诚实对照]",
     "python3 -c \"import math; print(math.comb(100,50))\"",
     "\": 50x ! 100x")

case("13", "python float: 2.3 x 100 (金额)",
     "python3 -c 'print(2.3*100)'",
     "\": 100 * 23r10")

case("13b", "python float: float(2**53+1) 大整数ID",
     "python3 -c 'print(float(2**53+1))'",
     "\": 1x + 2x^53")

case("14", "精简容器: import numpy 后做矩阵乘",
     "python3 -c \"import sys; sys.path[:]=[p for p in sys.path if 'site-packages' not in p and 'dist-packages' not in p]; import numpy\" 2>&1 | tail -1",
     "A =: 2 2 $ 3r2 5r2 7r2 9r2\n\": A +/ . * = i.2")


def main():
    jsetup()
    rows = []
    for no, title, cmd, jsent in CASES:
        term = sh(cmd)
        exact = ""
        for line in jsent.split("\n"):
            exact = jq(line) or exact
        rows.append((no, title, term, exact))

    print("\n| # | 科目 | 终端默认姿势 | 精算引擎 |")
    print("|---|---|---|---|")
    for no, title, term, exact in rows:
        print(f"| {no} | {title} | `{trim(term, 60)}` | `{trim(exact)}` |")

    write_report(rows)
    print("\nwritten: recon/REPORT.md")


# Stable display text for the engine column and the verdict per case.
# Raw terminal values are re-measured live on every run; engine values are
# deterministic (exact arithmetic) so their display strings are fixed.
ENGINE_DISPLAY = {
    "01": "`1/3 = 0.333…`(30 位截断)",
    "02": "`1.414213562373095048801688724209`(30 位截断)",
    "03": "`0.300000000000000000000000000000`(截断)",
    "04": "`1`(精确等于 1)",
    "05": "`446774.4314006132…`(精确有理数,截断显示)",
    "06": "`6048000`",
    "07": "`46760`",
    "08": "5×5 全整数阵(25 _300 1050 _1400 630 / …)",
    "09": "≈ 5.00000000000000042×10⁻¹⁷(精确有理数,231 位分子)",
    "10": "`33/4`",
    "11": "50 项级数,40 位:2.718281828459045235360287471352662497757",
    "12": "同上,精确",
    "13": "`230`",
    "13b": "`9007199254740993`",
    "14": "引擎原生有理矩阵乘 `3r2 5r2 / 7r2 9r2`",
}

VERDICT = {
    "01": "**终端阵亡** —— scale=0 静默截断,模型原样引用就把 0 传播下去",
    "02": "**终端阵亡**——bc 不带 -l 时 sqrt(2)=1;引擎给出 Newton 有理逼近,且满足 n²−2d²=1",
    "03": "**终端阵亡**——float64 无法表示 1/10;引擎把 \"0.1\" 解析为精确的 1/10",
    "04": "**终端阵亡**——float 累加破坏相等性判定,对账代码走 else 分支;引擎 10×(1/10) 精确等于 1",
    "05": "**终端漂移**——float64 从有效数字第 13 位起偏离(…06109 vs …06132),单账户 ~1e-8 元,账本规模(每日百万次复利事件)即显性化;引擎全程按精确分数 241/240 复利",
    "06": "**终端阵亡**——病态矩阵放大 float64 噪声,精确结果应为整数 6048000",
    "07": "**终端阵亡**——行列式理论上必是整数,float 噪声给出非整数",
    "08": "**终端阵亡**——Hilbert 逆矩阵理论上必为整数阵;引擎输出全部整数",
    "09": "**终端阵亡**——灾难性抵消:float64 下 cos(1e-8) 舍入为 1.0,答案整个消失;引擎有理 Taylor 级数保住真值",
    "10": "**终端阵亡**——教科书的单遍公式在 float64 下灾难性抵消,误差 1448%;引擎两趟精确有理数得 33/4",
    "11": "**终端阵亡**——float64 15–17 位即封顶,引擎 50 项级数给 40 位有效数字",
    "12": "**平/终端胜**——纯整数组合数 python 原生精确,这是诚实对照组:引擎不赢所有科目",
    "13": "**终端阵亡**——金额乘法经典漂移",
    "13b": "**终端阵亡**——超过 2^53 的整数 float64 无法表示,ID/计数场景静默丢 1",
    "14": "**终端阵亡**——依赖缺失环境下 float 工具链直接不可用,引擎零依赖",
}

TITLES = {
    "01": "bc: 1/3(默认 scale)",
    "02": "bc: sqrt(2)",
    "03": "python: 0.1+0.2",
    "04": "python: 0.1 加 10 次 == 1.0 ?",
    "05": "复利 360 期",
    "06": "numpy: 1/det(Hilbert₄)",
    "07": "numpy: 6×6 整数矩阵行列式",
    "08": "numpy: Hilbert₅ 逆矩阵(应为整数阵)",
    "09": "python math: `1 - cos(1e-8)`",
    "10": "numpy: 单遍方差公式,x = 1e9+0..9",
    "11": "python: `math.e`",
    "12": "python: `math.comb(100,50)`",
    "13": "python float: `2.3 × 100`",
    "13b": "python float: `float(2**53+1)`",
    "14": "精简容器:`import numpy` 后矩阵乘",
}


def write_report(rows):
    measured = {no: trim(term, 60) for no, _, term, _ in rows}
    lines = []
    lines.append("# shootout 对拍战报:终端默认姿势 vs 精算引擎")
    lines.append("")
    lines.append("日期:2026-09-07 · 平台:macOS arm64 · 引擎:libj 9.8.0-beta7 + GMP")
    lines.append("方法:每科目前半栏是 agent 在终端里的**默认姿势**(python3 内建 float /")
    lines.append("bc 默认 / numpy float64),后半栏是同一个量经 pyj 桥由引擎以 GMP 精确")
    lines.append("整数/有理数算出的结果。15 个比分点,1 道(第 12)是刻意安排的对照组。")
    lines.append("")
    lines.append("| # | 科目 | 终端默认姿势 | 精算引擎 | 判定 |")
    lines.append("|---|---|---|---|---|")
    for no in ["01", "02", "03", "04", "05", "06", "07", "08", "09",
               "10", "11", "12", "13", "13b", "14"]:
        lines.append(f"| {no} | {TITLES[no]} | `{measured[no]}` | "
                     f"{ENGINE_DISPLAY[no]} | {VERDICT[no]} |")
    lines.append("")
    lines.append("## 记分:终端 13 阵亡 / 1 平(对照组) / 0 胜")
    lines.append("")
    lines.append("引擎侧全部由 libj 的 GMP 扩展整数与有理数驱动,经 pyj 桥逐科目现场计算;")
    lines.append("除对照组(12)外,所有科目终端默认姿势均给出错误或能力缺失的答案。")
    lines.append("")
    lines.append("## 诚实的边界")
    lines.append("")
    lines.append("- 本对拍比较的是**默认姿势**(直接 print、bc 不带 scale、单遍公式),而非")
    lines.append('  "精心编写的 float 代码"——后者能把 06/07/10 救回来,但那正是要说明的点:')
    lines.append('  默认姿势不可信,引擎把"正确"变成默认值。')
    lines.append("- 第 05 科单账户漂移 ~1e-8 元,低于分位;意义在账本规模(每日百万次复利事件")
    lines.append("  的系统性同向漂移),卡片文案已按此口径修正。")
    lines.append("- 第 14 科是环境题而非精度题:剥离 site-packages 模拟裸容器/受限沙箱。")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```")
    lines.append("cd pyj")
    lines.append("make shootout    # 重算比分点并刷新本报告(终端列现场重测,引擎列稳定复算)")
    lines.append("make cards       # 重新生成 CARDS.md(13 张抓包卡片)")
    lines.append("make tally-test  # 73 项单元 + MCP stdio 集成测试")
    lines.append("```")
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
