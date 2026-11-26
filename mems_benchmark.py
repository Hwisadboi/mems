#!/usr/bin/env Python 3
# -*- coding: utf-8 -*-
"""
=================

Benchmark cho đề tài "Khảo sát độ tin cậy của nghiệm số gần ngưỡng pull-in
trong mô hình cân bằng tĩnh MEMS" (mục 13 của MEMS-pullin.pdf).

Bài toán một ẩn (chuẩn hóa):
        f(u) = u (1-u)^2 - alpha ,      alpha = r * 4/27 ,   delta = 1 - r
        f'(u) = (1-u)(1-3u)
    r < 1 : dưới ngưỡng (có nghiệm ổn định us, không ổn định uu, và nghiệm u>1)
    r = 1 : đúng ngưỡng (nghiệm kép u = 1/3)
    r > 1 : vượt ngưỡng (không còn nghiệm trong [0,1))

Ba phương pháp: Chia đôi (bisection), Newton, Secant.
Hai tiêu chuẩn dừng:  (R) |f(u_n)| <= tau     (S) |u_n - u_{n-1}| <= tau
    (với Chia đôi, (S) được hiểu là nửa bề rộng khoảng <= tau, tức cận sai số
     (b-a)/2^n <= tau, đúng như công thức số bước ở mục 12.2)

Nghiệm tham chiếu được tính bằng số học thập phân 80 chữ số (module decimal),
nên không phụ thuộc thư viện ngoài. Chỉ matplotlib là tùy chọn (để vẽ hình).

Hai loại sai số thật được ghi lại:
    err    : |u - u*| với u* là nghiệm đúng của bài toán lý tưởng (r chính xác)
    err_fp : |u - u*_fp| với u*_fp là nghiệm đúng của bài toán mà máy thực sự
             giải (alpha đã bị làm tròn về float64, mục 12.8)

Cách chạy:
    python mems_benchmark.py table2      # đối chiếu Bảng 2 của PDF (kiểm tra mã)
    python mems_benchmark.py e1          # E1: gần ngưỡng
    python mems_benchmark.py e2          # E2: điểm khởi tạo
    python mems_benchmark.py e3          # E3: tiêu chuẩn dừng
    python mems_benchmark.py all         # chạy tất cả
Kết quả (CSV + hình PNG) được ghi vào thư mục results/

=================
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from functools import lru_cache

getcontext().prec = 80

ALPHA_C = Fraction(4, 27)
U_C = 1.0 / 3.0
DIVERGE = 1e6
NAN = float("nan")
METHODS = ("bisection", "newton", "secant")


# =============================================================================
# 1. Nghiệm tham chiếu độ chính xác cao
# =============================================================================
def _dec(q: Fraction) -> Decimal:
    return Decimal(q.numerator) / Decimal(q.denominator)

"""Chia đôi bằng Decimal trên khoảng có đổi dấu (độ chính xác ~1e-80)."""
def _bisect_dec(alpha: Decimal, lo: Decimal, hi: Decimal, iters: int = 300) -> Decimal:
    def g(u):
        return u * (1 - u) ** 2 - alpha
    glo = g(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        gm = g(mid)
        if (gm > 0) == (glo > 0):
            lo, glo = mid, gm
        else:
            hi = mid
    return (lo + hi) / 2


@lru_cache(maxsize=None)
def exact_roots(alpha: Fraction) -> dict:
    """Các nghiệm đúng của u(1-u)^2 = alpha, nhãn -> Decimal.
       alpha < 4/27 : 'stable' (us), 'unstable' (uu), 'outside' (u>1)
       alpha = 4/27 : 'double' (u=1/3), 'outside' (u=4/3)
       alpha > 4/27 : chỉ còn 'outside'
    """
    d = _dec(alpha)
    third = Decimal(1) / Decimal(3)
    roots = {}
    if alpha < ALPHA_C:
        roots["stable"] = _bisect_dec(d, Decimal(0), third)
        roots["unstable"] = _bisect_dec(d, third, Decimal(1))
    elif alpha == ALPHA_C:
        roots["double"] = third
    roots["outside"] = _bisect_dec(d, Decimal(1), Decimal(2))
    return roots


@dataclass
class Ref:
    r: Fraction        # r = alpha/alpha_c chính xác (hữu tỉ)
    alpha: float       # alpha đã làm tròn float64 -> đưa vào bộ giải
    roots: dict        # nghiệm đúng của bài toán lý tưởng
    roots_fp: dict     # nghiệm đúng của bài toán với alpha float64

    @property
    def delta(self) -> float:
        return float(1 - self.r)


def get_ref(r: Fraction) -> Ref:
    alpha_true = ALPHA_C * r
    alpha_fp = float(alpha_true)
    return Ref(r, alpha_fp, exact_roots(alpha_true), exact_roots(Fraction(alpha_fp)))


def fprime_at(root: Decimal) -> float:
    return float((1 - root) * (1 - 3 * root))

"""Sai số thật so với nghiệm ổn định (r<1) hoặc nghiệm kép (r=1); NaN nếu r>1."""
def err_to(u: float, roots: dict) -> float:
    if not math.isfinite(u):
        return NAN
    for key in ("stable", "double"):
        if key in roots:
            return float(abs(Decimal(u) - roots[key]))
    return NAN

"""Nhãn nhánh của một điểm hội tụ: nghiệm đúng nào là gần nhất trong khe hở."""
def classify(u: float, roots: dict) -> str:
    if not math.isfinite(u):
        return "nan"
    if u < 0:
        return "out_of_domain"
    if u >= 1:
        return "outside"
    cand = {k: abs(Decimal(u) - v) for k, v in roots.items() if k != "outside"}
    if not cand:
        return "no_equilibrium"        # r > 1: hội tụ 'giả' trong khe hở
    return min(cand, key=cand.get)


# =============================================================================
# 2. Bài toán và ba phương pháp giải
# =============================================================================
def problem(alpha: float):
    def f(u):
        return u * (1.0 - u) ** 2 - alpha

    def df(u):
        return (1.0 - u) * (1.0 - 3.0 * u)
    return f, df


@dataclass
class Result:
    method: str
    status: str        # converged | max_iter | zero_derivative | zero_denominator
    #                    | diverged | no_bracket
    u: float           # điểm lặp cuối
    iters: int         # số vòng lặp
    nf: int            # số lần tính f
    ndf: int           # số lần tính f'
    res: float         # |f(u)| tại điểm cuối
    step: float        # |u_n - u_{n-1}| (Chia đôi: nửa bề rộng khoảng)


def _stop(crit, fu, step, tol):
    return (crit == "R" and abs(fu) <= tol) or (crit == "S" and step <= tol)


def bisection(f, a, b, tol, crit, max_iter):
    fa, fb = f(a), f(b)
    nf = 2
    if not ((fa < 0 < fb) or (fb < 0 < fa)):
        return Result("bisection", "no_bracket", NAN, 0, nf, 0, NAN, NAN)
    for n in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fc = f(c)
        nf += 1
        half = 0.5 * (b - a)
        if _stop(crit, fc, half, tol):
            return Result("bisection", "converged", c, n, nf, 0, abs(fc), half)
        if (fa < 0) == (fc < 0):
            a, fa = c, fc
        else:
            b = c
    return Result("bisection", "max_iter", c, max_iter, nf, 0, abs(fc), half)


def newton(f, df, u0, tol, crit, max_iter):
    u = u0
    fu = f(u)
    nf, ndf = 1, 0
    if crit == "R" and abs(fu) <= tol:
        return Result("newton", "converged", u, 0, nf, ndf, abs(fu), NAN)
    step = NAN
    for n in range(1, max_iter + 1):
        d = df(u)
        ndf += 1
        if d == 0.0:
            return Result("newton", "zero_derivative", u, n - 1, nf, ndf, abs(fu), step)
        u_new = u - fu / d
        if not math.isfinite(u_new) or abs(u_new) > DIVERGE:
            return Result("newton", "diverged", u_new, n, nf, ndf, abs(fu), NAN)
        fu_new = f(u_new)
        nf += 1
        step = abs(u_new - u)
        u, fu = u_new, fu_new
        if _stop(crit, fu, step, tol):
            return Result("newton", "converged", u, n, nf, ndf, abs(fu), step)
    return Result("newton", "max_iter", u, max_iter, nf, ndf, abs(fu), step)


def secant(f, u0, u1, tol, crit, max_iter):
    f0, f1 = f(u0), f(u1)
    nf = 2
    step = abs(u1 - u0)
    for n in range(1, max_iter + 1):
        if f1 == f0:
            return Result("secant", "zero_denominator", u1, n - 1, nf, 0, abs(f1), step)
        u2 = u1 - f1 * (u1 - u0) / (f1 - f0)
        if not math.isfinite(u2) or abs(u2) > DIVERGE:
            return Result("secant", "diverged", u2, n, nf, 0, abs(f1), NAN)
        f2 = f(u2)
        nf += 1
        step = abs(u2 - u1)
        u0, f0, u1, f1 = u1, f1, u2, f2
        if _stop(crit, f1, step, tol):
            return Result("secant", "converged", u1, n, nf, 0, abs(f1), step)
    return Result("secant", "max_iter", u1, max_iter, nf, 0, abs(f1), step)


def solve(method, alpha, *, u0, tol, crit, max_iter, secant_h, bracket=(0.0, U_C)):
    f, df = problem(alpha)
    if method == "bisection":
        return bisection(f, bracket[0], bracket[1], tol, crit, max_iter)
    if method == "newton":
        return newton(f, df, u0, tol, crit, max_iter)
    if method == "secant":
        u1 = u0 + secant_h if u0 + secant_h < 1.0 else u0 - secant_h
        return secant(f, u0, u1, tol, crit, max_iter)
    raise ValueError(method)

"""Gộp kết quả bộ giải với nghiệm tham chiếu thành một dòng dữ liệu."""
def evaluate(res: Result, ref: Ref) -> dict:
    _, df = problem(ref.alpha)
    u = res.u
    out = dict(status=res.status, u=u, iters=res.iters, nf=res.nf, ndf=res.ndf,
               res=res.res, step=res.step,
               err=err_to(u, ref.roots), err_fp=err_to(u, ref.roots_fp))
    if res.status == "converged":
        out["branch"] = classify(u, ref.roots)
        out["check_stable"] = int(0.0 <= u < 1.0 and df(u) > 0.0)   # phép kiểm tra (3)
        out["valid"] = int(out["branch"] in ("stable", "double"))
    else:
        out["branch"] = "fail"
        out["check_stable"] = ""
        out["valid"] = 0
    return out


# =============================================================================
# 3. Tiện ích
# =============================================================================
def lsq_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def write_csv(path, rows):
    if not rows:
        return
    names = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> đã ghi {path} ({len(rows)} dòng)")

"""Định dạng số khoa học, NaN -> '-'."""
def e(x, nd=2):
    if x is None or x == "" or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}e}"


def cell(row):
    if row["status"] != "converged":
        return row["status"]
    return f"n={row['iters']:<3d} err={e(row['err'])} {row['branch']}"


def try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:          # noqa: BLE001
        print(f"  (bỏ qua vẽ hình: không import được matplotlib: {exc})")
        return None


# =============================================================================
# 4. Kiểm tra mã: tái tạo Bảng 2 của PDF
# =============================================================================
def run_table2(args):
    print("\n=== Đối chiếu Bảng 2 (nghiệm chính xác & công thức tiệm cận) ===")
    print(f"{'r':>12} {'delta':>9} {'us':>9} {'uu':>9} {'f(us)':>9} {'uu-us':>9} {'0.770sqrt(d)':>12}")
    for rs in ("0.5", "0.9", "0.99", "0.999", "0.9999", "0.999999", "0.99999999"):
        r = Fraction(rs)
        ref = get_ref(r)
        us, uu = ref.roots["stable"], ref.roots["unstable"]
        d = ref.delta
        print(f"{rs:>12} {d:9.1e} {float(us):9.5f} {float(uu):9.5f} "
              f"{fprime_at(us):9.3g} {float(uu - us):9.3g} {4 / math.sqrt(27) * math.sqrt(d):12.3g}")
    a = ref_fp_note()
    print(a)


def ref_fp_note():
    alpha_fp = Fraction(float(ALPHA_C))
    rel = float((ALPHA_C - alpha_fp) / ALPHA_C)
    roots = exact_roots(alpha_fp)
    half_gap = float(Decimal(1) / Decimal(3) - roots["stable"])
    return (f"\nfloat64(4/27) thấp hơn giá trị đúng {rel:.3e} (tương đối) -> "
            f"máy giải bài toán r=1 như delta_eff≈{rel:.2e}, hai nghiệm cách 1/3 khoảng {half_gap:.2e}")


# =============================================================================
# 5. E1 - Gần ngưỡng
# =============================================================================
def run_e1(args, plt):
    print("\n=== E1: gần ngưỡng pull-in ===")
    print(f"u0={args.u0}, secant_h={args.secant_h}, tau={args.tau:g}, max_iter={args.max_iter}")
    cases = []                                   # (nhãn, r, k) với delta=10^-k
    cases.append(("delta=5e-1", Fraction(1) - Fraction("0.5"), None))
    for k in range(1, args.e1_kmax + 1):
        cases.append((f"delta=1e-{k}", Fraction(1) - Fraction(f"1e-{k}"), k))
    cases.append(("r=1", Fraction(1), None))
    for rs in ("1.001", "1.01", "1.1"):
        cases.append((f"r={rs}", Fraction(rs), None))

    rows = []
    for crit in ("R", "S"):
        for label, r, k in cases:
            ref = get_ref(r)
            for m in METHODS:
                res = solve(m, ref.alpha, u0=args.u0, tol=args.tau, crit=crit,
                            max_iter=args.max_iter, secant_h=args.secant_h)
                row = dict(case=label, k=k if k else "", r=float(r), delta=ref.delta,
                           method=m, crit=crit, tau=args.tau)
                row.update(evaluate(res, ref))
                rows.append(row)
    write_csv(os.path.join(args.outdir, "e1_near_threshold.csv"), rows)

    for crit in ("R", "S"):
        print(f"\n-- Tiêu chuẩn dừng {crit} --")
        print(f"{'case':<12}| {'Chia đôi [0,1/3]':<30}| {'Newton':<30}| {'Secant':<30}")
        for label, r, k in cases:
            sub = {m: next(x for x in rows if x["case"] == label and x["crit"] == crit
                           and x["method"] == m) for m in METHODS}
            print(f"{label:<12}| " + "| ".join(f"{cell(sub[m]):<30}" for m in METHODS))

    # Tốc độ tăng số vòng lặp trên mỗi decade (dự đoán lý thuyết mục 12.6)
    print(f"\n-- Số vòng lặp tăng thêm mỗi decade (hồi quy trên delta=1e-{args.fit_lo}..1e-{args.fit_hi}) --")
    pred = {"newton": math.log(10) / (2 * math.log(2)),
            "secant": math.log(10) / (2 * math.log((1 + math.sqrt(5)) / 2))}
    fits = {}
    for crit in ("R", "S"):
        for m in ("newton", "secant"):
            pts = [(x["k"], x["iters"]) for x in rows
                   if x["method"] == m and x["crit"] == crit and x["k"] != ""
                   and args.fit_lo <= x["k"] <= args.fit_hi and x["status"] == "converged"]
            if len(pts) >= 3:
                fits[(crit, m)] = lsq_slope([p[0] for p in pts], [p[1] for p in pts])
                print(f"  {m:<7} crit={crit}: thực nghiệm {fits[(crit, m)]:.2f}   "
                      f"dự đoán lý thuyết {pred[m]:.2f}")
    bis = [x["iters"] for x in rows if x["method"] == "bisection" and x["crit"] == "S"
           and x["status"] == "converged" and x["k"] != ""]
    if bis:
        print(f"  Chia đôi (crit=S): số bước {min(bis)}..{max(bis)} (lý thuyết: "
              f"{math.ceil(math.log2(U_C / args.tau))}, không phụ thuộc delta)")

    if plt:
        plot_e1(args, plt, rows, fits, pred)
    return rows


def plot_e1(args, plt, rows, fits, pred):
    colors = {"bisection": "tab:gray", "newton": "tab:blue", "secant": "tab:red"}
    names = {"bisection": "Chia đôi", "newton": "Newton", "secant": "Secant"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for m in METHODS:
        pts = [x for x in rows if x["method"] == m and x["crit"] == "R" and x["k"] != ""
               and x["status"] == "converged"]
        ax[0].plot([x["k"] for x in pts], [x["iters"] for x in pts], "o-", color=colors[m],
                   label=names[m])
        ax[1].semilogy([x["k"] for x in pts], [max(x["err"], 1e-18) for x in pts], "o-",
                       color=colors[m], label=names[m])
        pts_s = [x for x in rows if x["method"] == m and x["crit"] == "S" and x["k"] != ""
                 and x["status"] == "converged"]
        ax[2].semilogy([x["k"] for x in pts_s], [max(x["err"], 1e-18) for x in pts_s], "o-",
                       color=colors[m], label=names[m])
    ks = sorted({x["k"] for x in rows if x["k"] != ""})
    ax[1].semilogy(ks, [args.tau / (4 / math.sqrt(27) * 10 ** (-k / 2)) for k in ks], "k:",
                   label=r"$\tau/f'(u_s)\approx\tau/(0.770\sqrt{\delta})$")
    ax[1].axhline(args.tau, color="k", ls="--", lw=0.8, label=r"$\tau$")
    ax[2].axhline(args.tau, color="k", ls="--", lw=0.8, label=r"$\tau$")
    ax[0].set_title(f"Số vòng lặp (crit R, τ={args.tau:g}, u0={args.u0})")
    ax[1].set_title("Sai số thật |u − us| (crit R)")
    ax[2].set_title("Sai số thật |u − us| (crit S)")
    for a in ax:
        a.set_xlabel(r"$\log_{10}(1/\delta)$  (càng phải càng gần ngưỡng)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("số vòng lặp")
    ax[1].set_ylabel("sai số thật")
    if ("R", "newton") in fits:
        ax[0].text(0.52, 0.30, f"độ dốc Newton ≈ {fits[('R', 'newton')]:.2f}/decade\n"
                   f"(lý thuyết {pred['newton']:.2f})\n"
                   f"độ dốc Secant ≈ {fits[('R', 'secant')]:.2f}/decade\n"
                   f"(lý thuyết {pred['secant']:.2f})", transform=ax[0].transAxes, fontsize=8,
                   va="top")
    fig.tight_layout()
    path = os.path.join(args.outdir, "e1_near_threshold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> đã ghi {path}")


# =============================================================================
# 6. E2 - Điểm khởi tạo
# =============================================================================
E2_LABELS = ["stable", "unstable", "outside", "double", "no_equilibrium", "out_of_domain", "fail"]


def run_e2(args, plt):
    print("\n=== E2: quét điểm khởi tạo u0 trên [0,1) ===")
    rs = [Fraction(s) for s in args.e2_r]
    n = args.e2_points
    grid = sorted({i / n for i in range(n)} | {U_C})
    print(f"r = {[float(r) for r in rs]}, {len(grid)} điểm u0, tau={args.tau:g}, crit={args.e2_crit}, "
          f"secant_h={args.secant_h}")
    rows = []
    for r in rs:
        ref = get_ref(r)
        for name, br in (("bisection[0,1/3]", (0.0, U_C)), ("bisection[0,2]", (0.0, 2.0))):
            res = solve("bisection", ref.alpha, u0=0.0, tol=args.tau, crit=args.e2_crit,
                        max_iter=args.max_iter, secant_h=args.secant_h, bracket=br)
            row = dict(r=float(r), delta=ref.delta, method=name, u0="", crit=args.e2_crit)
            row.update(evaluate(res, ref))
            rows.append(row)
        for u0 in grid:
            for m in ("newton", "secant"):
                res = solve(m, ref.alpha, u0=u0, tol=args.tau, crit=args.e2_crit,
                            max_iter=args.max_iter, secant_h=args.secant_h)
                row = dict(r=float(r), delta=ref.delta, method=m, u0=u0, crit=args.e2_crit)
                row.update(evaluate(res, ref))
                rows.append(row)
    write_csv(os.path.join(args.outdir, "e2_initial_point.csv"), rows)

    summary = []
    print(f"\n{'r':>10} {'phương pháp':<17} {'#':>5} " + " ".join(f"{l[:9]:>9}" for l in E2_LABELS)
          + "   | kiểm tra (3): TP   FP   TN   FN  đúng%")
    for r in rs:
        for m in ("bisection[0,1/3]", "bisection[0,2]", "newton", "secant"):
            sub = [x for x in rows if x["r"] == float(r) and x["method"] == m]
            cnt = {l: sum(1 for x in sub if x["branch"] == l) for l in E2_LABELS}
            conv = [x for x in sub if x["status"] == "converged"]
            tp = sum(1 for x in conv if x["check_stable"] == 1 and x["branch"] == "stable")
            fp = sum(1 for x in conv if x["check_stable"] == 1 and x["branch"] != "stable")
            tn = sum(1 for x in conv if x["check_stable"] == 0 and x["branch"] != "stable")
            fn = sum(1 for x in conv if x["check_stable"] == 0 and x["branch"] == "stable")
            acc = 100.0 * (tp + tn) / len(conv) if conv else NAN
            tot = len(sub)
            srow = dict(r=float(r), method=m, runs=tot,
                        **{f"pct_{l}": 100.0 * cnt[l] / tot for l in E2_LABELS},
                        check_TP=tp, check_FP=fp, check_TN=tn, check_FN=fn,
                        check_accuracy_pct=acc)
            summary.append(srow)
            print(f"{float(r):>10.6f} {m:<17} {tot:>5} "
                  + " ".join(f"{100.0 * cnt[l] / tot:>8.1f}%" for l in E2_LABELS)
                  + f"   | {tp:>17} {fp:>4} {tn:>4} {fn:>4}  {acc:5.1f}")
    write_csv(os.path.join(args.outdir, "e2_summary.csv"), summary)

    if plt:
        plot_e2(args, plt, rows, rs, grid)
    return rows


def plot_e2(args, plt, rows, rs, grid):
    import numpy as np
    from matplotlib.colors import ListedColormap
    order = ["stable", "unstable", "double", "outside", "no_equilibrium", "out_of_domain", "fail"]
    cmap = ListedColormap(["#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#8c564b", "#17becf",
                           "#bbbbbb"])
    strips, names = [], []
    for r in rs:
        for m in ("newton", "secant"):
            lab = {x["u0"]: x["branch"] for x in rows if x["r"] == float(r) and x["method"] == m}
            strips.append([order.index(lab[u]) if lab[u] in order else len(order) - 1 for u in grid])
            names.append(f"r={float(r):g}\n{'Newton' if m == 'newton' else 'Secant'}")
    arr = np.array(strips)
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(strips) + 2.2))
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=len(order) - 1,
              extent=[grid[0], grid[-1], len(strips), 0], interpolation="nearest")
    ax.set_yticks([i + 0.5 for i in range(len(strips))])
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(U_C, color="k", lw=0.8, ls=":")
    ax.set_xlabel("điểm khởi tạo $u_0$   (đường chấm: $u=1/3$)")
    ax.set_title("E2: nghiệm mà từng điểm khởi tạo hội tụ tới")
    from matplotlib.patches import Patch
    legend_names = {"stable": "nhánh ổn định us", "unstable": "nhánh không ổn định uu",
                    "double": "nghiệm kép", "outside": "nghiệm ngoài khe (u≥1)",
                    "no_equilibrium": "hội tụ giả (r>1)", "out_of_domain": "u<0",
                    "fail": "không hội tụ"}
    ax.legend(handles=[Patch(color=cmap(i), label=legend_names[o]) for i, o in enumerate(order)
                       if (arr == i).any()], loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    path = os.path.join(args.outdir, "e2_basins.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> đã ghi {path}")


# =============================================================================
# 7. E3 - Tiêu chuẩn dừng
# =============================================================================

"""Cận/ước lượng lý thuyết của sai số thật khi dừng (mục 12.7); NaN nếu không có."""
def err_bound(method, crit, r, ref, tau):
    if crit == "R":
        if r < 1:
            return tau / abs(fprime_at(ref.roots["stable"]))       # nghiệm đơn: tau/|f'(us)|
        if r == 1:
            return math.sqrt(tau)                                  # nghiệm kép: sqrt(tau)
        return NAN
    if method == "bisection":
        return tau                                                 # cận (b-a)/2^n <= tau
    if r == 1:
        rho = 0.5 if method == "newton" else (math.sqrt(5) - 1) / 2
        return rho / (1 - rho) * tau                               # |e_n| ≈ rho/(1-rho) tau
    return NAN                                                     # nghiệm đơn: sai số << tau


def run_e3(args, plt):
    print("\n=== E3: tiêu chuẩn dừng R (phần dư) và S (độ thay đổi bước) ===")
    cases = [("r=0.9", Fraction("0.9")), ("r=0.99", Fraction("0.99")),
             ("r=0.999", Fraction("0.999")), ("r=1-1e-6", Fraction(1) - Fraction("1e-6")),
             ("r=1", Fraction(1))]
    taus = [10.0 ** (-j) for j in range(2, args.e3_jmax + 1)]
    print(f"u0={args.u0}, tau = 1e-2..1e-{args.e3_jmax}, secant_h={args.secant_h}")
    rows = []
    for label, r in cases:
        ref = get_ref(r)
        for tau in taus:
            for crit in ("R", "S"):
                for m in METHODS:
                    res = solve(m, ref.alpha, u0=args.u0, tol=tau, crit=crit,
                                max_iter=args.max_iter, secant_h=args.secant_h)
                    row = dict(case=label, r=float(r), delta=ref.delta, method=m, crit=crit, tau=tau)
                    row.update(evaluate(res, ref))
                    b = err_bound(m, crit, r, ref, tau)
                    row["err_bound"] = b
                    row["err_over_bound"] = (row["err"] / b) if (b == b and b > 0
                                                                 and row["err"] == row["err"]) else NAN
                    rows.append(row)
    write_csv(os.path.join(args.outdir, "e3_stopping.csv"), rows)

    # Bảng tóm tắt tại các tau tiêu biểu
    show = [t for t in (1e-4, 1e-8, 1e-12) if t in taus] or taus[::4]
    for label, r in cases:
        print(f"\n-- {label} (f'(us)={e(fprime_at(get_ref(r).roots['stable'])) if r < 1 else '0 (nghiệm kép)'}) --")
        print(f"{'tau':>8} {'crit':>4} | " + " | ".join(f"{m:<26}" for m in METHODS))
        for tau in show:
            for crit in ("R", "S"):
                parts = []
                for m in METHODS:
                    x = next(y for y in rows if y["case"] == label and y["tau"] == tau
                             and y["crit"] == crit and y["method"] == m)
                    parts.append(f"{cell(x) if x['status'] != 'converged' else 'n=%-3d err=%s' % (x['iters'], e(x['err'])):<26}")
                print(f"{tau:8.0e} {crit:>4} | " + " | ".join(parts))
    print("\n  Ghi chú: cận lý thuyết (cột err_bound trong CSV): R nghiệm đơn ≤ tau/f'(us); "
          "R nghiệm kép ≈ sqrt(tau);\n  S nghiệm kép: Newton ≈ tau, Secant ≈ 1.62 tau.")

    if plt:
        plot_e3(args, plt, rows, cases, taus)
    return rows


def plot_e3(args, plt, rows, cases, taus):
    colors = {"bisection": "tab:gray", "newton": "tab:blue", "secant": "tab:red"}
    names = {"bisection": "Chia đôi", "newton": "Newton", "secant": "Secant"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.ravel()
    for ax, (label, r) in zip(axes, cases):
        ref = get_ref(r)
        for m in METHODS:
            for crit, ls, mk in (("R", "-", "o"), ("S", "--", "s")):
                pts = [x for x in rows if x["case"] == label and x["method"] == m
                       and x["crit"] == crit and x["status"] == "converged"
                       and x["err"] == x["err"]]
                if pts:
                    ax.loglog([x["tau"] for x in pts], [max(x["err"], 1e-18) for x in pts],
                              ls=ls, marker=mk, ms=4, color=colors[m],
                              label=f"{names[m]} – {crit}")
        ax.loglog(taus, taus, "k:", lw=0.9, label=r"$\tau$")
        if r < 1:
            fp = abs(fprime_at(ref.roots["stable"]))
            ax.loglog(taus, [t / fp for t in taus], color="green", lw=0.9, ls="-.",
                      label=r"$\tau/f'(u_s)$")
        else:
            ax.loglog(taus, [math.sqrt(t) for t in taus], color="green", lw=0.9, ls="-.",
                      label=r"$\sqrt{\tau}$")
        ax.set_title(label)
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("sai số thật |u − u*|")
        ax.grid(alpha=0.3, which="both")
        ax.invert_xaxis()
    axes[0].legend(fontsize=7, ncol=2)
    for ax in axes[len(cases):]:
        ax.axis("off")
    fig.suptitle("E3: sai số thật theo ngưỡng dừng τ  (đường liền: R – phần dư, nét đứt: S – bước)")
    fig.tight_layout()
    path = os.path.join(args.outdir, "e3_stopping.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> đã ghi {path}")


# =============================================================================
# 8. main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["all", "e1", "e2", "e3", "table2"])
    p.add_argument("--outdir", default="results", help="thư mục ghi kết quả")
    p.add_argument("--tau", type=float, default=1e-12, help="ngưỡng dừng cho E1/E2")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--u0", type=float, default=0.0, help="điểm đầu cho E1/E3 (Newton, Secant)")
    p.add_argument("--secant-h", type=float, default=1e-2, help="u1 = u0 + h cho Secant")
    p.add_argument("--e1-kmax", type=int, default=14, help="E1: delta = 1e-1 ... 1e-kmax")
    p.add_argument("--fit-lo", type=int, default=3, help="E1: cận dưới k khi hồi quy độ dốc")
    p.add_argument("--fit-hi", type=int, default=10, help="E1: cận trên k khi hồi quy độ dốc")
    p.add_argument("--e2-r", nargs="+", default=["0.9", "0.99", "0.999", "0.999999"],
                   help="E2: các mức r (dưới ngưỡng)")
    p.add_argument("--e2-points", type=int, default=1000, help="E2: số điểm u0 trên [0,1)")
    p.add_argument("--e2-crit", choices=["R", "S"], default="R")
    p.add_argument("--e3-jmax", type=int, default=14, help="E3: tau = 1e-2 ... 1e-jmax")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plt = None if args.no_plots else try_matplotlib()

    if args.command in ("table2", "all"):
        run_table2(args)
    if args.command in ("e1", "all"):
        run_e1(args, plt)
    if args.command in ("e2", "all"):
        run_e2(args, plt)
    if args.command in ("e3", "all"):
        run_e3(args, plt)


if __name__ == "__main__":
    main()
