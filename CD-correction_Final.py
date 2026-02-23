import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

APP_NAME = "CD spectral correction"


# -------------------------
# Excel + parsing helpers
# -------------------------
def _norm(s):
    return "" if s is None else str(s).strip().lower()


def _trailing_num(name: str):
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else 10**9


def ensure_openpyxl_available():
    """
    Pandas needs openpyxl to read/write .xlsx.
    Give a friendly message if missing.
    """
    try:
        import openpyxl  # noqa: F401
    except Exception as e:
        raise ImportError(
            "Missing optional dependency 'openpyxl'.\n\n"
            "Fix (in VS Code terminal):\n"
            "  python -m pip install openpyxl\n\n"
            f"Original error: {e}"
        )


def parse_excel_columns(file_path, sheet_name=None):
    """
    Expected columns:
      - Wavelength (or wl/lambda/λ)
      - N columns of CD_XK_* (or CDXK_*)
      - N columns of E_T_* (or ET_*)

    Returns:
      (wl, CDXK, ET, labels), sheet_names
    """
    ensure_openpyxl_available()

    xls = pd.ExcelFile(file_path)  # engine auto-detected (openpyxl for xlsx)
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    cols = list(df.columns)
    cols_norm = [_norm(c) for c in cols]

    # wavelength column
    wl_col = None
    for c, cn in zip(cols, cols_norm):
        if cn in {"wavelength", "wl", "lambda", "λ"}:
            wl_col = c
            break
    if wl_col is None:
        raise ValueError("Missing wavelength column (accepted: Wavelength, wl, lambda, λ).")

    # CD_XK and E_T columns
    cd_cols = []
    et_cols = []

    for c, cn in zip(cols, cols_norm):
        if cn.startswith("cd_xk") or cn.startswith("cdxk"):
            cd_cols.append(c)
        elif cn.startswith("e_t") or cn == "et" or cn.startswith("et"):
            et_cols.append(c)

    cd_cols = sorted(cd_cols, key=_trailing_num)
    et_cols = sorted(et_cols, key=_trailing_num)

    if not cd_cols:
        raise ValueError("No CD_XK columns found. Expected headers like CD_XK_1 ... CD_XK_N.")
    if not et_cols:
        raise ValueError("No E_T columns found. Expected headers like E_T_1 ... E_T_N.")

    N = min(len(cd_cols), len(et_cols))
    cd_cols = cd_cols[:N]
    et_cols = et_cols[:N]

    wl = pd.to_numeric(df[wl_col], errors="coerce").to_numpy(dtype=float)
    CDXK = df[cd_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ET = df[et_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # drop rows with NaNs
    good = np.isfinite(wl) & np.isfinite(CDXK).all(axis=1) & np.isfinite(ET).all(axis=1)
    wl = wl[good]
    CDXK = CDXK[good, :]
    ET = ET[good, :]

    labels = [f"Sample {i+1}" for i in range(N)]
    return (wl, CDXK, ET, labels), xls.sheet_names


# -------------------------
# Solver core
# -------------------------
def solve_per_wavelength(wl, cd_xk, et, C):
    """
    For each wavelength j, solve least squares for x = [r, CD]:

      y_i = cd_xk[j,i]
      t_i = 10^(et[j,i])

      (y_i*t_i)*r + (-C_i)*CD = (-y_i)

    Returns dict with arrays of length nwl.
    """
    wl = np.asarray(wl, dtype=float)
    Y = np.asarray(cd_xk, dtype=float)
    ET = np.asarray(et, dtype=float)
    C = np.asarray(C, dtype=float)

    nwl, ns = Y.shape
    CD = np.full(nwl, np.nan, dtype=float)
    r = np.full(nwl, np.nan, dtype=float)
    rank = np.zeros(nwl, dtype=int)
    cond = np.full(nwl, np.nan, dtype=float)
    rms_resid = np.full(nwl, np.nan, dtype=float)

    for j in range(nwl):
        y = Y[j, :]
        t = np.power(10.0, ET[j, :])

        A = np.column_stack([y * t, -C])  # (N,2)
        b = -y

        # condition number diagnostic
        try:
            svals = np.linalg.svd(A, compute_uv=False)
            if len(svals) >= 2 and svals[-1] > 0:
                cond[j] = float(svals[0] / svals[-1])
            else:
                cond[j] = np.inf
        except Exception:
            cond[j] = np.nan

        x, _, rnk, _ = np.linalg.lstsq(A, b, rcond=None)
        rank[j] = int(rnk)

        r[j] = float(x[0])
        CD[j] = float(x[1])

        resid_vec = A @ x - b
        rms_resid[j] = float(np.sqrt(np.mean(resid_vec**2)))

    return {
        "Wavelength": wl,
        "CD": CD,
        "Is_I0": r,
        "Rank": rank,
        "ConditionNumber": cond,
        "RMS_Residual_per_wavelength": rms_resid,
    }


def forward_model_cd_xk(CD, Is_I0, et, C):
    """
    CD_XK_calc_i(λ) = (C_i * CD(λ)) / (1 + 10^(E_T,i(λ)) * Is_I0(λ))
    """
    CD = np.asarray(CD, dtype=float)[:, None]       # (nwl,1)
    r = np.asarray(Is_I0, dtype=float)[:, None]     # (nwl,1)
    ET = np.asarray(et, dtype=float)                # (nwl,ns)
    C = np.asarray(C, dtype=float)[None, :]         # (1,ns)

    denom = 1.0 + np.power(10.0, ET) * r
    return (C * CD) / denom


def numerator_from_each_sample(cd_xk_exp, et, Is_I0):
    """
    Per-sample numerator estimate derived from each sample's CD_XK_i and E_T_i:

      (C_i*CD)_from_sample_i(λ) = CD_XK_exp,i(λ) * (1 + 10^(E_T,i(λ)) * r(λ))

    This is what you asked for: it depends on each sample's experimental CD_XK_i.
    """
    Y = np.asarray(cd_xk_exp, dtype=float)          # (nwl,ns)
    ET = np.asarray(et, dtype=float)                # (nwl,ns)
    r = np.asarray(Is_I0, dtype=float)[:, None]     # (nwl,1)
    return Y * (1.0 + np.power(10.0, ET) * r)       # (nwl,ns)


# -------------------------
# GUI
# -------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1500x920")

        self.file_path = None
        self.sheet_names = []
        self.data = None     # (wl, CDXK, ET, labels)
        self.results = None  # dict

        self._build()

    def _build(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open Excel file…", command=self.open_file).pack(side=tk.LEFT, **pad)

        ttk.Label(top, text="Sheet:").pack(side=tk.LEFT, **pad)
        self.sheet_var = tk.StringVar()
        self.sheet_combo = ttk.Combobox(top, textvariable=self.sheet_var, state="readonly", width=30)
        self.sheet_combo.pack(side=tk.LEFT, **pad)
        self.sheet_combo.bind("<<ComboboxSelected>>", lambda e: self.parse_sheet())

        self.knorm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top,
            text="I confirm CD_XK is K-factor normalized (CD_XK = CD_X(λ) / K)",
            variable=self.knorm_var
        ).pack(side=tk.LEFT, **pad)

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(mid, text="Concentration factors C1..CN (comma/space separated):").pack(side=tk.LEFT, **pad)
        self.c_entry = ttk.Entry(mid, width=70)
        self.c_entry.pack(side=tk.LEFT, **pad)

        ttk.Button(mid, text="Run solve + plot", command=self.run).pack(side=tk.LEFT, **pad)
        ttk.Button(mid, text="Export results to Excel…", command=self.export).pack(side=tk.LEFT, **pad)

        self.status = tk.StringVar(value="Load an Excel file to begin.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        # Plots
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=1)

        # Left: CD, Is/I0, (Ci*CD)_from_each_sample
        self.figL = Figure(figsize=(6.6, 6.2), dpi=100)
        self.ax1 = self.figL.add_subplot(311)
        self.ax2 = self.figL.add_subplot(312)
        self.ax3 = self.figL.add_subplot(313)

        self.canvasL = FigureCanvasTkAgg(self.figL, master=left)
        self.canvasL.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvasL, left)

        # Right: CD_XK overlay + residuals
        self.figR = Figure(figsize=(6.6, 6.2), dpi=100)
        self.ax4 = self.figR.add_subplot(211)
        self.ax5 = self.figR.add_subplot(212)

        self.canvasR = FigureCanvasTkAgg(self.figR, master=right)
        self.canvasR.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvasR, right)

        helpf = ttk.LabelFrame(self, text="Expected Excel columns")
        helpf.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        txt = (
            "Required: Wavelength; N columns like CD_XK_1..N; N columns like E_T_1..N.\n"
            "Solver finds shared CD(λ) and Is/I0(λ) using least squares across N samples.\n"
            "Plot #3 and export include per-sample (C_i*CD)(λ) derived from each sample's CD_XK,i."
        )
        ttk.Label(helpf, text=txt).pack(anchor="w", padx=8, pady=4)

    def open_file(self):
        fp = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if not fp:
            return
        self.file_path = fp

        try:
            ensure_openpyxl_available()
            xls = pd.ExcelFile(fp)
            self.sheet_names = xls.sheet_names
            self.sheet_combo["values"] = self.sheet_names
            self.sheet_var.set(self.sheet_names[0])
            self.parse_sheet()
            self.status.set(f"Selected file: {fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read Excel file:\n\n{e}")
            self.file_path = None

    def parse_sheet(self):
        if not self.file_path:
            return
        sheet = self.sheet_var.get() or None
        try:
            (wl, CDXK, ET, labels), _ = parse_excel_columns(self.file_path, sheet)
            self.data = (wl, CDXK, ET, labels)
            N = CDXK.shape[1]
            self.status.set(f"Parsed '{sheet}': {len(wl)} wavelengths × {N} samples.")
            if not self.c_entry.get().strip():
                self.c_entry.delete(0, tk.END)
                self.c_entry.insert(0, ", ".join(["1"] * N))
        except Exception as e:
            self.data = None
            messagebox.showerror("Parse error", str(e))
            self.status.set("Parsing failed. Check headers (CD_XK_* and E_T_*).")

    def run(self):
        if not self.knorm_var.get():
            messagebox.showwarning(
                "K-normalization required",
                "Terminated.\n\nThe CD spectra in the input file must be K-normalized:\n"
                "CD_XK = CD_X(λ) / K"
            )
            return
        if self.data is None:
            messagebox.showwarning("No data", "Load and parse an Excel sheet first.")
            return

        wl, CDXK, ET, labels = self.data
        N = CDXK.shape[1]

        # concentrations
        try:
            C = [float(x) for x in re.split(r"[,\s;]+", self.c_entry.get().strip()) if x]
        except Exception:
            messagebox.showerror("Input error", "Could not parse concentration factors. Example: 1, 0.5, 0.25")
            return
        if len(C) != N:
            messagebox.showerror("Input error", f"Number of concentration factors ({len(C)}) must match N ({N}).")
            return
        C = np.asarray(C, float)

        self.status.set("Solving CD(λ) and Is/I0(λ) per wavelength…")
        self.update_idletasks()

        solved = solve_per_wavelength(wl, CDXK, ET, C)
        CD = solved["CD"]
        r = solved["Is_I0"]

        # forward model + residual
        CDXK_calc = forward_model_cd_xk(CD, r, ET, C)
        resid = CDXK - CDXK_calc

        # per-sample numerator estimate derived from each sample's CD_XK,i
        CxCD_from_sample = numerator_from_each_sample(CDXK, ET, r)          # (nwl,N)
        CD_from_sample = CxCD_from_sample / C[None, :]                       # (nwl,N)

        self.results = {
            "inputs": {"Wavelength": wl, "CD_XK": CDXK, "E_T": ET, "C": C, "labels": labels},
            "solved": solved,
            "CD_XK_calc": CDXK_calc,
            "Residual": resid,
            "CxCD_from_each_sample": CxCD_from_sample,
            "CD_from_each_sample": CD_from_sample,
        }

        self._plot()
        self.status.set("Solved successfully. Export is available.")

    def _plot(self):
        inp = self.results["inputs"]
        sol = self.results["solved"]

        wl = sol["Wavelength"]
        CD = sol["CD"]
        r = sol["Is_I0"]
        labels = inp["labels"]

        CDXK = inp["CD_XK"]
        CDXK_calc = self.results["CD_XK_calc"]
        resid = self.results["Residual"]

        CxCD_from_sample = self.results["CxCD_from_each_sample"]
        N = CDXK.shape[1]

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            ax.clear()

        # CD(λ)
        self.ax1.plot(wl, CD, label="Solved CD(λ)")
        self.ax1.set_title("Solved CD(λ)")
        self.ax1.set_xlabel("Wavelength")
        self.ax1.set_ylabel("CD")
        self.ax1.grid(True)
        self.ax1.legend(loc="best", fontsize=8)

        # Is/I0(λ)
        self.ax2.plot(wl, r, label="Solved Is/I0(λ)")
        self.ax2.set_title("Solved Is(λ)/I0(λ)")
        self.ax2.set_xlabel("Wavelength")
        self.ax2.set_ylabel("Is/I0")
        self.ax2.grid(True)
        self.ax2.legend(loc="best", fontsize=8)

        # (Ci*CD)(λ) derived from each sample's CD_XK,i (THIS is what you asked for)
        for i in range(N):
            self.ax3.plot(wl, CxCD_from_sample[:, i], label=f"{labels[i]}: (Ci·CD)i from CD_XK")
        self.ax3.set_title("(Ci × CD)(λ) derived from each sample's CD_XK,i")
        self.ax3.set_xlabel("Wavelength")
        self.ax3.set_ylabel("Ci·CD (from sample i)")
        self.ax3.grid(True)
        self.ax3.legend(loc="best", fontsize=7, ncol=2)

        self.figL.tight_layout()
        self.canvasL.draw()

        # CD_XK exp vs calc
        for i in range(N):
            self.ax4.plot(wl, CDXK[:, i], linestyle="--", label=f"Exp {labels[i]}")
            self.ax4.plot(wl, CDXK_calc[:, i], linestyle="-", label=f"Calc {labels[i]}")
        self.ax4.set_title("CD_XK: experimental (dashed) vs calculated (solid)")
        self.ax4.set_xlabel("Wavelength")
        self.ax4.set_ylabel("CD_XK")
        self.ax4.grid(True)
        self.ax4.legend(loc="best", fontsize=7, ncol=2)

        # Residuals
        for i in range(N):
            self.ax5.plot(wl, resid[:, i], label=f"{labels[i]}")
        self.ax5.set_title("Residuals: experimental − calculated")
        self.ax5.set_xlabel("Wavelength")
        self.ax5.set_ylabel("Residual")
        self.ax5.grid(True)
        self.ax5.legend(loc="best", fontsize=7, ncol=2)

        self.figR.tight_layout()
        self.canvasR.draw()

    def export(self):
        if self.results is None:
            messagebox.showwarning("No results", "Run the solve first, then export.")
            return

        outpath = filedialog.asksaveasfilename(
            title="Save results as Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if not outpath:
            return

        ensure_openpyxl_available()

        inp = self.results["inputs"]
        sol = self.results["solved"]
        wl = sol["Wavelength"]
        N = inp["CD_XK"].shape[1]
        labels = inp["labels"]
        C = inp["C"]

        # README
        df_readme = pd.DataFrame({"README": [
            APP_NAME,
            "",
            "Input requirement:",
            " - CD_XK must be K-factor normalized: CD_XK = CD_X(λ) / K",
            "",
            "Model (per sample i):",
            " CD_XK,i(λ) = (C_i * CD(λ)) / (1 + 10^(E_T,i(λ)) * (Is(λ)/I0(λ)))",
            "",
            "Unknowns solved per wavelength:",
            " - CD(λ)   (shared across samples)",
            " - Is(λ)/I0(λ) (shared across samples)",
            "",
            "Additional per-sample derived quantity (from experimental CD_XK,i):",
            " - (C_i*CD)_from_sample_i(λ) = CD_XK_exp,i(λ) * (1 + 10^(E_T,i(λ)) * (Is(λ)/I0(λ)))",
            " - CD_from_sample_i(λ) = (C_i*CD)_from_sample_i(λ) / C_i",
            "",
            "Sheets:",
            " - Solved: CD(λ), Is/I0(λ), and per-wavelength diagnostics",
            " - Inputs: wavelength + CD_XK,i + E_T,i + C_i",
            " - CD_XK_Calculated: model-predicted CD_XK,i(λ) using solved CD and Is/I0",
            " - Residuals: Experimental − Calculated",
            " - CxCD_from_each_sample: per-sample (C_i*CD)(λ) derived from each CD_XK,i",
            
        ]})

        df_solved = pd.DataFrame({
            "Wavelength": wl,
            "CD": sol["CD"],
            "Is_I0": sol["Is_I0"],
            "Rank": sol["Rank"],
            "ConditionNumber": sol["ConditionNumber"],
            "RMS_Residual_per_wavelength": sol["RMS_Residual_per_wavelength"],
        })

        df_inputs = pd.DataFrame({"Wavelength": wl})
        for i in range(N):
            df_inputs[f"CD_XK_exp_{i+1}"] = inp["CD_XK"][:, i]
        for i in range(N):
            df_inputs[f"E_T_{i+1}"] = inp["E_T"][:, i]
        for i in range(N):
            df_inputs[f"C_{i+1}"] = C[i]

        df_calc = pd.DataFrame({"Wavelength": wl})
        for i in range(N):
            df_calc[f"CD_XK_calc_{i+1}"] = self.results["CD_XK_calc"][:, i]

        df_res = pd.DataFrame({"Wavelength": wl})
        for i in range(N):
            df_res[f"Residual_{i+1}"] = self.results["Residual"][:, i]

        # NEW: per-sample (Ci*CD)(λ) derived from each sample's CD_XK,i
        df_cxcd = pd.DataFrame({"Wavelength": wl})
        cxcd = self.results["CxCD_from_each_sample"]
        for i in range(N):
            df_cxcd[f"CxCD_from_sample_{i+1}_{labels[i]}_C={C[i]}"] = cxcd[:, i]


        with pd.ExcelWriter(outpath, engine="openpyxl") as w:
            df_readme.to_excel(w, sheet_name="README", index=False)
            df_solved.to_excel(w, sheet_name="Solved", index=False)
            df_inputs.to_excel(w, sheet_name="Inputs", index=False)
            df_calc.to_excel(w, sheet_name="CD_XK_Calculated", index=False)
            df_res.to_excel(w, sheet_name="Residuals", index=False)
            df_cxcd.to_excel(w, sheet_name="CxCD_from_each_sample", index=False)

        messagebox.showinfo("Export complete", f"Saved:\n{outpath}")


if __name__ == "__main__":
    app = App()
    app.mainloop()