"""Main window: Home tab + one top-level tab per category, one side-tab per algorithm."""
from __future__ import annotations

import tkinter as tk
import warnings
from tkinter import messagebox, ttk

from ml_lab.core.registry import discover
from ml_lab.gui.algo_tab import AlgoTab

CATEGORY_ORDER = ["Supervised", "Unsupervised", "Robotics Demos", "Reinforcement Learning"]

HOME_TEXT = """\
## Welcome to ML Lab
An interactive playground for classical machine learning, robotics perception and control, and reinforcement learning, visualized step by step.

## How to use
1. Pick a top tab (Supervised / Unsupervised / Robotics Demos / Reinforcement Learning), then an algorithm from the list on the left.
2. Adjust the parameters (hover the \u24d8 labels for hints).  'Data' parameters regenerate the dataset immediately.
3. Press \u25b6 Run to watch the algorithm learn. Use Pause / Step to go frame by frame, Finish to jump to the result.
4. Click on the plot to add your own data points (see the hint under the parameters).
5. Read the Explanation tab under each plot - every algorithm has an "embedded engineer" analogy and things to try.

## ML in embedded-engineering language
    Dataset            <->  logged sensor captures (ADC samples, CAN frames...)
    Feature            <->  a signal / channel, usually scaled (like ADC normalisation)
    Model              <->  a transfer function / lookup table you did not hand-write
    Weights            <->  calibration constants / filter coefficients
    Hyper-parameters   <->  design choices: filter order, PID gains, sample rate
    Loss function      <->  the error signal you want to minimise (e.g. RMS error)
    Gradient descent   <->  an integrator walking down the error surface
    Learning rate      <->  loop gain: too high = oscillation/divergence, too low = sluggish
    Over-fitting       <->  a high-order filter that also tracks the noise
    Regularisation     <->  damping / limiting coefficient magnitude
    Train / test split <->  bench calibration vs. field validation
    Inference          <->  the forward pass running on the MCU every sample
    Clustering         <->  discovering operating modes (idle / run / fault) in unlabeled logs
    PCA                <->  finding the dominant vibration axis; compressing 3-axis IMU data
    Anomaly detection  <->  condition monitoring / fault detection
    Particle filter    <->  robot pose estimate from odometry + landmark ranges
    Kalman filter      <->  fuse noisy GPS position and wheel-encoder velocity
    Occupancy grid     <->  build a map from repeated laser scans
    DBSCAN             <->  group LiDAR returns into obstacle candidates
    Gaussian Process   <->  calibrate an actuator and quantify uncertainty

## Write or tweak your own algorithm
Every algorithm is a file in  ml_lab/algorithms/<category>.
Each file contains a pure-NumPy model class (the maths) and a Demo class (the visualisation).
  \u2022 Press "Open source code" in any tab, edit, save, then press "Reload code" - no restart needed.
  \u2022 To add a new algorithm: copy  ml_lab/algorithms/_TEMPLATE.py  to e.g.
    ml_lab/algorithms/supervised/my_algo.py, rename the class, restart the app. It appears automatically.
  \u2022 Set  category = "My Lab"  in your class and a new top-level tab is created for it.
  \u2022 Parameters are declared as a list - Float(...), Int(...), Choice(...), Bool(...) - and the GUI
    builds the sliders for you.
  \u2022 run() is a Python generator: draw on self.ax, then  yield "status text"  to show one animation frame.
    \u2022 See  architecture.md  for module boundaries, testing, and the extension checklist.
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Lab - Machine Learning Playground")
        self.geometry("1500x920")
        self.minsize(1100, 700)
        self._setup_style()
        self._built: set[str] = set()
        self._lazy: dict[str, type] = {}

        classes, errors = discover()

        self.main_nb = ttk.Notebook(self, style="Main.TNotebook")
        self.main_nb.pack(fill="both", expand=True)
        self._home(classes, errors)

        cats = sorted({c.category for c in classes},
                      key=lambda c: (CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99, c))
        for cat in cats:
            inner = ttk.Notebook(self.main_nb, style="Side.TNotebook")
            self.main_nb.add(inner, text=f"   {cat}   ")
            for cls in sorted((c for c in classes if c.category == cat),
                              key=lambda c: (c.order, c.name)):
                holder = ttk.Frame(inner)
                inner.add(holder, text=f" {cls.name}")
                self._lazy[str(holder)] = cls
            inner.bind("<<NotebookTabChanged>>", lambda e: self._ensure_built(e.widget))
        self.main_nb.bind("<<NotebookTabChanged>>", self._main_changed)

        if errors:
            messagebox.showwarning("Some algorithms failed to load",
                                   "\n\n".join(f"{n}:\n{tb.strip().splitlines()[-1]}" for n, tb in errors))

    def _setup_style(self):
        st = ttk.Style(self)
        if "vista" in st.theme_names():
            st.theme_use("vista")
        elif "clam" in st.theme_names():
            st.theme_use("clam")
        st.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1f4e79")
        st.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#666666")
        st.configure("Hint.TLabel", font=("Segoe UI", 8), foreground="#555555")
        st.configure("Status.TLabel", font=("Segoe UI", 9), foreground="#1f4e79", padding=4)
        st.configure("Error.TLabel", font=("Segoe UI", 9, "bold"), foreground="#b00020", padding=4)
        st.configure("Accent.TButton", font=("Segoe UI", 9, "bold"))
        st.configure("Main.TNotebook.Tab", font=("Segoe UI", 11, "bold"), padding=(14, 6))
        st.configure("Side.TNotebook", tabposition="wn")
        st.configure("Side.TNotebook.Tab", font=("Segoe UI", 10), padding=(8, 5), width=30)

    def _home(self, classes, errors):
        f = ttk.Frame(self.main_nb)
        self.main_nb.add(f, text="   \U0001f3e0 Home   ")
        t = tk.Text(f, wrap="word", font=("Segoe UI", 11), padx=24, pady=16, relief="flat",
                    background="#fbfbfd")
        sb = ttk.Scrollbar(f, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        t.pack(fill="both", expand=True)
        t.tag_configure("h", font=("Segoe UI", 15, "bold"), foreground="#1f4e79", spacing1=10, spacing3=4)
        t.tag_configure("mono", font=("Consolas", 10), foreground="#333333")
        t.tag_configure("err", font=("Consolas", 9), foreground="#b00020")
        for line in HOME_TEXT.splitlines():
            if line.startswith("## "):
                t.insert("end", line[3:] + "\n", "h")
            elif line.startswith("    "):
                t.insert("end", line + "\n", "mono")
            else:
                t.insert("end", line + "\n")
        t.insert("end", "Loaded algorithms\n", "h")
        for cat in sorted({c.category for c in classes}):
            groups: dict[str, list[str]] = {}
            for c in sorted(classes, key=lambda c: c.order):
                if c.category == cat:
                    groups.setdefault(c.group, []).append(c.name)
            t.insert("end", f"{cat}\n", "mono")
            for g, names in groups.items():
                t.insert("end", f"    {g:<26}{', '.join(names)}\n", "mono")
        if errors:
            t.insert("end", "Load errors\n", "h")
            for name, tb in errors:
                t.insert("end", f"{name}\n{tb}\n", "err")
        t.configure(state="disabled")

    def _main_changed(self, _e):
        w = self.main_nb.nametowidget(self.main_nb.select())
        if isinstance(w, ttk.Notebook):
            self._ensure_built(w)

    def _ensure_built(self, nb: ttk.Notebook):
        sel = nb.select()
        if not sel or sel in self._built or sel not in self._lazy:
            return
        self._built.add(sel)
        holder = nb.nametowidget(sel)
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            AlgoTab(holder, self._lazy[sel]).pack(fill="both", expand=True)
        finally:
            self.config(cursor="")


def main():
    warnings.filterwarnings("ignore", message="constrained_layout not applied")
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
