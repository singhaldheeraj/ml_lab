"""One tab = one algorithm: controls on the left, live plot on the right."""
from __future__ import annotations

import importlib
import inspect
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
import traceback
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ml_lab.gui.widgets import ParamPanel, ScrollableFrame, ToolTip


class AlgoTab(ttk.Frame):
    def __init__(self, master, algo_cls):
        super().__init__(master)
        self.algo_cls = algo_cls
        self._job = None
        self._gen = None
        self._paused = False
        self.data = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        for w in self.winfo_children():
            w.destroy()
        self.algo = self.algo_cls()
        self.algo.log = self.log

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ScrollableFrame(paned, width=330)
        right = ttk.PanedWindow(paned, orient="vertical")
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        L = left.inner

        ttk.Label(L, text=self.algo.name, style="Title.TLabel", wraplength=300).pack(
            anchor="w", padx=6, pady=(8, 0))
        ttk.Label(L, text=f"{self.algo.category}  \u203a  {self.algo.group}",
                  style="Sub.TLabel").pack(anchor="w", padx=6, pady=(0, 4))

        bf = ttk.Frame(L)
        bf.pack(fill="x", padx=4, pady=4)
        buttons = [("\u25b6 Run", self.run, "Accent.TButton", "Train from scratch with animation"),
                   ("\u23ef Pause", self.pause, "TButton", "Pause / resume the animation"),
                   ("\u23ed Step", self.step, "TButton", "Advance exactly one frame"),
                   ("\u23e9 Finish", self.finish, "TButton", "Skip to the end result"),
                   ("\U0001f3b2 New data", self.new_data, "TButton", "Regenerate the dataset"),
                   ("\u21ba Defaults", self.reset_params, "TButton", "Reset all parameters")]
        for i, (txt, cmd, sty, tip) in enumerate(buttons):
            b = ttk.Button(bf, text=txt, command=cmd, style=sty)
            b.grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=2)
            ToolTip(b, tip)
        for c in range(3):
            bf.columnconfigure(c, weight=1)

        sf = ttk.Frame(L)
        sf.pack(fill="x", padx=8)
        ttk.Label(sf, text="Animation delay (ms)").pack(anchor="w")
        self.delay = tk.IntVar(value=30)
        tk.Scale(sf, from_=1, to=800, orient="horizontal", variable=self.delay,
                 highlightthickness=0, font=("Segoe UI", 8)).pack(fill="x")

        self.panel = ParamPanel(L, self.algo.get_params(), on_change=self._param_changed)
        self.panel.pack(fill="x", padx=2, pady=2)

        if self.algo.click_help:
            ttk.Label(L, text="\U0001f5b1 " + self.algo.click_help, wraplength=300,
                      style="Hint.TLabel").pack(anchor="w", padx=8, pady=4)

        dev = ttk.LabelFrame(L, text=" Developer ")
        dev.pack(fill="x", padx=6, pady=6)
        src = inspect.getsourcefile(self.algo_cls) or "?"
        ttk.Button(dev, text="\U0001f4dd Open source code", command=self.open_code).pack(
            fill="x", padx=4, pady=2)
        ttk.Button(dev, text="\U0001f504 Reload code (after editing)", command=self.reload_code).pack(
            fill="x", padx=4, pady=2)
        try:
            shown = os.path.relpath(src)
        except ValueError:
            shown = src
        ttk.Label(dev, text=shown, wraplength=290, style="Hint.TLabel").pack(anchor="w", padx=4, pady=2)

        # ---------------- right: plot + explanation
        plotf = ttk.Frame(right)
        right.add(plotf, weight=4)
        self.fig = Figure(figsize=(9, 6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotf)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plotf, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self.status_var = tk.StringVar(value="Ready")
        self.status_lbl = ttk.Label(plotf, textvariable=self.status_var, style="Status.TLabel",
                                    anchor="w")
        self.status_lbl.pack(side="bottom", fill="x")

        self.bottom = ttk.Notebook(right)
        right.add(self.bottom, weight=1)
        self.info_txt = self._text_tab(self.bottom, "\U0001f4d6 Explanation")
        self.log_txt = self._text_tab(self.bottom, "\U0001f4cb Log")
        self._write_info(self.algo.info)

        self.after(50, self.new_data)

    @staticmethod
    def _text_tab(nb, title):
        f = ttk.Frame(nb)
        nb.add(f, text=title)
        t = tk.Text(f, wrap="word", height=10, font=("Segoe UI", 10), padx=10, pady=6,
                    relief="flat", background="#fbfbfd")
        sb = ttk.Scrollbar(f, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        t.pack(fill="both", expand=True)
        t.tag_configure("h", font=("Segoe UI", 11, "bold"), foreground="#1f4e79", spacing1=6)
        t.tag_configure("mono", font=("Consolas", 10), foreground="#333333")
        t.tag_configure("err", foreground="#b00020", font=("Consolas", 9))
        return t

    def _write_info(self, text):
        t = self.info_txt
        t.configure(state="normal")
        t.delete("1.0", "end")
        for line in inspect.cleandoc(text).splitlines():
            if line.startswith("## "):
                t.insert("end", line[3:] + "\n", "h")
            elif line.startswith("    "):
                t.insert("end", line + "\n", "mono")
            else:
                t.insert("end", line + "\n")
        t.configure(state="disabled")

    # ------------------------------------------------------------------ helpers
    def log(self, msg, tag=None):
        self.log_txt.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n", tag)
        self.log_txt.see("end")

    def status(self, msg, error=False):
        self.status_var.set(msg)
        self.status_lbl.configure(style="Error.TLabel" if error else "Status.TLabel")

    def _error(self):
        tb = traceback.format_exc()
        self.log(tb, "err")
        self.status("\u26a0 Error - see the Log tab", error=True)
        self.bottom.select(1)

    def p(self):
        return self.panel.values()

    # ------------------------------------------------------------------ actions
    def _cancel(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def new_data(self):
        self._cancel()
        self._gen = None
        try:
            p = self.p()
            self.data = self.algo.make_data(p)
            self.algo.setup_figure(self.fig)
            self.algo.draw_data(self.data, p)
            self.canvas.draw_idle()
            self.status("New data ready - press \u25b6 Run")
        except Exception:
            self._error()

    def reset_params(self):
        self.panel.reset()
        self.new_data()

    def _param_changed(self, prm):
        if prm.group == "Data":
            self.new_data()
        elif self.algo.instant:
            self.finish(restart=True)

    def _start(self):
        self._cancel()
        self._gen = None
        self._paused = False
        if self.data is None:
            self.new_data()
        try:
            p = self.p()
            self.algo.setup_figure(self.fig)
            g = self.algo.run(self.data, p)
            if not inspect.isgenerator(g):
                self.canvas.draw_idle()
                return False
            self._gen = g
            return True
        except Exception:
            self._error()
            return False

    def _advance(self):
        try:
            msg = next(self._gen)
        except StopIteration:
            self._gen = None
            self.canvas.draw_idle()
            if not self.status_var.get().startswith("\u2714"):
                self.status("\u2714 " + self.status_var.get())
            return False
        except Exception:
            self._gen = None
            self._error()
            return False
        if msg:
            self.status(str(msg))
        self.canvas.draw()
        return True

    def _tick(self):
        self._job = None
        if self._gen is None or self._paused:
            return
        if self._advance():
            self._job = self.after(max(1, int(self.delay.get())), self._tick)

    def run(self):
        if self._start():
            self._tick()

    def pause(self):
        if self._gen is None:
            return
        self._paused = not self._paused
        if self._paused:
            self._cancel()
            self.status("\u23f8 Paused - press Pause again to resume, or Step")
        else:
            self._tick()

    def step(self):
        self._cancel()
        if self._gen is None and not self._start():
            return
        self._paused = True
        self._advance()

    def finish(self, restart=False):
        self._cancel()
        if (self._gen is None or restart) and not self._start():
            return
        self.config(cursor="watch")
        self.update_idletasks()
        msg = None
        self.algo.fast_forward = True
        try:
            while True:
                msg = next(self._gen) or msg
        except StopIteration:
            pass
        except Exception:
            self._error()
            msg = None
        finally:
            self.algo.fast_forward = False
            self._gen = None
            self.config(cursor="")
        if msg:
            self.status("\u2714 " + str(msg))
        self.canvas.draw_idle()

    def _on_click(self, event):
        if str(self.toolbar.mode) or event.inaxes is None or event.inaxes is not self.algo.ax:
            return
        if self.data is None or event.xdata is None:
            return
        try:
            changed = self.algo.on_click(self.data, event.xdata, event.ydata, event.button, self.p())
        except Exception:
            self._error()
            return
        if not changed:
            self.canvas.draw_idle()
            return
        if self.algo.rerun_on_click:
            self.finish(restart=True)
        else:
            self._cancel()
            self._gen = None
            self.algo.setup_figure(self.fig)
            self.algo.draw_data(self.data, self.p())
            self.canvas.draw_idle()
            self.status("Data edited - press \u25b6 Run")

    # ------------------------------------------------------------------ developer
    def open_code(self):
        path = inspect.getsourcefile(self.algo_cls)
        try:
            if shutil.which("code"):
                subprocess.Popen(["code", path], shell=(os.name == "nt"))
            elif os.name == "nt":
                subprocess.Popen(["notepad.exe", path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-t", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            self._error()

    def reload_code(self):
        self._cancel()
        try:
            mod = importlib.reload(sys.modules[self.algo_cls.__module__])
            self.algo_cls = getattr(mod, self.algo_cls.__name__)
            self._build()
            self.status(f"\U0001f504 Reloaded {mod.__name__}")
        except Exception:
            self._error()
