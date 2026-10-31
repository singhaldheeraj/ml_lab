"""Reusable Tk widgets: tooltip, scrollable frame and the auto-generated parameter panel."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ml_lab.core.params import Param, Values


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0", relief="solid",
                 borderwidth=1, wraplength=340, font=("Segoe UI", 9)).pack(ipadx=4, ipady=2)

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, width=340, **kw):
        super().__init__(master, **kw)
        bg = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        self.canvas = tk.Canvas(self, highlightthickness=0, width=width, background=bg)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.inner.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.inner.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120), "units")


class ParamPanel(ttk.Frame):
    """Builds sliders / combo boxes / check boxes from a list of Param objects."""

    def __init__(self, master, params: list[Param], on_change=None):
        super().__init__(master)
        self.on_change = on_change
        self.vars: dict[str, tuple[Param, tk.Variable]] = {}
        self._pending: dict[str, str] = {}
        groups: dict[str, list[Param]] = {}
        for prm in params:
            groups.setdefault(prm.group, []).append(prm)
        for gname, plist in groups.items():
            lf = ttk.LabelFrame(self, text=f" {gname} parameters ")
            lf.pack(fill="x", padx=4, pady=4)
            for prm in plist:
                self._build(lf, prm)

    def _build(self, parent, prm: Param):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=4, pady=1)
        cb = lambda *_a, n=prm.name: self._changed(n)
        if prm.kind == "bool":
            var = tk.BooleanVar(value=prm.default)
            w = ttk.Checkbutton(row, text=prm.label, variable=var, command=cb)
            w.pack(anchor="w")
        else:
            lbl = ttk.Label(row, text=prm.label + ("  \u24d8" if prm.help else ""))
            lbl.pack(anchor="w")
            w = lbl
            if prm.kind in ("float", "int"):
                var = tk.DoubleVar(value=prm.default)
                tk.Scale(row, from_=prm.lo, to=prm.hi, resolution=prm.step, orient="horizontal",
                         variable=var, command=cb, highlightthickness=0, length=220,
                         font=("Segoe UI", 8)).pack(fill="x")
            elif prm.kind == "choice":
                var = tk.StringVar(value=str(prm.default))
                combo = ttk.Combobox(row, textvariable=var, state="readonly",
                                     values=[str(c) for c in prm.choices])
                combo.pack(fill="x")
                combo.bind("<<ComboboxSelected>>", cb)
            else:
                raise ValueError(f"Unknown param kind {prm.kind}")
        if prm.help:
            ToolTip(w, prm.help)
        self.vars[prm.name] = (prm, var)

    def _changed(self, name):
        if self.on_change is None:
            return
        if name in self._pending:
            self.after_cancel(self._pending[name])
        prm = self.vars[name][0]
        self._pending[name] = self.after(250, lambda: (self._pending.pop(name, None),
                                                        self.on_change(prm)))

    def values(self) -> Values:
        out = {}
        for name, (prm, var) in self.vars.items():
            v = var.get()
            if prm.kind == "int":
                v = int(round(float(v)))
            elif prm.kind == "float":
                v = float(v)
            elif prm.kind == "choice":
                v = next((c for c in prm.choices if str(c) == v), v)
            out[name] = v
        return Values(**out)

    def reset(self):
        for prm, var in self.vars.values():
            var.set(str(prm.default) if prm.kind == "choice" else prm.default)
