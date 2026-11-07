"""Filterable, scrollable algorithm catalog with lazily constructed demo views."""
import tkinter as tk
from tkinter import ttk

from ml_lab.gui.algo_tab import AlgoTab


class SearchCatalog(ttk.Frame):
    def __init__(self, master, classes, title="Searching algorithms"):
        super().__init__(master)
        self.classes = sorted(classes, key=lambda cls: (cls.order, cls.name))
        self._classes = {cls.__module__: cls for cls in self.classes}
        self._views = {}
        self._active = None
        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True)
        sidebar = ttk.Frame(panes, width=290, padding=8)
        self.content = ttk.Frame(panes)
        self.empty = ttk.Label(self.content, text="No matching algorithms", anchor="center", style="Sub.TLabel")
        panes.add(sidebar, weight=0)
        panes.add(self.content, weight=1)
        ttk.Label(sidebar, text=title, style="Title.TLabel").pack(anchor="w", pady=(2, 8))
        self.query = tk.StringVar()
        ttk.Label(sidebar, text="Filter").pack(anchor="w")
        entry = ttk.Entry(sidebar, textvariable=self.query)
        entry.pack(fill="x", pady=(2, 8))
        tree_frame = ttk.Frame(sidebar)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, show="tree", selectmode="browse")
        self.tree.column("#0", width=275, minwidth=200, stretch=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.count = ttk.Label(sidebar, style="Sub.TLabel")
        self.count.pack(anchor="w", pady=(6, 0))
        self.tree.tag_configure("group", font=("Segoe UI", 9, "bold"))
        self.tree.bind("<<TreeviewSelect>>", self._select)
        self.query.trace_add("write", lambda *_: self._populate())
        self._populate()

    def _populate(self):
        selected = self.tree.selection()
        current = selected[0] if selected else None
        self.tree.delete(*self.tree.get_children())
        query = self.query.get().strip().casefold()
        groups, matches = {}, []
        for cls in self.classes:
            if query not in (cls.name + " " + cls.group + " " + cls.info).casefold():
                continue
            if cls.group not in groups:
                groups[cls.group] = self.tree.insert("", "end", text=cls.group, open=True, tags=("group",))
            self.tree.insert(groups[cls.group], "end", iid=cls.__module__, text=cls.name)
            matches.append(cls.__module__)
        self.count.configure(text=f"{len(matches)} / {len(self.classes)} algorithms")
        if matches:
            self.empty.pack_forget()
            target = current if current in matches else matches[0]
            self.tree.selection_set(target)
            self.tree.see(target)
        else:
            if self._active is not None:
                previous = self._views[self._active]
                previous._cancel()
                previous._paused = True
                previous.pack_forget()
                self._active = None
            self.empty.pack(fill="both", expand=True)

    def _select(self, _event=None):
        selection = self.tree.selection()
        if not selection or selection[0] not in self._classes:
            return
        key = selection[0]
        if key == self._active:
            return
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            if key not in self._views:
                self._views[key] = AlgoTab(self.content, self._classes[key])
            if self._active is not None:
                previous = self._views[self._active]
                previous._cancel()
                previous._paused = True
                previous.pack_forget()
            self._views[key].pack(fill="both", expand=True)
            self._active = key
        finally:
            self.configure(cursor="")