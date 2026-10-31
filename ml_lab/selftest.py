"""Headless smoke test: runs every registered demo to completion.
Usage:  python -m ml_lab.selftest           (all)
        python -m ml_lab.selftest kmeans    (names containing 'kmeans')"""
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from ml_lab.core.params import Values  # noqa: E402
from ml_lab.core.registry import discover  # noqa: E402


def main(filter_text=""):
    classes, errors = discover()
    for name, tb in errors:
        print(f"IMPORT ERROR {name}\n{tb}")
    failed = len(errors)
    for cls in sorted(classes, key=lambda c: (c.category, c.order)):
        if filter_text.lower() not in (cls.__module__ + cls.name).lower():
            continue
        demo = cls()
        demo.log = lambda m: None
        p = Values(**{prm.name: prm.default for prm in demo.get_params()})
        t0 = time.perf_counter()
        try:
            fig = Figure(figsize=(9, 6), layout="constrained")
            data = demo.make_data(p)
            demo.setup_figure(fig)
            demo.draw_data(data, p)
            frames, last = 0, ""
            for msg in demo.run(data, p):
                frames += 1
                last = msg
            fig.canvas.draw()
            if demo.click_to_add and demo.ax is not None:
                demo.on_click(data, 0.1, 0.2, 1, p)
                demo.setup_figure(fig)
                demo.draw_data(data, p)
            print(f"OK   {cls.category:<13}{cls.name:<40}{frames:5d} frames {time.perf_counter() - t0:6.2f}s  {last}")
        except Exception:
            failed += 1
            print(f"FAIL {cls.category:<13}{cls.name}\n{traceback.format_exc()}")
    print(f"\n{failed} failure(s)")
    return failed


def run_all_choices(filter_text=""):
    """Run every demo once per value of every Choice parameter."""
    classes, _ = discover()
    failed = 0
    for cls in sorted(classes, key=lambda c: (c.category, c.order)):
        if filter_text.lower() not in (cls.__module__ + cls.name).lower():
            continue
        demo = cls()
        demo.log = lambda m: None
        demo.fast_forward = True
        base = {prm.name: prm.default for prm in demo.get_params()}
        for prm in demo.get_params():
            if prm.kind not in ("choice", "bool"):
                continue
            for value in (prm.choices if prm.kind == "choice" else (True, False)):
                p = Values(**dict(base, **{prm.name: value}))
                try:
                    fig = Figure(figsize=(9, 6), layout="constrained")
                    data = demo.make_data(p)
                    demo.setup_figure(fig)
                    demo.draw_data(data, p)
                    for _ in demo.run(data, p):
                        pass
                except Exception:
                    failed += 1
                    print(f"FAIL {cls.name} {prm.name}={value}\n{traceback.format_exc()}")
        print(f"done {cls.name}")
    print(f"\n{failed} failure(s)")
    return failed


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--choices" in args:
        args.remove("--choices")
        sys.exit(1 if run_all_choices(args[0] if args else "") else 0)
    sys.exit(1 if main(args[0] if args else "") else 0)
