"""Read-only checkpoint/config audit; write reports beside this script."""

import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import re

import numpy as np
from numpy.core.multiarray import _reconstruct
from PIL import Image, ImageDraw, ImageFont
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def numpy_from_jax(reconstruct, args, state, *metadata):
    # Inspect stored numerical arrays without importing JAX or allocating a GPU.
    array = reconstruct(*args)
    array.__setstate__(state)
    return array


class ArrayReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("jax._src.array", "_reconstruct_array"):
            return numpy_from_jax
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name == "_reconstruct":
                return _reconstruct
            if name == "scalar":
                return np.core.multiarray.scalar
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if (module, name) == ("flax.core.frozen_dict", "FrozenDict"):
            return dict
        raise ValueError(f"Unsupported stored class: {module}.{name}")


def arrays(value, prefix=""):
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            yield from arrays(child, prefix + "/" + key)
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from arrays(child, prefix + "/" + str(i))
    elif isinstance(value, np.ndarray):
        yield prefix, value


configs = []
for path in sorted((ROOT / "outputs").glob("*/*/.hydra/config.yaml")):
    config = yaml.safe_load(path.read_text())
    if config.get("ENV_NAME") == "ToyCoopNoPink":
        stamp = datetime.strptime("/".join(path.parts[-4:-2]), "%Y-%m-%d/%H-%M-%S")
        configs.append((stamp, path, config))

rows = []
for family in ("ippo", "e3t", "idaac"):
    root = ROOT / "ckpts" / family / "ToyCoopNoPink" / "modified_wall"
    for path in sorted(root.rglob("*.pkl")):
        rel = path.relative_to(ROOT)
        if "cec_popart_layout_eval" in path.parts or path.name == "resume_ckpt.pkl":
            continue
        match = re.fullmatch(r"seed(\d+)_(ckpt0_.*|best_e3t|progress_\d+)\.pkl", path.name)
        if not match:
            continue
        seed = int(match[1])
        variant = path.relative_to(root).parts[0]
        envs = 256 if "numenv256" in variant else 64
        algo = "IDAAC+CEC" if family == "idaac" else (
            "CEC" if "cec_layout_eval" in path.parts else family.upper()
        )
        layout = "mixed" if variant.startswith("mixed") else (
            "wall_a" if variant.startswith("wall_a") else "empty"
        )
        stamp = datetime.strptime(path.parent.name, "lr-%Y%m%d-%H%M%S")
        candidates = [(abs((stamp - t).total_seconds()), p, c) for t, p, c in configs
                      if int(c.get("SEED", -1)) == seed
                      and c.get("NUM_ENVS") == envs
                      and c.get("map_name") == layout
                      and abs((stamp - t).total_seconds()) <= 5]
        candidates.sort(key=lambda item: item[0])
        cfg_path, cfg = (candidates[0][1], candidates[0][2]) if candidates else (None, {})
        row = dict(algorithm=algo, num_envs=envs, layout=layout, seed=seed,
                   kind="final" if match[2].startswith("ckpt0") else match[2],
                   path=str(rel), bytes=path.stat().st_size,
                   config_path=str(cfg_path.relative_to(ROOT)) if cfg_path else "MISSING",
                   config_env=cfg.get("ENV_NAME", ""),
                   config_model=cfg.get("model_name", ""),
                   random_reset=cfg.get("ENV_KWARGS", {}).get("random_reset", ""),
                   check_held_out=cfg.get("ENV_KWARGS", {}).get("check_held_out", ""),
                   heldout_random=cfg.get("TOY_HELDOUT_NUM", ""),
                   config_layouts=json.dumps(cfg.get("layout_names", [])),
                   total_timesteps=cfg.get("TOTAL_TIMESTEPS", ""),
                   rollout_steps=cfg.get("NUM_STEPS", ""),
                   xp_partner=cfg.get("XP_KWARGS", {}).get("partner_seed", ""))
        try:
            with path.open("rb") as stream:
                checkpoint = ArrayReader(stream).load()
            leaves = list(arrays(checkpoint["params"]))
            digest = hashlib.sha256()
            for key, array in leaves:
                digest.update(key.encode())
                digest.update(str((array.shape, array.dtype)).encode())
                digest.update(array.tobytes())
            row.update(
                status="OK" if leaves and all(np.isfinite(a).all() for _, a in leaves) else "NONFINITE_OR_EMPTY",
                update_steps=int(checkpoint.get("update_steps", -1)),
                parameter_count=sum(a.size for _, a in leaves),
                conv_shapes=json.dumps({k: a.shape for k, a in leaves if a.ndim == 4}),
                params_sha256=digest.hexdigest(),
            )
        except Exception as error:
            row.update(status=f"ERROR: {error}", update_steps="", parameter_count="",
                       conv_shapes="", params_sha256="")
        rows.append(row)

with (OUT / "checkpoints.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

finals = [r for r in rows if r["kind"] == "final"]
summary = []
for group in sorted({(r["algorithm"], r["num_envs"], r["layout"]) for r in finals}):
    selected = [r for r in finals if (r["algorithm"], r["num_envs"], r["layout"]) == group]
    target = [r for r in selected if r["seed"] in range(6)]
    summary.append(dict(algorithm=group[0], num_envs=group[1], layout=group[2],
                        seeds=sorted(r["seed"] for r in target),
                        extra_seeds=sorted(r["seed"] for r in selected if r["seed"] not in range(6)),
                        configs_found=sum(r["config_path"] != "MISSING" for r in target),
                        valid_arrays=sum(r["status"] == "OK" for r in target),
                        distinct_parameters=len({r["params_sha256"] for r in target}),
                        updates=sorted({r["update_steps"] for r in target})))
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print("Missing historical configs:", [r["path"] for r in finals if r["config_path"] == "MISSING"])
print("Array errors:", [r for r in rows if r["status"] != "OK"])

# Render the exact literal layouts, without importing training code.
tree = ast.parse((ROOT / "jaxmarl/environments/toy_coop/modified_wall_toy_coop.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModifiedWallToyCoop")
layouts = next(ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "LAYOUTS" for t in n.targets))
canvas = Image.new("RGB", (1120, 670), "white")
draw = ImageDraw.Draw(canvas)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(font_path, 20)
title_font = ImageFont.truetype(font_path, 28)
draw.text((32, 20), "ToyCoopNoPink: fixed initial states", fill="#17202a", font=title_font)
for index, name in enumerate(("empty", "wall_a")):
    ox, oy, cell = 70 + index * 560, 128, 82
    draw.text((ox, 72), name, fill="#17202a", font=title_font)
    agent = 0
    for y, line in enumerate(layouts[name]):
        draw.text((ox - 28, oy + y * cell + 27), str(y), fill="#667078", font=font)
        for x, token in enumerate(line):
            if y == 0:
                draw.text((ox + x * cell + 33, oy - 29), str(x), fill="#667078", font=font)
            box = (ox + x * cell, oy + y * cell, ox + (x + 1) * cell, oy + (y + 1) * cell)
            fill = {"B": "#515a61", "G": "#9adb91"}.get(token, "#f7f9fa")
            draw.rectangle(box, fill=fill, outline="#8b959e", width=2)
            if token == "G":
                draw.text((box[0] + 32, box[1] + 26), "G", fill="#174d26", font=title_font)
            if token == "A":
                draw.ellipse((box[0] + 10, box[1] + 10, box[2] - 10, box[3] - 10),
                             fill=("#d74444", "#277bd1")[agent])
                draw.text((box[0] + 23, box[1] + 28), f"A{agent}", fill="white", font=font)
                agent += 1
draw.text((50, 568), "A0 = agent_0     A1 = agent_1     G = shared goal     Gray = wall", fill="#17202a", font=font)
draw.text((50, 605), "Both agents must occupy different goals. Coordinates: (x, y), origin at top-left.", fill="#17202a", font=font)
canvas.save(OUT / "fixed_maps.png")
