"""Patch a wandb run config value for finished runs, matched by display name.

Runs in states listed in --exclude-states are skipped (default: running, to
avoid racing with an in-progress run that will overwrite the change on its
next sync).

Dry-run by default — pass --apply to actually write the change.
"""
from __future__ import annotations

import argparse

import wandb

ENTITY = "overcooked_ai"
PROJECT = "crossenv_ued_gradient"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--display-name", required=True, help="exact run display name to match")
    parser.add_argument("--key", default="model_name", help="config key to patch")
    parser.add_argument("--value", required=True, help="new value for the config key")
    parser.add_argument("--exclude-states", nargs="*", default=["running"],
                         help="run states to skip")
    parser.add_argument("--apply", action="store_true",
                         help="actually write the change (default: dry-run only)")
    return parser.parse_args()


def main():
    args = parse_args()
    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}",
                     filters={"display_name": args.display_name})

    targets = [r for r in runs if r.state not in args.exclude_states]
    skipped = [r for r in runs if r.state in args.exclude_states]

    for r in skipped:
        print(f"skip  {r.id}  state={r.state}")
    for r in targets:
        old_value = r.config.get(args.key)
        print(f"{'apply' if args.apply else 'dry-run'}  {r.id}  state={r.state}  "
              f"{args.key}: {old_value!r} -> {args.value!r}")
        if args.apply:
            r.config[args.key] = args.value
            r.update()

    if not args.apply:
        print(f"\n{len(targets)} run(s) would be updated. Re-run with --apply to write changes.")


if __name__ == "__main__":
    main()
