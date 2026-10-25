"""CLI surface for explicit media plans, native runs, review and publication."""
import json
from pathlib import Path

from . import media, sysprobe
from .pipeline import media_release


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def _components(values):
    members = []
    for value in values or []:
        if "=" not in value:
            raise media.MediaValidationError("use --component role=path or role=org/repo::path")
        role, source = value.split("=", 1)
        member = {"role": role, "quantizable": role not in ("vae", "audio_vae", "llm_vision", "config", "tokenizer", "scheduler"),
                  "precision": "source"}
        if "::" in source:
            member["source_repo"], source = source.split("::", 1)
        member["source_path"] = source
        members.append(member)
    return members or None


def command(args):
    try:
        action = args.media_command
        if action == "backends":
            if args.build:
                binary, provenance = media_release.ensure_backend(args.compute_backend)
                result = {"binary": str(binary), "provenance": provenance,
                          "converter": media_release.converter_types(binary=binary)}
            else:
                result = {"compute_toolchains": sysprobe.compute_backends(),
                          "capabilities": [c.__dict__ for c in media.capabilities()],
                          "converter": media_release.converter_types()}
        elif action == "inspect":
            result = media.inspect_repository(args.repo)
        elif action == "review":
            result = media_release.record_review(args.run_id, args.accept or args.reject, args.reviewer,
                bool(args.accept), notes=args.notes, watched_video=args.watched_video, listened_audio=args.listened_audio)
        elif action == "publish":
            result = media_release.publish_run(args.run_id, args.quant)
        elif action == "verify":
            result = {"problems": media_release.load_bundle(args.manifest).problems(Path(args.manifest).parent)}
        else:
            def make_plan():
                return media.plan_release(
                args.repo, target_repo=args.target, quants=args.quant, family=args.family, task=args.task,
                components=_components(args.component),
                compute_backend=args.compute_backend, sequential=args.sequential, hunt_forks=not args.no_fork_hunt,
                contract=_read(args.contract), quality_profile=_read(args.profile),
                conditioning={key: value for key, value in {
                    "init_image": args.init_image, "end_image": args.end_image,
                    "ref_image": args.ref_image, "ref_audio": args.ref_audio}.items() if value},
                settings={key: getattr(args, key) for key in ("width", "height", "steps", "seed", "frames", "fps", "sampler", "scheduler", "cfg_scale")
                          if getattr(args, key) is not None})
            saved = _read(args.plan) if getattr(args, "plan", None) else None
            plan = saved if saved is not None else make_plan()
            if action == "run" and saved is not None:
                media.validate_plan(plan, check_runtime=False)
            if action == "run" and not args.dry_run and not plan["blockers"]:
                from . import runtime_control
                revision = ({"revision": plan["engine_revision"], "repository": "leejet/stable-diffusion.cpp"}
                            if saved and not saved.get("runtime_snapshot") else None)
                handle = runtime_control.ensure_runtime(
                    "stable-diffusion.cpp", compute_backend=plan["compute_backend"],
                    frozen=saved.get("runtime_snapshot") if saved else None, revision=revision)
                if handle and saved is None:
                    plan = make_plan()
            if getattr(args, "output", None):
                media.atomic_json(args.output, plan)
            print(json.dumps(plan, indent=2))
            if action == "plan" or args.dry_run:
                return 1 if plan["blockers"] else 0
            if plan["blockers"]:
                raise media.MediaValidationError("media plan is blocked: " + "; ".join(plan["blockers"]))
            if args.repo and args.repo != plan["repo_id"]:
                raise media.MediaValidationError("repository differs from the saved approved plan")
            if not args.yes and input("Build and validate this exact media plan? [y/N] ").strip().lower() not in ("y", "yes"):
                return 0
            result = media_release.run_release(plan, user_approved=True, publish=args.publish,
                                                preserve_failed=args.preserve_failed)
        print(json.dumps(result, indent=2))
        if result.get("problems") or any(c.get("rejected") or c.get("status") == "quality-rejected" for c in result.get("candidates", [])):
            return 1
        if getattr(args, "publish", False) and any(not c.get("published") for c in result.get("candidates", []) if c.get("quant") != "f16"):
            return 1
        return 0
    except (media.MediaValidationError, OSError, ValueError) as error:
        print(f"media: {error}")
        return 1


def add_parser(subparsers):
    parser = subparsers.add_parser("media", help="pinned stable-diffusion.cpp component bundles")
    commands = parser.add_subparsers(dest="media_command", required=True)
    backends = commands.add_parser("backends", help="inspect loader capabilities or build and probe the engine")
    backends.add_argument("--build", action="store_true")
    backends.add_argument("--compute-backend", default="auto")
    backends.set_defaults(func=command)
    inspect = commands.add_parser("inspect", help="read repository metadata without downloading weights")
    inspect.add_argument("repo")
    inspect.set_defaults(func=command)
    for name in ("plan", "run"):
        item = commands.add_parser(name)
        item.add_argument("repo")
        item.add_argument("--target")
        item.add_argument("--quant", action="append")
        item.add_argument("--family")
        item.add_argument("--task", choices=sorted(media.MEDIA_TASKS))
        item.add_argument("--contract", help="reviewed component source contract JSON")
        item.add_argument("--component", action="append", help="resolve an ambiguous member: role=path or role=org/repo::path")
        item.add_argument("--init-image", nargs="?", const="fixtures/reference.png", help="use a packaged initial-frame fixture")
        item.add_argument("--end-image", nargs="?", const="fixtures/last.png", help="use a packaged last-frame fixture")
        item.add_argument("--ref-image", nargs="?", const="fixtures/reference.png", help="use a packaged reference-image fixture")
        item.add_argument("--ref-audio", nargs="?", const="fixtures/reference.wav", help="use the packaged licensed reference-audio fixture")
        for setting in ("width", "height", "steps", "seed", "frames", "fps"):
            item.add_argument("--" + setting, type=int, default=None)
        item.add_argument("--sampler")
        item.add_argument("--scheduler")
        item.add_argument("--cfg-scale", type=float)
        item.add_argument("--profile", help="calibrated family/task quality profile JSON")
        item.add_argument("--compute-backend", default="auto")
        item.add_argument("--sequential", action="store_true")
        item.add_argument("--no-fork-hunt", action="store_true")
        item.add_argument("--output", help="save the exact reviewable plan")
        if name == "run":
            item.add_argument("--plan", help="execute this saved plan without repricing or changing its sources")
            item.add_argument("--yes", "-y", action="store_true")
            item.add_argument("--dry-run", action="store_true")
            item.add_argument("--publish", action="store_true", help="publish only candidates with an existing matching human review")
            item.add_argument("--preserve-failed", action="store_true")
        item.set_defaults(func=command)
    review = commands.add_parser("review")
    review.add_argument("run_id")
    decision = review.add_mutually_exclusive_group(required=True)
    decision.add_argument("--accept", metavar="QUANT")
    decision.add_argument("--reject", metavar="QUANT")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", default="")
    review.add_argument("--watched-video", action="store_true", help="confirm playback and extracted-frame inspection")
    review.add_argument("--listened-audio", action="store_true")
    review.set_defaults(func=command)
    publish = commands.add_parser("publish", help="publish an already validated and human-reviewed candidate")
    publish.add_argument("run_id")
    publish.add_argument("--quant", required=True)
    publish.set_defaults(func=command)
    verify = commands.add_parser("verify", help="verify downloaded artifacts against their manifest")
    verify.add_argument("manifest")
    verify.set_defaults(func=command)
