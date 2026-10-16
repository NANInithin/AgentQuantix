"""Media tools share the CLI's exact plan and publication boundaries."""
from .. import media
from ..pipeline import media_release


def _tool(name, description, properties, required=()):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": list(required)}}


MODEL = {"type": "string"}
TOOLS = [
    _tool("inspect_media_model", "Inspect repository metadata and the pinned engine catalog. Read-only; no model weights or builds.",
          {"model": MODEL}, ["model"]),
    _tool("plan_media_release", "Read-only exact media plan: source revisions, components, native policies, licenses, estimates, evaluator, targets and blocked questions. Image/video models never use the text converter.",
          {"model": MODEL, "family": MODEL, "target_repo": MODEL, "task": MODEL,
           "quants": {"type": "array", "items": MODEL}, "sequential": {"type": "boolean"},
           "compute_backend": MODEL, "contract": {"type": "object"}, "quality_profile": {"type": "object"},
           "conditioning": {"type": "object"},
           "hunt_forks": {"type": "boolean"}}, ["model"]),
    _tool("start_media_release", "Execute only the exact plan shown to and approved by the user. Runs real native inference and quality gates. Publication still requires a separate human image/video/audio review.",
          {"plan": {"type": "object"}, "user_approved": {"type": "boolean"},
           "publish": {"type": "boolean"}}, ["plan", "user_approved"]),
    _tool("record_media_review", "Record the user's explicit review of these exact candidate artifacts. Never invent a human review. Video requires playback and frame inspection; audio-video requires listening too.",
          {"run_id": MODEL, "quant": MODEL, "reviewer": MODEL, "accepted": {"type": "boolean"},
           "notes": MODEL, "watched_video": {"type": "boolean"}, "listened_audio": {"type": "boolean"}},
          ["run_id", "quant", "reviewer", "accepted"]),
    _tool("publish_media_release", "Publish a previously approved, natively validated and human-reviewed media candidate in one Hub commit, then verify remote sizes and SHA-256 hashes.",
          {"run_id": MODEL, "quant": MODEL, "user_approved": {"type": "boolean"}},
          ["run_id", "quant", "user_approved"]),
]
NAMES = {t["name"] for t in TOOLS}


def call(name, arguments):
    if name == "inspect_media_model":
        metadata = media.inspect_repository(arguments["model"])
        return {"metadata": metadata, "likely_media": media.looks_like_media(arguments["model"], metadata),
                "capabilities": [c.__dict__ for c in media.capabilities()]}
    if name == "plan_media_release":
        args = dict(arguments)
        return media.plan_release(args.pop("model"), **args)
    if name == "start_media_release":
        return media_release.run_release(arguments["plan"], user_approved=arguments["user_approved"],
                                          publish=arguments.get("publish", False))
    if name == "record_media_review":
        return media_release.record_review(arguments["run_id"], arguments["quant"], arguments["reviewer"],
            arguments["accepted"], notes=arguments.get("notes", ""), watched_video=arguments.get("watched_video", False),
            listened_audio=arguments.get("listened_audio", False))
    if name == "publish_media_release":
        if arguments.get("user_approved") is not True:
            raise media.MediaValidationError("publication requires user_approved=true after the media review")
        return media_release.publish_run(arguments["run_id"], arguments["quant"])
    raise ValueError(f"unknown media tool: {name}")
