"""Metadata routing: one repository id in, a task, pipeline and engine out.

Evidence is consulted from strongest to weakest: an explicit choice, an
installed engine catalog that declares the source, the Hub's own task metadata,
and finally conservative name hints. No model-name conditionals live here; the
catalogs and Hub metadata carry the contract. Genuine conflicts come back as a
short list of choices rather than a guess, and recognised tasks without an
adapter are blocked instead of being sent through the text imatrix pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .registry import (HUB_TAG_TASKS, MEDIA, MULTIMODAL, RETRIEVAL, TEXT, VISION, VOICE,
                       TASKS, task_spec)

# transformers head-class conventions for a generative language model.
GENERATIVE_HEADS = ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")


@dataclass
class Route:
    repo_id: str
    task: str | None = None
    pipeline: str | None = None
    engine: str | None = None
    family: str | None = None
    reasons: list = field(default_factory=list)
    choices: list = field(default_factory=list)
    blocker: str | None = None
    metadata: dict = field(default_factory=dict, repr=False)

    @property
    def ready(self):
        return self.pipeline is not None and not self.blocker and not self.choices

    def to_dict(self):
        data = asdict(self)
        data.pop("metadata")
        data["ready"] = self.ready
        return data


def hub_metadata(repo_id):
    """Small, weight-free Hub metadata. Unavailable metadata is not fatal."""
    from huggingface_hub import HfApi
    from .. import config
    info = HfApi(token=config.TOKEN).model_info(repo_id, files_metadata=False)
    model_config = getattr(info, "config", None) or {}
    return {"pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None),
            "tags": list(getattr(info, "tags", None) or []),
            "model_type": model_config.get("model_type"),
            "architectures": list(model_config.get("architectures") or [])}


def _voice_task(backend):
    from .. import voice
    return {
        voice.TTS: "text-to-speech",
        voice.ASR: "automatic-speech-recognition",
        voice.SEP: "audio-source-separation",
        voice.VC: "voice-conversion",
        voice.ALIGN: "forced-alignment",
    }[backend.track]


def _explicit(route, task, engine, family):
    from .. import media, vision, voice
    spec = task_spec(task)
    route.task, route.reasons = spec.id, [f"task {spec.id} was selected explicitly"]
    if not spec.implemented:
        route.blocker = (f"{spec.id} is a recognised task whose {', '.join(spec.engines)} adapter "
                         f"is scheduled for {spec.milestone} and is not implemented yet"
                         + (f": {spec.note}" if spec.note else ""))
        return route
    route.pipeline = spec.pipeline
    if spec.pipeline == VOICE:
        track = {
            "text-to-speech": voice.TTS,
            "automatic-speech-recognition": voice.ASR,
            "audio-source-separation": voice.SEP,
            "voice-conversion": voice.VC,
            "forced-alignment": voice.ALIGN,
        }[spec.id]
        backend = voice.backend_for(route.repo_id, track=track, family=family)
        if backend:
            route.engine, route.family = backend.backend, backend.family
        else:
            route.engine = engine or spec.engines[0]
    elif spec.pipeline == MEDIA:
        route.engine, route.family = "stable-diffusion.cpp", family
    elif spec.pipeline in (RETRIEVAL, MULTIMODAL):
        route.engine = spec.engines[0]
    elif spec.pipeline == VISION:
        match = vision.family_for(route.repo_id, route.metadata, family=family)
        if not match or match["task"] != spec.id:
            route.pipeline = None
            route.blocker = _vision_later_route(spec.id)
            return route
        route.engine, route.family = "vision.cpp", match["family"]
    else:
        route.engine = "llama.cpp"
    if engine and route.engine and engine != route.engine:
        route.blocker = f"{route.repo_id} resolves to {route.engine} for {spec.id}, not {engine}"
    elif engine and engine not in spec.engines:
        route.blocker = f"{engine} does not provide {spec.id}; candidates: {', '.join(spec.engines)}"
    if spec.pipeline == MEDIA and not media.looks_like_media(route.repo_id, route.metadata) and route.metadata:
        route.reasons.append("Hub metadata does not look like a diffusion source; the media plan verifies components")
    return route


def route(repo_id, *, task=None, engine=None, family=None, metadata=None, fetch=None):
    """Resolve one repository. ``fetch`` is injectable for offline callers."""
    from .. import media, vision, voice
    result = Route(repo_id, metadata=dict(metadata or {}))
    if engine is not None:
        from ..runtimes import RUNTIMES
        if engine not in RUNTIMES:
            raise ValueError(f"unknown engine {engine!r}; choose from {', '.join(RUNTIMES)}")
    if task:
        if metadata is None and task_spec(task).pipeline in (MEDIA, VISION):
            result.metadata = _fetched(repo_id, fetch, result)
        return _explicit(result, task, engine, family)

    # An installed engine catalog that declares this exact source or package is
    # the strongest available contract, and needs no network round trip.
    backend = voice.backend_for(repo_id, family=family)
    if backend is not None:
        result.task, result.pipeline = _voice_task(backend), VOICE
        result.engine, result.family = backend.backend, backend.family
        result.reasons.append(f"{backend.backend} declares family {backend.family}")
        return _engine_check(result, engine)
    vision_match = vision.family_for(repo_id, {}, family=family)
    if vision_match:
        spec = task_spec(vision_match["task"])
        result.task, result.pipeline = spec.id, VISION
        result.engine, result.family = "vision.cpp", vision_match["family"]
        result.reasons.append(f"vision.cpp declares family {result.family}")
        return _engine_check(result, engine)
    if family:
        if any(c.family.casefold() == family.casefold() for c in media.capabilities()):
            result.pipeline, result.engine, result.family = MEDIA, "stable-diffusion.cpp", family
            result.reasons.append(f"stable-diffusion.cpp declares family {family}")
            return _engine_check(result, engine)
        result.blocker = f"no installed engine catalog declares family {family!r}"
        return result

    unavailable = False
    if metadata is None:
        result.metadata = _fetched(repo_id, fetch, result)
        unavailable = not result.metadata
    tag = result.metadata.get("pipeline_tag")
    library = result.metadata.get("library_name")
    architectures = result.metadata.get("architectures") or []
    tagged = HUB_TAG_TASKS.get(tag)
    evidence = f"Hub pipeline_tag is {tag}"
    if tag == "feature-extraction" and "CLIPModel" in architectures:
        tagged = "image-feature-extraction"
        evidence = "Hub CLIPModel architecture with feature-extraction task"
    # sentence-transformers publishes cross-encoders as text-classification
    # and bi-encoders without a tag; its library contract names the task.
    if library == "sentence-transformers" and tag in (None, "text-classification"):
        tagged = "reranking" if tag == "text-classification" else "embeddings"
        evidence = f"sentence-transformers {'cross-encoder' if tag else 'encoder'} library contract"
    if tagged and TASKS[tagged].pipeline == VISION:
        model_config = {"architectures": architectures,
                        "model_type": result.metadata.get("model_type")}
        match = vision.family_for(repo_id, model_config)
        if match and match["task"] == tagged:
            result.task, result.pipeline, result.engine = tagged, VISION, "vision.cpp"
            result.family = match["family"]
            result.reasons.append(
                f"Hub pipeline_tag is {tag}; vision.cpp declares family {match['family']}")
            return _engine_check(result, engine)
        result.task = tagged
        result.blocker = _vision_later_route(tagged)
        return result
    signals = []
    if tagged:
        signals.append((TASKS[tagged].pipeline, tagged, evidence))
    if media.looks_like_media(repo_id, result.metadata):
        media_task = tag if tag in media.MEDIA_TASKS else None
        signals.append((MEDIA, media_task, "diffusion metadata or naming"))
    if voice.looks_like_voice_model(repo_id) and not tag:
        signals.append((VOICE, None, "voice naming without a catalog match"))
    generative = not architectures or any(a.endswith(GENERATIVE_HEADS) for a in architectures)
    if not signals and not tag and generative and (
            library == "transformers" or "transformers" in result.metadata.get("tags", [])):
        signals.append((TEXT, "text-generation", "transformers checkpoint without a task tag"))
    if not signals and not tag and not generative:
        result.blocker = (f"architecture {', '.join(architectures)} has no generative head and the "
                          "repository declares no task; pass --task explicitly")
        return result
    if not signals and unavailable:
        # The text assessment reads the same Hub repository and reports its
        # real error (typo, gated, offline), then checks the architecture
        # against the installed converter before anything is approved.
        signals.append((TEXT, "text-generation",
                        "no task evidence offline; the text assessment verifies the architecture"))

    if not signals:
        result.blocker = ("cannot determine the task from Hub metadata"
                          + (f" (pipeline_tag {tag})" if tag else "")
                          + "; pass --task explicitly")
        return result
    implemented = [s for s in signals if s[0] is not None]
    if not implemented:
        spec = TASKS[signals[0][1]]
        result.task, result.reasons = spec.id, [signals[0][2]]
        if spec.id == "audio-to-audio" and tag == "audio-to-audio":
            result.blocker = ("audio-to-audio is ambiguous; pass --task "
                              "audio-source-separation or --task voice-conversion")
        else:
            result.blocker = (f"{repo_id} is a {spec.id} model; its {', '.join(spec.engines)} adapter "
                              f"is scheduled for {spec.milestone} and is not implemented yet. "
                              "It is not sent through the text-generation pipeline."
                              + (f" {spec.note}" if spec.note else ""))
        return result
    pipelines = list(dict.fromkeys(s[0] for s in implemented))
    if len(pipelines) > 1:
        result.reasons = [s[2] for s in implemented]
        result.choices = [{"pipeline": s[0], "task": s[1], "evidence": s[2]} for s in implemented]
        result.blocker = "conflicting task evidence; choose one with --task"
        return result
    selected = next(s for s in implemented if s[0] == pipelines[0])
    result.pipeline, result.task = selected[0], selected[1] or next(
        (s[1] for s in implemented if s[1]), None)
    result.reasons = [s[2] for s in implemented]
    result.engine = {TEXT: "llama.cpp", MEDIA: "stable-diffusion.cpp"}.get(result.pipeline)
    if result.pipeline == RETRIEVAL:
        result.engine = TASKS[result.task].engines[0]
    if result.pipeline == MULTIMODAL:
        result.engine = TASKS[result.task].engines[0]
    if result.pipeline == VISION:
        match = vision.family_for(
            repo_id, {"architectures": architectures,
                      "model_type": result.metadata.get("model_type")})
        if match and match["task"] == result.task:
            result.engine, result.family = "vision.cpp", match["family"]
        else:
            result.pipeline = None
            result.blocker = _vision_later_route(result.task)
    if result.pipeline == VOICE:
        result.reasons.append("no installed voice catalog declares this source; the voice plan runs the fork hunt")
    return _engine_check(result, engine)


def _vision_later_route(task):
    later = {
        "mask-generation": "sam3cpp is the later route in 0.5.8",
        "image-segmentation": "ultralytics-ggml/rf-detr.cpp is the later route in 0.5.7",
        "depth-estimation": "depth-anything.cpp is the later route in 0.5.8",
    }.get(task, "no later native route is scheduled")
    return f"no vision.cpp family matches {task}; {later}"


def _engine_check(result, engine):
    if engine and result.engine and engine != result.engine:
        result.blocker = f"{result.repo_id} resolves to {result.engine}, not {engine}"
    elif engine and not result.engine:
        result.engine = engine
    return result


def _fetched(repo_id, fetch, result):
    try:
        return (fetch or hub_metadata)(repo_id)
    except Exception as error:
        result.reasons.append(f"Hub metadata unavailable ({type(error).__name__}); name evidence only")
        return {}


def is_short_name(name):
    """A research-report shorthand rather than a repository or package id."""
    return "/" not in name and ":" not in name
