"""Task and capability registry shared by routing, planning and every harness.

A task names what a model does; a pipeline is the AgentQuantix adapter that can
release it; an engine is the native runtime that executes it. They are kept
separate because they change independently: a task can be recognised long
before an adapter exists for it, and one engine serves several tasks.

Capability facts (load, convert, quantize, package, platform, acceptance) are
independent of each other. A catalog entry or a valid GGUF header is never
recorded as conversion support or as accepted native inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

TEXT, VOICE, MEDIA, RETRIEVAL, MULTIMODAL, VISION = (
    "text", "voice", "media", "retrieval", "multimodal", "vision"
)
PIPELINES = (TEXT, VOICE, MEDIA, RETRIEVAL, MULTIMODAL, VISION)


@dataclass(frozen=True)
class TaskSpec:
    id: str
    modality: str
    pipeline: str | None          # None: recognised, adapter not implemented
    engines: tuple[str, ...]
    hub_tags: tuple[str, ...] = ()
    milestone: str = ""
    review: str = "none"          # human gate required before publication
    publication: str = ""
    note: str = ""

    @property
    def implemented(self):
        return self.pipeline is not None

    def to_dict(self):
        return {**asdict(self), "implemented": self.implemented}


_MEDIA_PUBLICATION = ("one atomic Hub commit per reviewed candidate after native "
                      "inference, automated quality gates and human review")
_MEDIA_TASKS = {
    "text-to-image": ("image", "visual"), "image-to-image": ("image", "visual"),
    "image-editing": ("image", "visual"), "text-to-video": ("video", "playback"),
    "image-to-video": ("video", "playback"), "first-last-to-video": ("video", "playback"),
    "reference-to-video": ("video", "playback"),
    "text-to-audio-video": ("audio-video", "playback+listening"),
    "image-to-audio-video": ("audio-video", "playback+listening"),
    "first-last-to-audio-video": ("audio-video", "playback+listening"),
    "reference-to-audio-video": ("audio-video", "playback+listening"),
}
_MEDIA_HUB_TAGS = {"text-to-image": ("text-to-image",), "image-to-image": ("image-to-image",),
                   "text-to-video": ("text-to-video",), "image-to-video": ("image-to-video",)}

TASKS = {spec.id: spec for spec in (
    TaskSpec("text-generation", "text", TEXT, ("llama.cpp",),
             ("text-generation", "text2text-generation", "image-text-to-text", "any-to-any"),
             "0.4", "none",
             "each GGUF is uploaded as soon as it is built, then verified and carded"),
    TaskSpec("text-to-speech", "audio", VOICE, ("llama.cpp", "audio.cpp"),
             ("text-to-speech",), "0.4", "listening",
             "one atomic bundle commit of accepted quants after ASR round-trip and listening review"),
    TaskSpec("automatic-speech-recognition", "audio", VOICE, ("whisper.cpp", "audio.cpp"),
             ("automatic-speech-recognition",), "0.4", "none",
             "one atomic bundle commit after the fixed-corpus WER/CER regression gate"),
    TaskSpec("audio-source-separation", "audio", VOICE, ("audio.cpp",), (),
             "0.5.5", "none",
             "atomic bundle of passing precisions after native task gates"),
    TaskSpec("voice-conversion", "audio", VOICE, ("audio.cpp",), (),
             "0.5.5", "none",
             "atomic bundle of passing precisions after native task gates"),
    TaskSpec("forced-alignment", "audio", VOICE, ("audio.cpp",), (),
             "0.5.5", "none",
             "atomic bundle of passing precisions after native task gates"),
    *(TaskSpec(task, modality, MEDIA, ("stable-diffusion.cpp",), _MEDIA_HUB_TAGS.get(task, ()),
               "0.5.0 (experimental)", review, _MEDIA_PUBLICATION)
      for task, (modality, review) in _MEDIA_TASKS.items()),
    # Recognised so they are routed honestly instead of falling into the text
    # imatrix pipeline. Each gains a pipeline when its milestone is accepted.
    TaskSpec("embeddings", "retrieval", RETRIEVAL, ("llama.cpp",),
             ("feature-extraction", "sentence-similarity"), "0.5.3", "none",
             "one atomic bundle commit of passing quants after native inference parity and retrieval-quality gates"),
    TaskSpec("reranking", "retrieval", RETRIEVAL, ("llama.cpp",), ("text-ranking",), "0.5.3", "none",
             "one atomic bundle commit of passing quants after native inference parity and retrieval-quality gates"),
    TaskSpec("zero-shot-image-classification", "retrieval", RETRIEVAL, ("clip.cpp",),
             ("zero-shot-image-classification",), "0.5.3", "none",
             "one atomic bundle commit of passing quants after native inference parity and retrieval-quality gates"),
    TaskSpec("image-feature-extraction", "retrieval", RETRIEVAL, ("clip.cpp",),
             ("image-feature-extraction",), "0.5.3", "none",
             "one atomic bundle commit of passing quants after native inference parity and retrieval-quality gates"),
    TaskSpec("multimodal-understanding", "multimodal", MULTIMODAL, ("llama.cpp",),
             ("visual-question-answering", "document-question-answering", "image-to-text",
              "video-text-to-text", "audio-text-to-text"), "0.5.4", "none",
             "one atomic bundle commit of passing text quants plus the F16 projector after native multimodal task gates"),
    TaskSpec("audio-generation", "audio", None, ("audio.cpp",), ("text-to-audio",),
             "post-0.6.0", note="no permissively licensed audio.cpp generation package under the evaluation size limit was natively accepted"),
    TaskSpec("audio-to-audio", "audio", None, ("audio.cpp",), ("audio-to-audio",), "0.5.5"),
    TaskSpec("audio-classification", "audio", None, ("audio.cpp",), ("audio-classification",),
             "post-0.6.0", note="audio.cpp exposes no classification task at the pinned revision"),
    TaskSpec("voice-activity-detection", "audio", None, ("audio.cpp", "whisper.cpp"),
             ("voice-activity-detection",), "post-0.6.0",
             note="audio.cpp VAD models are bundled runtime assets, not releasable sources"),
    TaskSpec("image-segmentation", "vision", VISION,
             ("vision.cpp", "ultralytics-ggml", "rf-detr.cpp"),
             ("image-segmentation",), "0.5.7", "none",
             "one atomic bundle commit of passing precisions after native task gates"),
    TaskSpec("super-resolution", "vision", None, ("vision.cpp",), (), "post-0.6.0",
             note="native MI-GAN/Real-ESRGAN outputs did not beat the naive baseline under PSNR/SSIM; no perceptual quality gate is established"),
    TaskSpec("inpainting", "vision", None, ("vision.cpp",), (), "post-0.6.0",
             note="native MI-GAN/Real-ESRGAN outputs did not beat the naive baseline under PSNR/SSIM; no perceptual quality gate is established"),
    TaskSpec("object-detection", "vision", VISION, ("ultralytics-ggml", "rf-detr.cpp"),
             ("object-detection",), "0.5.7", "none",
             "one atomic bundle commit of passing precisions after native task gates"),
    TaskSpec("image-classification", "vision", VISION, ("ultralytics-ggml",),
             ("image-classification",), "0.5.7", "none",
             "one atomic bundle commit of passing precisions after native task gates"),
    TaskSpec("keypoint-detection", "vision", VISION, ("ultralytics-ggml",),
             ("keypoint-detection",), "0.5.7", "none",
             "one atomic bundle commit of passing precisions after native task gates"),
    TaskSpec("depth-estimation", "vision", VISION, ("vision.cpp", "depth-anything.cpp"),
             ("depth-estimation",), "0.5.8", "none",
             "one atomic bundle commit of passing precisions after native task gates"),
    TaskSpec("mask-generation", "vision", VISION, ("vision.cpp", "sam3cpp"),
             ("mask-generation",), "0.5.8 (blocked)", "none",
             "official facebook/sam3 weights are gated under the SAM License; no ungated source with verifiable conversion provenance; sam3cpp not accepted"),
    TaskSpec("time-series-forecasting", "timeseries", None, ("timeseries.cpp",),
             ("time-series-forecasting",), "0.5.9"),
)}

HUB_TAG_TASKS = {tag: spec.id for spec in TASKS.values() for tag in spec.hub_tags}


def task_spec(task):
    try:
        return TASKS[task]
    except KeyError as error:
        raise ValueError(f"unknown task {task!r}; choose from {', '.join(sorted(TASKS))}") from error


def tasks_for_engine(engine):
    return [spec for spec in TASKS.values() if engine in spec.engines]


@dataclass
class Capability:
    """One engine revision × family × task cell and its independent facts."""
    engine: str
    family: str
    task: str
    revision: str | None
    load: bool | None = None
    convert: str | None = None
    quantize: list = field(default_factory=list)
    package: str | None = None
    platforms: dict = field(default_factory=dict)
    evidence: str = "catalog"           # catalog | probe | accepted
    acceptance: str = "pending"
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def capabilities(engine=None, task=None):
    """Every known cell, gathered from the catalogs each engine declares."""
    cells = []
    if engine in (None, "llama.cpp"):
        cells.append(Capability("llama.cpp", "convert_hf_to_gguf architectures", "text-generation",
                                None, load=True, convert="convert_hf_to_gguf.py",
                                quantize=["llama-quantize table of the installed checkout"],
                                evidence="probe",
                                notes="architecture support is read per model from the installed converter"))
        cells.extend((
            Capability("llama.cpp", "sentence-transformers encoders", "embeddings", None,
                       load=True, convert="convert_hf_to_gguf.py",
                       quantize=["installed llama-quantize table"], evidence="probe"),
            Capability("llama.cpp", "cross-encoders", "reranking", None,
                       load=True, convert="convert_hf_to_gguf.py",
                       quantize=["installed llama-quantize table"], evidence="probe"),
        ))
    if engine in (None, "clip.cpp"):
        for retrieval_task in ("zero-shot-image-classification", "image-feature-extraction"):
            cells.append(Capability("clip.cpp", "CLIPModel", retrieval_task, None, load=True,
                                    convert="models/convert_hf_to_gguf.py",
                                    quantize=["q8_0", "q5_1", "q5_0", "q4_1", "q4_0"],
                                    evidence="catalog"))
    if engine in (None, "llama.cpp", "audio.cpp", "whisper.cpp"):
        from .. import voice
        catalog = voice.audio_cpp_catalog()
        for backend in (*voice.CORE_BACKENDS, *catalog["backends"]):
            if engine not in (None, backend.backend):
                continue
            task_by_track = {
                voice.TTS: "text-to-speech",
                voice.ASR: "automatic-speech-recognition",
                voice.SEP: "audio-source-separation",
                voice.VC: "voice-conversion",
                voice.ALIGN: "forced-alignment",
            }
            cells.append(Capability(
                backend.backend, backend.family,
                task_by_track[backend.track],
                catalog["revision"] if backend.backend == "audio.cpp" else None,
                convert=backend.converter or None, quantize=list(backend.supported_quants),
                package=backend.model_format,
                evidence="catalog" if backend.backend == "audio.cpp" else "probe",
                notes=backend.status if backend.enabled else "disabled"))
    if engine in (None, "stable-diffusion.cpp"):
        from .. import media
        for item in media.capabilities():
            for item_task in item.tasks:
                cells.append(Capability(
                    item.engine, item.family, item_task, item.backend_revision, load=item.load,
                    convert=item.convert, quantize=sorted(item.quant_policies),
                    package=", ".join(item.load_formats), evidence="catalog", notes=item.status))
    return [cell for cell in cells if task in (None, cell.task)]
