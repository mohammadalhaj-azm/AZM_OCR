"""Several handwriting models side by side (HANDWRITING_MODELS_FILE), LoRA adapters / 4-bit loading checks, and the
`lines` segmenter. No model and no detector is loaded: backends and detectors are stubs."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import cli
from app.api.main import create_app
from app.config import ConfigError
from app.handwriting import HandwritingService, HandwritingUnavailable, UnknownChoice
from app.handwriting import backends as hw_backends
from app.handwriting import service as hw_service
from app.handwriting.backends import adapter_applied, adapter_path, load_transformers, required_packages
from app.handwriting.regions import LinesSegmenter, Region, group_lines, pad_box
from app.handwriting.registry import DEFAULT_ID, load_models
from tests import synth_handwriting as hw
from tests.test_api import H, StubServices, fake_analyze, make_settings
from tests.test_handwriting import StubBackend, post

MODELS_YAML = """\
default: small
models:
  small:
    model: org/small-model
  big:
    model: org/big-base
    adapter: org/big-lora
    quantization: 4bit
    segmenter: lines
    prompt: "Transcribe the line."
    max_new_tokens: 160
    repetition_penalty: 1.05
"""


def models_file(tmp_path, text=MODELS_YAML):
    p = tmp_path / "hw_models.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


class NamedStub(StubBackend):
    def __init__(self, model_name, text):
        super().__init__(text)
        self.model_name = model_name

    def info(self):
        return {"backend": self.name, "name": self.model_name, "device": "cuda"}


class FakeLines:
    """A segmenter that returns `n` regions (stub lines) for every page."""
    name = "lines"

    def __init__(self, n):
        self.n = n
        self.pages = 0

    def split(self, page):
        self.pages += 1
        return [Region(page[i * 10:(i + 1) * 10], (0, i * 10, page.shape[1], (i + 1) * 10)) for i in range(self.n)]


@pytest.fixture
def two_models(tmp_path, monkeypatch):
    """A service with two configured models; each model's backend is a stub that answers with its own name."""
    stubs = {}

    def fake_make_backend(s):
        stubs[s.handwriting_model] = NamedStub(s.handwriting_model, f"text from {s.handwriting_model}")
        return stubs[s.handwriting_model]

    monkeypatch.setattr(hw_service, "make_backend", fake_make_backend)
    s = make_settings(tmp_path, handwriting_enabled=True, handwriting_models_file=models_file(tmp_path))
    svc = HandwritingService(s)
    svc._segmenters["lines"] = FakeLines(3)
    with TestClient(create_app(s, services=StubServices(), analyze_fn=fake_analyze, handwriting_service=svc)) as c:
        yield c, svc, stubs


def post_with(c, data, **fields):
    return c.post("/v1/handwriting", files={"file": ("note.png", data, "application/octet-stream")}, data=fields,
                  headers=H)


# --- The models file -------------------------------------------------------------------------------------------------

def test_without_a_file_there_is_one_model_from_the_env(tmp_path):
    s = make_settings(tmp_path)
    default, models = load_models(s)
    assert default == DEFAULT_ID and models == {DEFAULT_ID: s}


def test_entries_override_the_env_and_keep_the_rest(tmp_path):
    s = make_settings(tmp_path, handwriting_models_file=models_file(tmp_path), handwriting_dpi=250,
                      handwriting_max_new_tokens=900)
    default, models = load_models(s)
    assert default == "small" and list(models) == ["small", "big"]
    small, big = models["small"], models["big"]
    assert small.handwriting_model == "org/small-model" and small.handwriting_max_new_tokens == 900
    assert small.handwriting_adapter == "" and small.handwriting_quantization == "none"
    assert big.handwriting_adapter == "org/big-lora" and big.handwriting_quantization == "4bit"
    assert big.handwriting_segmenter == "lines" and big.handwriting_prompt == "Transcribe the line."
    assert big.handwriting_max_new_tokens == 160 and big.handwriting_repetition_penalty == 1.05
    assert big.handwriting_dpi == 250 and big.api_keys == s.api_keys  # everything else is the service's


@pytest.mark.parametrize("text, match", [
    ("default: a\nmodels:\n  a: {model: x, tempreature: 0}\n", "tempreature"),  # a typo is an error
    ("default: b\nmodels:\n  a: {model: x}\n", "default 'b' is not one of the models"),
    ("models:\n  a: {model: x}\n", "default"),
    ("default: A\nmodels:\n  A: {model: x}\n", "model id 'A'"),
    ("default: a\nmodels:\n  a: {model: x, quantization: 8bit}\n", "quantization"),
    ("default: a\nmodels:\n  a: {model: x, api_key: secret}\n", "api_key"),  # no secrets in the file
    ("default: a\nmodels: [\n", "not valid YAML"),
])
def test_invalid_files_fail_the_start(tmp_path, text, match):
    s = make_settings(tmp_path, handwriting_enabled=True, handwriting_models_file=models_file(tmp_path, text))
    with pytest.raises(ConfigError, match=match):
        HandwritingService(s, backend=StubBackend())


def test_missing_file_fails_the_start(tmp_path):
    s = make_settings(tmp_path, handwriting_enabled=True, handwriting_models_file=str(tmp_path / "nope.yaml"))
    with pytest.raises(ConfigError, match="can't be read"):
        HandwritingService(s, backend=StubBackend())


def test_disabled_service_never_reads_the_file(tmp_path):
    s = make_settings(tmp_path, handwriting_enabled=False, handwriting_models_file=str(tmp_path / "nope.yaml"))
    assert HandwritingService(s).status() == {"enabled": False}


def test_each_models_settings_are_checked(tmp_path):
    text = "default: a\nmodels:\n  a: {model: x}\n  b: {model: y, backend: openai, adapter: org/lora}\n"
    s = make_settings(tmp_path, handwriting_enabled=True, handwriting_models_file=models_file(tmp_path, text))
    with pytest.raises(ConfigError, match="b: HANDWRITING_BACKEND=openai needs HANDWRITING_BASE_URL.*b: "
                                          "HANDWRITING_ADAPTER and HANDWRITING_QUANTIZATION"):
        HandwritingService(s, backend=StubBackend())
    with pytest.raises(ConfigError, match="HANDWRITING_MAX_LINES"):
        HandwritingService(make_settings(tmp_path, handwriting_enabled=True, handwriting_max_lines=0))


# --- Picking a model per request -------------------------------------------------------------------------------------

def test_without_fields_the_default_model_answers(two_models):
    c, _, stubs = two_models
    r = post(c, hw.png(hw.few_words()))
    assert r.status_code == 200
    out = r.json()
    assert out["fullText"] == "text from org/small-model" and stubs["org/big-base"].calls == []
    assert out["model"] == {"id": "small", "backend": "transformers", "name": "org/small-model", "device": "cuda",
                            "maxPixels": 1280 * 28 * 28, "segmenter": "whole_page"}


def test_model_field_picks_the_other_model_with_its_settings(two_models):
    c, svc, stubs = two_models
    r = post_with(c, hw.png(hw.few_words()), model="big")
    assert r.status_code == 200
    out = r.json()
    assert out["model"] == {"id": "big", "backend": "transformers", "name": "org/big-base", "adapter": "org/big-lora",
                            "quantization": "4bit", "device": "cuda", "maxPixels": 1280 * 28 * 28,
                            "segmenter": "lines"}
    # its own segmenter: one model call per line, the lines joined in order
    assert len(stubs["org/big-base"].calls) == 3 and stubs["org/small-model"].calls == []
    assert out["pages"][0]["text"] == "\n".join(["text from org/big-base"] * 3)
    assert svc._segmenters["lines"].pages == 1


def test_segmenter_field_overrides_the_models_own(two_models):
    c, _, stubs = two_models
    # an image that is already one line: whole_page sends it to the line model as it is
    r = post_with(c, hw.png(hw.few_words()), model="big", segmenter="whole_page")
    assert r.status_code == 200 and r.json()["model"]["segmenter"] == "whole_page"
    assert len(stubs["org/big-base"].calls) == 1 and stubs["org/big-base"].calls[0].shape == (hw.H, hw.W, 3)
    r = post_with(c, hw.png(hw.few_words()), segmenter="lines")  # the default model, split into lines
    assert r.status_code == 200 and r.json()["model"]["segmenter"] == "lines"
    assert r.json()["model"]["id"] == "small" and len(stubs["org/small-model"].calls) == 3


def test_unknown_model_or_segmenter_is_400(two_models):
    c, _, stubs = two_models
    r = post_with(c, hw.png(hw.few_words()), model="nope")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_model"
    assert "small, big" in r.json()["error"]["message"]
    r = post_with(c, hw.png(hw.few_words()), segmenter="columns")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
    assert all(not s.calls for s in stubs.values())


def test_status_lists_every_model(two_models):
    c, _, _ = two_models
    st = c.get("/health/handwriting").json()
    assert st["enabled"] is True and st["default"] == "small" and st["segmenter"] == "whole_page"
    assert set(st["models"]) == {"small", "big"}
    assert st["models"]["big"] == {"segmenter": "lines", "name": "org/big-base", "adapter": "org/big-lora",
                                   "quantization": "4bit", "backend": "transformers", "loaded": True}


def test_one_request_at_a_time_across_models(two_models):
    c, svc, _ = two_models
    svc._slot.acquire()
    try:
        assert post_with(c, hw.png(hw.few_words()), model="big").json()["error"]["code"] == "busy"
    finally:
        svc._slot.release()


def test_unknown_model_raised_by_the_service(tmp_path):
    svc = HandwritingService(make_settings(tmp_path, handwriting_enabled=True), backend=StubBackend())
    with pytest.raises(UnknownChoice, match="configured: default"):
        svc.resolve("other")
    m, seg = svc.resolve()
    assert m.id == DEFAULT_ID and seg.name == "whole_page"


# --- Lines -----------------------------------------------------------------------------------------------------------

def test_group_lines_merges_pieces_of_a_line_and_orders_top_to_bottom():
    boxes = [
        (600, 300, 900, 360),  # line 2, right piece
        (100, 100, 400, 150),  # line 1, left word
        (100, 305, 500, 355),  # line 2, left piece
        (450, 95, 800, 158),   # line 1, right word (a bit taller)
        (100, 500, 300, 540),  # line 3
        (50, 50, 50, 80),      # empty box: ignored
    ]
    assert group_lines(boxes) == [(100, 95, 800, 158), (100, 300, 900, 360), (100, 500, 300, 540)]


def test_group_lines_keeps_close_lines_apart():
    # descenders touch the next line: a small overlap (< 50% of the smaller height) doesn't merge them
    assert group_lines([(0, 0, 500, 60), (0, 50, 500, 110)]) == [(0, 0, 500, 60), (0, 50, 500, 110)]


def test_pad_box_stays_on_the_page():
    assert pad_box((10, 10, 200, 50), 1000, 1000) == (0, 4, 212, 56)  # 40 px high: pad 6 vertically, 12 horizontally
    assert pad_box((900, 970, 1000, 1000), 1000, 1000) == (891, 966, 1000, 1000)


def test_lines_segmenter_crops_each_line_from_the_full_page(tmp_path):
    page = np.zeros((1000, 800, 3), np.uint8)
    page[100:150, 100:700] = 255
    seen = []

    def detector(img):
        seen.append(img.shape)
        h = img.shape[0] / 1000  # boxes in the (maybe downscaled) detector image
        return [[[100 * h, 100 * h], [700 * h, 100 * h], [700 * h, 150 * h], [100 * h, 150 * h]]]

    regions = LinesSegmenter(make_settings(tmp_path), detector=detector).split(page)
    assert seen == [(1000, 800, 3)]
    assert [r.bbox for r in regions] == [(85, 92, 715, 158)] and regions[0].image.shape == (66, 630, 3)
    # a page larger than OCR_MAX_SIDE is detected downscaled, but cropped from the full-resolution page
    regions = LinesSegmenter(make_settings(tmp_path, ocr_max_side=500), detector=detector).split(page)
    assert seen[-1] == (500, 400, 3) and [r.bbox for r in regions] == [(85, 92, 715, 158)]


def test_no_lines_is_a_warning_not_a_whole_page_guess(tmp_path):
    backend = StubBackend()
    svc = HandwritingService(make_settings(tmp_path, handwriting_enabled=True, handwriting_segmenter="lines"),
                             backend=backend, segmenter=FakeLines(0))
    page = svc.transcribe(hw.png(hw.few_words()), "image/png")["pages"][0]
    assert page["text"] == "" and page["warnings"] == ["no_lines"] and backend.calls == []


def test_too_many_lines_reads_the_first_ones_and_warns(tmp_path):
    backend = StubBackend("x")
    svc = HandwritingService(make_settings(tmp_path, handwriting_enabled=True, handwriting_max_lines=2),
                             backend=backend, segmenter=FakeLines(5))
    page = svc.transcribe(hw.png(hw.few_words()), "image/png")["pages"][0]
    assert page["text"] == "x\nx" and page["warnings"] == ["too_many_lines"] and len(backend.calls) == 2


def test_blank_pages_skip_the_line_detector_too(tmp_path):
    lines = FakeLines(3)
    svc = HandwritingService(make_settings(tmp_path, handwriting_enabled=True), backend=StubBackend(), segmenter=lines)
    assert svc.transcribe(hw.png(hw.blank()), "image/png")["pages"][0]["skipped"] == "blank" and lines.pages == 0


# --- Adapter and 4-bit loading ---------------------------------------------------------------------------------------

def test_adapter_applied_counts_zero_lora_b():
    loaded = [("x.lora_A.weight", np.ones((2, 2))), ("x.lora_B.weight", np.full((2, 2), 0.1)),
              ("y.lora_B.weight", np.ones((2, 2))), ("y.base_layer.weight", np.zeros((2, 2)))]
    assert adapter_applied(loaded) == (2, 2)
    not_loaded = [("x.lora_A.weight", np.ones((2, 2))), ("x.lora_B.weight", np.zeros((2, 2)))]
    assert adapter_applied(not_loaded) == (1, 0)
    assert adapter_applied([("w", np.ones(2))]) == (0, 0)  # no LoRA layers at all


def peft_remap(key: str, mapping: dict) -> str:
    """What PEFT (load_peft_weights) does with key_mapping: strip `base_model.model.`, first matching pattern wins."""
    import re
    key = key.removeprefix("base_model.model.")
    for pattern, replacement in mapping.items():
        new, n = re.subn(pattern, replacement, key)
        if n:
            return "base_model.model." + new
    return "base_model.model." + key


def test_old_layout_adapter_keys_are_mapped_to_the_new_qwen_vl_layout():
    new_layout = ["model.visual.blocks.0.attn.qkv.weight", "model.language_model.layers.0.mlp.up_proj.weight",
                  "lm_head.weight"]
    mapping = hw_backends.adapter_key_mapping(new_layout)
    assert mapping is not None
    # the Fanar adapter's file layout (seen in its safetensors header)
    assert peft_remap("base_model.model.model.layers.0.mlp.down_proj.lora_A.weight", mapping) == \
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    assert peft_remap("base_model.model.model.layers.27.self_attn.q_proj.lora_B.weight", mapping) == \
        "base_model.model.model.language_model.layers.27.self_attn.q_proj.lora_B.weight"
    assert peft_remap("base_model.model.visual.blocks.0.mlp.up_proj.lora_A.weight", mapping) == \
        "base_model.model.model.visual.blocks.0.mlp.up_proj.lora_A.weight"
    for already_new in ("base_model.model.model.language_model.layers.0.mlp.up_proj.lora_A.weight",
                        "base_model.model.model.visual.blocks.0.mlp.up_proj.lora_A.weight",
                        "base_model.model.lm_head.lora_A.weight"):
        assert peft_remap(already_new, mapping) == already_new
    # a model that still has the old layout: nothing to map
    assert hw_backends.adapter_key_mapping(["model.layers.0.mlp.up_proj.weight", "visual.blocks.0.x"]) is None


def test_old_layout_adapter_loads_into_a_tiny_qwen_vl(tmp_path):
    """End to end with real PEFT and transformers on a random tiny model: without the mapping no LoRA weight loads
    (the error seen on the server: 0 of 196), with it every one does, with the file's values."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file

    cfg = transformers.Qwen2_5_VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=256,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           fullatt_block_indexes=[0]))
    dims = {"self_attn.q_proj": (64, 64), "self_attn.k_proj": (64, 32), "self_attn.v_proj": (64, 32),
            "self_attn.o_proj": (64, 64), "mlp.gate_proj": (64, 128), "mlp.up_proj": (64, 128),
            "mlp.down_proj": (128, 64)}
    weights = {}
    for i in range(2):
        for m, (fin, fout) in dims.items():
            weights[f"base_model.model.model.layers.{i}.{m}.lora_A.weight"] = torch.randn(16, fin)
            weights[f"base_model.model.model.layers.{i}.{m}.lora_B.weight"] = torch.randn(fout, 16)
    tmp = tmp_path
    (tmp / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32, "base_model_name_or_path": "tiny",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj",
                           "layers.0.mlp.down_proj", "layers.1.mlp.gate_proj", "layers.1.mlp.up_proj",
                           "layers.1.mlp.down_proj"]}))
    save_file(weights, str(tmp / "adapter_model.safetensors"))

    def load(with_mapping):
        base = transformers.Qwen2_5_VLForConditionalGeneration(cfg)
        mapping = hw_backends.adapter_key_mapping(n for n, _ in base.named_parameters()) if with_mapping else None
        return peft.PeftModel.from_pretrained(base, str(tmp), key_mapping=mapping)

    with pytest.warns(UserWarning):  # PEFT warns about the unmatched keys
        assert adapter_applied(load(False).named_parameters()) == (14, 0)
    model = load(True)
    assert adapter_applied(model.named_parameters()) == (14, 14)
    p = dict(model.named_parameters())
    assert torch.equal(p["base_model.model.model.language_model.layers.1.mlp.down_proj.lora_B.default.weight"],
                       weights["base_model.model.model.layers.1.mlp.down_proj.lora_B.weight"])


def test_required_packages(tmp_path):
    assert required_packages(make_settings(tmp_path)) == ["torch", "transformers"]
    s = make_settings(tmp_path, handwriting_adapter="org/lora", handwriting_quantization="4bit")
    assert required_packages(s) == ["torch", "transformers", "peft", "bitsandbytes"]


def test_adapter_path(tmp_path):
    folder = tmp_path / "lora"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}")
    assert adapter_path(str(folder)) is None  # weights missing
    (folder / "adapter_model.safetensors").write_bytes(b"x")
    assert adapter_path(str(folder)) == str(folder)
    assert adapter_path("someone/not-a-cached-adapter-xyz") is None


def fake_model_folder(tmp_path):
    folder = tmp_path / "base"
    folder.mkdir()
    for f in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
        (folder / f).write_text("{}")
    return str(folder)


def test_missing_adapter_is_reported_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(hw_backends, "deps_installed", lambda s=None: True)
    s = make_settings(tmp_path, handwriting_model=fake_model_folder(tmp_path),
                      handwriting_adapter=str(tmp_path / "no-adapter"))
    with pytest.raises(HandwritingUnavailable, match="adapter is not downloaded"):
        load_transformers(s)


def test_missing_packages_are_named(tmp_path, monkeypatch):
    monkeypatch.setattr(hw_backends, "deps_installed", lambda s=None: False)
    s = make_settings(tmp_path, handwriting_adapter="org/lora", handwriting_quantization="4bit")
    with pytest.raises(HandwritingUnavailable, match="torch, transformers, peft, bitsandbytes"):
        load_transformers(s)


def test_4bit_needs_a_gpu(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.setattr(hw_backends, "deps_installed", lambda s=None: True)
    s = make_settings(tmp_path, handwriting_model=fake_model_folder(tmp_path), handwriting_quantization="4bit",
                      handwriting_device="cpu")
    with pytest.raises(HandwritingUnavailable, match="4bit needs a CUDA GPU"):
        load_transformers(s)


def test_adapter_status_is_reported(tmp_path):
    s = make_settings(tmp_path, handwriting_adapter="org/not-downloaded-xyz")
    assert hw_backends.TransformersBackend(s).status()["adapterDownloaded"] is False
    assert "adapterDownloaded" not in hw_backends.TransformersBackend(make_settings(tmp_path)).status()


# --- Download --------------------------------------------------------------------------------------------------------

def test_download_fetches_every_model_and_adapter_once(tmp_path, monkeypatch):
    text = MODELS_YAML + ("  big-full:\n    model: org/big-base\n    segmenter: lines\n"
                          "  served:\n    model: served-name\n    backend: openai\n    base_url: http://x/v1\n")
    s = make_settings(tmp_path, handwriting_models_file=models_file(tmp_path, text))
    assert cli.handwriting_repos(s) == ["org/small-model", "org/big-base", "org/big-lora"]
    assert cli.handwriting_repos(make_settings(tmp_path)) == ["sherif1313/Arabic-English-handwritten-OCR-v3"]
    got = []
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "download_handwriting_model", lambda repo: got.append(repo) or 0)
    assert cli.main(["download-handwriting-model"]) == 0 and got == cli.handwriting_repos(s)
    got.clear()
    assert cli.main(["download-handwriting-model", "--model", "org/one"]) == 0 and got == ["org/one"]


def test_shipped_models_file_is_valid(tmp_path):
    s = make_settings(tmp_path, handwriting_models_file="config/handwriting_models.yaml")
    default, models = load_models(s)
    assert default == "sherif-v3" and set(models) == {"sherif-v3", "fanar"}
    fanar = models["fanar"]
    assert (fanar.handwriting_model, fanar.handwriting_adapter) == ("QCRI/Fanar-2-Oryx-IVU",
                                                                    "mabdulaziz499/arabic-htr-fanar-7b-lora")
    assert fanar.handwriting_quantization == "4bit" and fanar.handwriting_segmenter == "lines"
    assert (fanar.handwriting_min_pixels, fanar.handwriting_max_pixels) == (200704, 802816)
    assert models["sherif-v3"].handwriting_model == "sherif1313/Arabic-English-handwritten-OCR-v3"
    assert json.dumps(fanar.handwriting_prompt).startswith('"Transcribe the handwritten Arabic text')
