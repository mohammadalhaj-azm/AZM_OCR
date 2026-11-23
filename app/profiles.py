"""Processing profiles: per-entity settings without code changes.

A profile (defined under `profiles:` in CLIENTS_FILE, see config/clients.example.yaml) changes how a client's documents
are processed, what is returned and kept, and for how long. Only the settings below can be changed; infrastructure
(database, LLM server, keys) never can. A setting that isn't given keeps its global value.

The built-in profile `default` changes nothing. When a job is uploaded, the profile is copied into it, so the job is
processed the same way on any replica even if CLIENTS_FILE changes meanwhile.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings

DEFAULT = "default"


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)  # an unknown key is an error: a typo can't be ignored

    name: str = DEFAULT

    # Processing
    id_patterns: str | None = Field(None, description="Rules file (keywords, ID-number patterns) instead of "
                                                      "ID_PATTERNS_PATH, e.g. config/patterns/civil.yaml")
    detect_id: bool = Field(True, description="false: POST /v1/documents extracts text only (no ID detection)")
    llm_enabled: bool | None = Field(None, description="false: the text LLM is never called (rules only)")
    vision_enabled: bool | None = Field(None, description="false: the vision model is never called")
    max_pages: int | None = Field(None, ge=1, description="At most MAX_PAGES")
    ocr_dpi: int | None = Field(None, ge=72, le=600, description="Rendering DPI of scanned PDF pages")
    min_text_chars_per_page: int | None = Field(None, ge=0)

    # Output (data minimization: what is not returned is not stored either)
    include_boxes: bool = Field(False, description="Default for the includeBoxes request field")
    return_text: bool = Field(True, description="false: no document text in results (pages, fullText, "
                                                "normalizedText, lines, the LLM's reasons)")
    return_mrz: bool = Field(True, description="false: no parsed MRZ fields in idDetection")

    # Retention
    result_ttl_s: int | None = Field(None, ge=0, description="Instead of RESULT_TTL_S for this profile's jobs")

    @property
    def is_default(self) -> bool:
        return self == Profile()

    def settings_for(self, base: Settings) -> Settings:
        """The global settings with this profile's processing settings applied."""
        update = {k: v for k, v in {
            "id_patterns_path": Path(self.id_patterns) if self.id_patterns else None,
            "llm_enabled": self.llm_enabled, "vision_enabled": self.vision_enabled, "max_pages": self.max_pages,
            "ocr_dpi": self.ocr_dpi, "min_text_chars_per_page": self.min_text_chars_per_page,
        }.items() if v is not None}
        return base.model_copy(update=update) if update else base

    def detection_key(self) -> tuple:
        """Profiles with the same key share an IdDetector."""
        return self.id_patterns, self.llm_enabled, self.vision_enabled

    def to_job(self) -> dict:
        """What a job stores (JSON): the name and the settings that differ from the defaults."""
        return self.model_dump(exclude_defaults=True) | {"name": self.name}


DEFAULT_PROFILE = Profile()


def withhold(result: dict, profile: Profile) -> dict:
    """Data minimization: remove what the profile doesn't return, before the result is stored or sent.
    Records what was withheld in `processing.withheld`."""
    withheld = []
    if not profile.return_text:
        withheld.append("text")
        for page in result.get("pages") or []:
            page["text"] = ""
            page.pop("lines", None)
        result["fullText"] = result["normalizedText"] = ""
        det = result.get("idDetection")
        if det:  # the models' free-text reasons can quote the document; the rules' reasons can't
            det["reasons"] = [r for r in det.get("reasons", []) if r.startswith("rules:")]
    if not profile.return_mrz and (result.get("idDetection") or {}).pop("mrz", None) is not None:
        withheld.append("mrz")
    if withheld:
        result.setdefault("processing", {})["withheld"] = withheld
    return result
