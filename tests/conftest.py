import pytest

from app.config import Settings
from tests import synth


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(scope="session")
def fixtures() -> dict[str, bytes]:
    return synth.all_fixtures()


@pytest.fixture(scope="session")
def engine(settings):
    from app.ocr.engine import OcrEngine

    return OcrEngine(settings).load()


def compact(text: str) -> str:
    return "".join(text.split()).upper()
