"""POST /v1/extract-and-answer (experimental): routed extraction, LLM correction and answers, as an async job.

Same job machinery as /v1/documents (queue limits, Idempotency-Key, callbackUrl, externalId, metadata, profiles,
encrypted storage, GET /v1/jobs/{id}); the questions travel inside the job's options, which are encrypted at rest by
the postgres store. The pipeline side is app/answer.
"""

from fastapi import APIRouter, Depends, File, Form, Header, Request, Response, UploadFile

from app.answer import TASK, AnswerOptions
from app.api.clients import Client, ClientRegistry
from app.api.deps import app_clients, app_settings, app_store, require_scope, use_profile
from app.api.errors import ERROR_RESPONSES, ApiError
from app.api.models import JobCreated
from app.api.routes.documents import MAX_EXTERNAL_ID, MAX_METADATA_BYTES, queue_job
from app.api.store import JobStore
from app.config import Settings

router = APIRouter()
MODES = ("auto", "printed", "handwritten")


@router.post("/v1/extract-and-answer", status_code=202, response_model=JobCreated, responses=ERROR_RESPONSES,
             tags=["experimental"],
             summary="Experimental: extract text (printed and handwritten), correct it, flag what needs review and "
                     "answer questions from the document (async job)",
             description="Every page is routed: text layers are taken as they are; scanned pages and images are read "
                         "by PaddleOCR, and pages that are handwritten or mixed go to the handwriting model (the "
                         "result's `pages[].route` says why, and `uncertain` when it is not clear). The local text "
                         "LLM then corrects and formats the text without changing numbers, dates or IDs, lists "
                         "what a person should review, and answers the `questions` from the document only, with "
                         "page citations. Questions are optional: without them the job only extracts and corrects. "
                         "Poll `GET /v1/jobs/{jobId}` (or use `callbackUrl`). Slow: handwriting takes about a "
                         "minute or more per page on CPU. At most `ANSWER_MAX_PAGES` pages. Builds on the "
                         "experimental handwriting model, so it may change while it is evaluated.")
async def extract_and_answer(
        request: Request, response: Response, file: UploadFile = File(...),
        questions: list[str] | None = Form(None, description="A question about the document; repeat the field for "
                                                             "several (at most ANSWER_MAX_QUESTIONS)"),
        mode: str | None = Form(None, description="auto (default): route each page; printed / handwritten: read "
                                                  "every page with that engine (handwritten also renders PDF pages "
                                                  "that have a text layer)"),
        correct: bool = Form(True, description="false: no LLM correction (extraction, routing and questions only)"),
        callbackUrl: str | None = Form(None, description="Webhook URL (host must be allow-listed)"),
        externalId: str | None = Form(None, description="Your reference, echoed in the job and webhook "
                                                        f"(at most {MAX_EXTERNAL_ID} characters)"),
        metadata: str | None = Form(None, description="JSON object echoed in the job and webhook "
                                                      f"(at most {MAX_METADATA_BYTES} bytes)"),
        includeBoxes: bool | None = Form(None, description="Add PaddleOCR's `lines` [{text, confidence, bbox}] and "
                                                           "the image size to OCR pages"),
        profile: str | None = Form(None, description="Processing profile: one of your client's (default: its first)"),
        idempotency_key: str | None = Header(None, alias="Idempotency-Key",
                                             description="Retry-safe uploads: the same key, file and options return "
                                                         "the same job (within RESULT_TTL_S)"),
        client: Client = Depends(require_scope("documents")), settings: Settings = Depends(app_settings),
        store: JobStore = Depends(app_store), clients: ClientRegistry = Depends(app_clients)):
    if not settings.answer_enabled:
        raise ApiError(503, "answer_unavailable", "extract-and-answer is disabled on this server")
    prof = use_profile(clients, client, profile)
    if not prof.return_text:
        raise ApiError(403, "forbidden", f"profile {prof.name} returns no text: use POST /v1/documents")
    qs = [q.strip() for q in questions or [] if q and q.strip()]
    if len(qs) > settings.answer_max_questions:
        raise ApiError(422, "invalid_request", f"at most {settings.answer_max_questions} questions")
    if any(len(q) > settings.answer_max_question_chars for q in qs):
        raise ApiError(422, "invalid_request",
                       f"a question must be at most {settings.answer_max_question_chars} characters")
    if (mode or "auto") not in MODES:
        raise ApiError(422, "invalid_request", "mode must be one of: " + ", ".join(MODES))
    options = AnswerOptions(qs, mode or "auto", correct)
    return await queue_job(request, response, file, client, settings, store, prof, callbackUrl, externalId, metadata,
                           includeBoxes, idempotency_key, endpoint="extract-and-answer",
                           task_options={"task": TASK, "answer": options.to_job()},
                           task_facts={"questions": len(qs), "mode": options.mode, "correct": correct})
