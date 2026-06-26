from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_HOME_HTML = (Path(__file__).resolve().parent / "templates" / "home.html").read_text(
    encoding="utf-8"
)


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home() -> HTMLResponse:
    return HTMLResponse(_HOME_HTML)
