import subprocess, sys
from pathlib import Path
from django.conf import settings
from django.http import FileResponse, HttpResponseBadRequest
from django.shortcuts import render

from .forms import PDFUploadForm

# Tenta usar função importável (ideal). Se não existir, caímos pro CLI.
def _run_extractor(pdf_path: Path, out_dir: Path, *, params: dict) -> Path:
    try:
        from extract_infos_2 import extract_to_xlsx  # ideal
        xlsx = extract_to_xlsx(
            pdf_path=str(pdf_path),
            out_dir=str(out_dir),
            max_imovel=params.get("max_imovel", -1),
            max_res=params.get("max_res", -1),
            min_lotes_in_poly=params.get("min_lotes_in_poly", 6),
            snap_tol=params.get("snap_tol", 6.0),
            min_cycle_area=params.get("min_cycle_area", 300.0),
            debug_regions=params.get("debug_regions", False),
            out_filename=pdf_path.stem + ".xlsx",
        )
        return Path(xlsx)
    except Exception:
        script = settings.BASE_DIR / "extract_infos_2.py"
        out_path = out_dir / (pdf_path.stem + ".xlsx")
        cmd = [
            sys.executable, str(script), str(pdf_path),
            "--out", str(out_path),
            "--min-lotes-in-poly", str(params.get("min_lotes_in_poly", 6)),
            "--snap-tol", str(params.get("snap_tol", 6.0)),
            "--min-cycle-area", str(params.get("min_cycle_area", 300.0)),
            "--max-imovel", str(params.get("max_imovel", -1)),
            "--max-res", str(params.get("max_res", -1)),
        ]
        if params.get("debug_regions"):
            cmd.append("--debug-regions")
        subprocess.run(cmd, check=True)
        return out_path

def upload_view(request):
    if request.method == "POST":
        form = PDFUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return HttpResponseBadRequest("Formulário inválido.")

        uploads = Path(settings.MEDIA_ROOT) / "uploads"
        results = Path(settings.MEDIA_ROOT) / "results"
        uploads.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)

        f = form.cleaned_data["pdf"]
        pdf_path = uploads / f.name
        with open(pdf_path, "wb") as dst:
            for chunk in f.chunks():
                dst.write(chunk)

        xlsx_path = _run_extractor(pdf_path, results, params=form.cleaned_data)
        return FileResponse(open(xlsx_path, "rb"), as_attachment=True, filename=xlsx_path.name)

    return render(request, "processor/upload.html", {"form": PDFUploadForm()})
