"""Content/package production-quality validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any
from urllib import request, error

VISUAL_STACK_ALIASES: dict[str, tuple[str, ...]] = {
    "remotion": ("remotion",),
    "gsap": ("gsap",),
    "lottie": ("lottie", "dotlottie"),
    "threejs": ("three.js", "threejs", "three-js"),
    "antv": ("antv", "chart", "charts"),
    "manim": ("manim",),
    "excalidraw": ("excalidraw",),
    "baoyu": ("baoyu", "infographic"),
    "hyperframes": ("hyperframes",),
}

CORE_MANIFEST_FIELDS = ("run_id", "stage", "generated_at", "owner_profile")


def validate_content_package(project_path: str | Path, *, write: bool = True, now: str | None = None) -> dict[str, Any]:
    """Validate a content/video package and optionally emit runtime + QA reports.

    The validator is intentionally strict for human-facing packages: structural files are
    necessary but not enough. Proof assets and claim-to-artifact traceability decide pass/hold.
    """

    root = Path(project_path).expanduser().resolve()
    generated_at = now or _utc_now()
    runtime_check = build_runtime_check(root, generated_at=generated_at)
    manifest, manifest_error = _load_manifest(root / "manifest.json")
    checks = _build_checks(root, manifest, manifest_error, runtime_check)
    verdict = "hold" if any(check["status"] == "hold" for check in checks if check.get("severity") in {"critical", "high"}) else "pass"

    qa_verdict = {
        "schema_version": 1,
        "run_id": _manifest_value(manifest, "run_id", "unknown"),
        "stage": "content_package_validation",
        "generated_at": generated_at,
        "generated_by": "hermes-evolve content validate",
        "owner_profile": _manifest_value(manifest, "owner_profile", "video-production"),
        "project_path": str(root),
        "verdict": verdict,
        "checks": checks,
        "summary": {
            "total": len(checks),
            "pass": sum(1 for check in checks if check["status"] == "pass"),
            "hold": sum(1 for check in checks if check["status"] == "hold"),
            "warn": sum(1 for check in checks if check["status"] == "warn"),
        },
        "runtime_check_path": str(root / "runtime_check.json"),
        "qa_verdict_path": str(root / "qa_verdict.json"),
        "quality_contract": {
            "empty_proof_assets_means": "hold",
            "tool_name_cards_are_proof": False,
            "requires_human_facing_critique": True,
        },
    }

    if write:
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "runtime_check.json", runtime_check)
        _write_json(root / "qa_verdict.json", qa_verdict)

    return {
        "verdict": verdict,
        "project_path": str(root),
        "runtime_check": runtime_check,
        "qa_verdict": qa_verdict,
        "runtime_check_path": str(root / "runtime_check.json"),
        "qa_verdict_path": str(root / "qa_verdict.json"),
        "holds": [check for check in checks if check["status"] == "hold"],
    }


def build_runtime_check(project_path: Path, *, generated_at: str | None = None) -> dict[str, Any]:
    """Build a deterministic runtime preflight report without mutating the project."""

    root = project_path.expanduser().resolve()
    generated_at = generated_at or _utc_now()
    bridge_url = os.environ.get("HERMES_VISION_BRIDGE_BASE_URL") or os.environ.get("HERMES_MAC_VISION_BRIDGE_URL")
    bridge_probe = _probe_http_bridge(bridge_url) if bridge_url else {"status": "na", "base_url": None, "reason": "No vision bridge URL configured"}
    tools = {name: _tool_probe(name) for name in ("ffmpeg", "ffprobe", "node", "npm")}
    writable = root.exists() and os.access(root, os.W_OK)

    return {
        "schema_version": 1,
        "run_id": _best_effort_run_id(root),
        "stage": "runtime_preflight",
        "generated_at": generated_at,
        "owner_profile": "video-production",
        "status": "pass" if root.exists() and writable else "hold",
        "host": platform.node(),
        "system": platform.platform(),
        "cwd": str(Path.cwd()),
        "project_path": str(root),
        "execution_boundary": _execution_boundary(root),
        "tailscale_bridge": bridge_probe,
        "vision_bridge_probe": bridge_probe,
        "tools": tools,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "prefix": sys.prefix,
            "venv": os.environ.get("VIRTUAL_ENV"),
        },
        "memory": _memory_probe(),
        "render_path": {
            "path": str(root / "renders"),
            "exists": (root / "renders").exists(),
            "project_writable": writable,
        },
    }


def _build_checks(root: Path, manifest: dict[str, Any] | None, manifest_error: str | None, runtime_check: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(_check("project_path", "pass" if root.exists() and root.is_dir() else "hold", "critical", f"Project path exists: {root}"))

    if manifest is None:
        checks.append(_check("manifest_exists", "hold", "critical", manifest_error or "manifest.json is missing or invalid"))
        manifest = {}
    else:
        checks.append(_check("manifest_exists", "pass", "critical", "manifest.json exists and parses as JSON"))

    missing_core = [field for field in CORE_MANIFEST_FIELDS if not manifest.get(field)]
    checks.append(
        _check(
            "manifest_core_fields",
            "hold" if missing_core else "pass",
            "critical",
            "Missing manifest fields: " + ", ".join(missing_core) if missing_core else "Manifest has run_id/stage/generated_at/owner_profile",
        )
    )

    artifact_report = _artifact_report(root, manifest.get("artifacts"))
    checks.append(
        _check(
            "declared_artifacts_exist",
            "hold" if artifact_report["missing"] else "pass",
            "critical",
            "Missing declared artifacts: " + ", ".join(artifact_report["missing"]) if artifact_report["missing"] else f"Declared artifacts exist: {len(artifact_report['present'])}",
            details=artifact_report,
        )
    )

    proof_dir = root / "proof-assets"
    proof_files = _proof_files(proof_dir)
    checks.append(_check("proof_assets_dir", "pass" if proof_dir.is_dir() else "hold", "critical", f"proof-assets directory {'exists' if proof_dir.is_dir() else 'is missing'}"))
    checks.append(
        _check(
            "proof_assets_non_empty",
            "pass" if proof_files else "hold",
            "critical",
            f"proof-assets contains {len(proof_files)} inspectable file(s)",
            details={"files": [str(path) for path in proof_files]},
        )
    )

    claimed_terms = _claimed_visual_terms(manifest.get("capability_scope", []))
    missing_claims = [term for term in claimed_terms if not _has_proof_for_term(term, manifest, proof_files)]
    checks.append(
        _check(
            "claimed_visual_stack_proof",
            "hold" if missing_claims else "pass",
            "high",
            "Missing proof for claimed visual stack: " + ", ".join(missing_claims) if missing_claims else "Every claimed visual stack item has proof mapping or proof file",
            details={"claimed_terms": claimed_terms, "missing_claims": missing_claims},
        )
    )

    runtime_status = runtime_check.get("status")
    checks.append(_check("runtime_preflight", "pass" if runtime_status == "pass" else "hold", "high", f"runtime_check status={runtime_status}"))

    human_gate_status = "pass" if proof_files and not missing_claims else "hold"
    checks.append(
        _check(
            "human_quality_gate",
            human_gate_status,
            "critical",
            "Proof density and claim traceability are sufficient for human-facing review" if human_gate_status == "pass" else "Package lacks enough visual proof to convince a skeptical human viewer",
            details={
                "proof_file_count": len(proof_files),
                "missing_claims": missing_claims,
                "requires_manual_review_after_pass": True,
            },
        )
    )
    return checks


def _check(check_id: str, status: str, severity: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"id": check_id, "status": status, "severity": severity, "message": message}
    if details is not None:
        payload["details"] = details
    return payload


def _artifact_report(root: Path, artifacts: Any) -> dict[str, Any]:
    present: list[dict[str, Any]] = []
    missing: list[str] = []
    if not isinstance(artifacts, dict) or not artifacts:
        return {"present": present, "missing": ["manifest.artifacts is empty or not an object"]}
    for name, raw_path in artifacts.items():
        if not isinstance(raw_path, str):
            missing.append(f"{name}: not a string path")
            continue
        path = _resolve_project_path(root, raw_path)
        if not path.exists() or not path.is_file():
            missing.append(f"{name}: {path}")
            continue
        present.append({"name": name, "path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {"present": present, "missing": missing}


def _proof_files(proof_dir: Path) -> list[Path]:
    if not proof_dir.is_dir():
        return []
    return sorted(path for path in proof_dir.rglob("*") if path.is_file() and not path.name.startswith("."))


def _claimed_visual_terms(scope: Any) -> list[str]:
    if not isinstance(scope, list):
        return []
    joined = "\n".join(str(item).lower() for item in scope)
    terms: list[str] = []
    for canonical, aliases in VISUAL_STACK_ALIASES.items():
        if any(alias in joined for alias in aliases):
            terms.append(canonical)
    return terms


def _has_proof_for_term(term: str, manifest: dict[str, Any], proof_files: list[Path]) -> bool:
    aliases = VISUAL_STACK_ALIASES[term]
    proof_artifacts = manifest.get("proof_artifacts")
    proof_blob = ""
    if isinstance(proof_artifacts, dict):
        proof_blob = json.dumps(proof_artifacts, sort_keys=True).lower()
    elif isinstance(proof_artifacts, list):
        proof_blob = json.dumps(proof_artifacts, sort_keys=True).lower()
    file_blob = "\n".join(str(path).lower() for path in proof_files)
    return any(alias in proof_blob or alias in file_blob for alias in aliases)


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"manifest.json missing at {path}"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"manifest.json invalid: {exc}"
    if not isinstance(data, dict):
        return None, "manifest.json root must be an object"
    return data, None


def _resolve_project_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _write_json(path: Path, data: dict[str, Any]):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _manifest_value(manifest: dict[str, Any] | None, key: str, default: str) -> str:
    if isinstance(manifest, dict) and manifest.get(key):
        return str(manifest[key])
    return default


def _best_effort_run_id(root: Path) -> str:
    manifest, _ = _load_manifest(root / "manifest.json")
    return _manifest_value(manifest, "run_id", "unknown")


def _execution_boundary(root: Path) -> str:
    path = str(root)
    if path.startswith("/Users/"):
        return "mac-mini"
    if path.startswith("/opt/") or path.startswith("/root/"):
        return "vps-or-linux"
    return "local"


def _tool_probe(name: str) -> dict[str, Any]:
    executable = shutil.which(name)
    if not executable:
        return {"status": "missing", "executable": None, "version": None}
    cmd = [executable, "--version"] if name in {"node", "npm"} else [executable, "-version"]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=5)
    except Exception as exc:  # pragma: no cover - defensive subprocess reporting
        return {"status": "error", "executable": executable, "version": None, "error": str(exc)}
    version_line = (proc.stdout or proc.stderr).splitlines()[0] if (proc.stdout or proc.stderr).splitlines() else ""
    return {"status": "present" if proc.returncode == 0 else "error", "executable": executable, "version": version_line}


def _probe_http_bridge(base_url: str) -> dict[str, Any]:
    try:
        with request.urlopen(base_url, timeout=3) as response:  # noqa: S310 - explicit user/runtime preflight URL
            return {"status": "pass", "base_url": base_url, "http_status": response.status}
    except error.HTTPError as exc:
        return {"status": "hold", "base_url": base_url, "http_status": exc.code, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - depends on user network
        return {"status": "hold", "base_url": base_url, "http_status": None, "error": str(exc)}


def _memory_probe() -> dict[str, Any]:
    if hasattr(os, "sysconf"):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_AVPHYS_PAGES")
            free_gb = round((page_size * pages) / (1024**3), 3)
            return {"free_gb": free_gb, "status": "pass" if free_gb > 1.0 else "warn"}
        except (ValueError, OSError, AttributeError):
            pass
    return {"free_gb": None, "status": "unknown"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
