"""Behavior tests for the empty-FQDN guard in nginx/gen-cert.sh."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "nginx" / "gen-cert.sh"


def _run(tmp_path: Path, fqdn: str | None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, CERTS_DIR=str(tmp_path / "certs"))
    if fqdn is None:
        env.pop("QIIP_FQDN", None)
    else:
        env["QIIP_FQDN"] = fqdn
    return subprocess.run([str(SCRIPT)], env=env, capture_output=True, text=True)


def test_empty_fqdn_exits_cleanly(tmp_path: Path) -> None:
    result = _run(tmp_path, "")
    assert result.returncode == 1
    assert "QIIP_FQDN" in result.stderr
    assert "cannot generate a certificate" in result.stderr
    assert not list((tmp_path / "certs").glob("*.pem"))


def test_unset_fqdn_exits_cleanly(tmp_path: Path) -> None:
    result = _run(tmp_path, None)
    assert result.returncode == 1
    assert "QIIP_FQDN" in result.stderr
    assert "cannot generate a certificate" in result.stderr
    assert not list((tmp_path / "certs").glob("*.pem"))
