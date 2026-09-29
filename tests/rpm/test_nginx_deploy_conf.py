"""Behavior tests for the %post nginx deploy guard (nginx/nginx-deploy-conf.sh)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "nginx" / "nginx-deploy-conf.sh"


def _run(tmp_path: Path, conf_exists: bool, modified: bool = False) -> int:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    conf = tmp_path / "nginx.conf"
    rpm_output = ""
    if conf_exists:
        conf.write_text("stock\n", encoding="utf-8")
        if modified:
            rpm_output = f"S.5....T. c {conf}"
    fake_rpm = bin_dir / "rpm"
    fake_rpm.write_text(f"#!/bin/sh\necho '{rpm_output}'\n", encoding="utf-8")
    fake_rpm.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ.get('PATH', '')}",
        NGINX_CONF=str(conf),
    )
    return subprocess.run(
        [str(SCRIPT)], env=env, capture_output=True, text=True
    ).returncode


def test_deploys_when_conf_missing(tmp_path: Path) -> None:
    assert _run(tmp_path, conf_exists=False) == 0


def test_deploys_when_stock_conf_unmodified(tmp_path: Path) -> None:
    assert _run(tmp_path, conf_exists=True) == 0


def test_never_clobbers_modified_conf(tmp_path: Path) -> None:
    assert _run(tmp_path, conf_exists=True, modified=True) == 1
