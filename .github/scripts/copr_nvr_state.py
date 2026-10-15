#!/usr/bin/env python3
"""Report whether a Copr NVR is published in the quadsdev/qiip repository.

Copr keeps one build per package version and names RPM files after the NVRA,
so every build of an NVR shares a single file. Removing any of those builds,
and Copr's own retention when it drops one of two builds of the same version,
takes the RPM and the SRPM with it: the package disappears from the
repository. The release workflows therefore publish an NVR once and skip or
refuse a second build of the same NVR.

Modes:
  state <name> <nvr>       print published, stale or absent for an NVR that
                           carries its dist tag, for example
                           qiip-dev-0.2.0~dev.14-1.fc43; stale also reports
                           the existing build ids and states on stderr
  names <chroot> <name>... print the names without a fetchable binary RPM in
                           the chroot, exit 1 when any name is missing
  chroots                  print the Fedora chroots the project builds
  --self-test              run the offline metadata parsing checks

published  every project chroot serves the binary RPM of the NVR and its file
           is fetchable
stale      a Copr build exists for the NVR but a chroot does not serve it
absent     neither a Copr build nor a file exists for the NVR
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import urllib.error
import urllib.request

PROJECT = "https://download.copr.fedorainfracloud.org/results/quadsdev/qiip"
API = "https://copr.fedorainfracloud.org/api_3"
TIMEOUT = 60
BUILD_LIMIT = 200

REPOMD = (
    b'<repomd><data type="primary">'
    b'<location href="repodata/abc-primary.xml.gz"/></data></repomd>'
)
PRIMARY = (
    b"<metadata>"
    b'<package type="rpm">'
    b"<name>python3-click</name><arch>noarch</arch>"
    b'<version epoch="0" ver="8.5.0" rel="1.fc43"/>'
    b'<location href="Packages/p/python3-click-8.5.0-1.fc43.noarch.rpm"/>'
    b"</package>"
    b'<package type="rpm">'
    b"<name>python3-click</name><arch>src</arch>"
    b'<version epoch="0" ver="8.5.0" rel="1.fc43"/>'
    b'<location href="Packages/p/python3-click-8.5.0-1.fc43.src.rpm"/>'
    b"</package>"
    b"</metadata>"
)


def fetch(url: str, headers: dict[str, str] | None = None) -> bytes:
    """Return the body of url, following redirects, bypassing shared caches."""
    request = urllib.request.Request(
        url, headers={"Cache-Control": "no-cache", **(headers or {})}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return bytes(response.read())


def primary_href(repomd: bytes) -> str:
    """Return the relative path of the primary metadata file in repomd.xml."""
    match = re.search(rb'<location href="(repodata/[^"]*primary\.xml\.gz)"', repomd)
    if match is None:
        raise SystemExit("no primary metadata declared in repomd.xml")
    return match.group(1).decode()


def binary_packages(primary: bytes) -> list[tuple[str, str, str]]:
    """Return (NVR, name, location) for every binary RPM in primary metadata.

    Source RPMs are skipped: they share the NVR of the package they were built
    from, and only a binary RPM makes an NVR installable.
    """
    packages: list[tuple[str, str, str]] = []
    for package in re.findall(rb'<package type="rpm">(.*?)</package>', primary, re.S):
        arch = re.search(rb"<arch>([^<]+)</arch>", package)
        if arch and arch.group(1) == b"src":
            continue
        name = re.search(rb"<name>([^<]+)</name>", package)
        version = re.search(
            rb'<version epoch="[^"]*" ver="([^"]+)" rel="([^"]+)"/>', package
        )
        location = re.search(rb'<location href="([^"]+)"/>', package)
        if not (name and version and location):
            raise SystemExit("unexpected package entry in the repository metadata")
        nvr = b"-".join((name.group(1), version.group(1), version.group(2)))
        packages.append(
            (nvr.decode(), name.group(1).decode(), location.group(1).decode())
        )
    return packages


def candidates_by_name(entries: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    """Group metadata locations by binary RPM package name."""
    grouped: dict[str, list[str]] = {}
    for _nvr, name, location in entries:
        grouped.setdefault(name, []).append(location)
    return grouped


def fetch_primary(chroot: str) -> bytes:
    """Return the primary metadata of one chroot's repository."""
    base = f"{PROJECT}/{chroot}"
    href = primary_href(fetch(f"{base}/repodata/repomd.xml"))
    return gzip.decompress(fetch(f"{base}/{href}"))


def dist_of(chroot: str) -> str:
    """Return the dist tag of a Fedora chroot, for example fc43."""
    dist = "fc" + chroot.removeprefix("fedora-").removesuffix("-x86_64")
    if not re.fullmatch(r"fc\d+", dist):
        raise SystemExit(f"unexpected chroot name: {chroot!r}")
    return dist


def version_release_of(name: str, nvr: str) -> str:
    """Return version-release without the dist tag, as the Copr API reports it."""
    stem = nvr.removeprefix(f"{name}-")
    version_release, _, dist = stem.rpartition(".")
    if stem == nvr or not re.fullmatch(r"fc\d+", dist):
        raise SystemExit(f"not a {name} NVR with a fedora dist tag: {nvr!r}")
    return version_release


def build_states(name: str, version_release: str) -> list[str]:
    """Return '<id> <state>' for every Copr build of this version-release."""
    url = (
        f"{API}/build/list?ownername=quadsdev&projectname=qiip"
        f"&packagename={name}&order=id&order_type=DESC&offset=0&limit={BUILD_LIMIT}"
    )
    items = json.loads(fetch(url))["items"]
    if len(items) >= BUILD_LIMIT:
        raise SystemExit(f"the build list for {name} is truncated")
    return [
        f"{build['id']} {build['state']}"
        for build in items
        if build["source_package"].get("version") == version_release
    ]


def file_exists(url: str) -> bool:
    """Return whether the artifact at url can be fetched; 404 means no file."""
    try:
        fetch(url, {"Range": "bytes=0-0"})
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            return False
        raise
    return True


def chroot_serves(name: str, version_release: str, chroot: str) -> bool:
    """Return whether this chroot serves the binary RPM of one version-release."""
    nvr = f"{name}-{version_release}.{dist_of(chroot)}"
    for entry_nvr, _entry_name, location in binary_packages(fetch_primary(chroot)):
        if entry_nvr == nvr:
            return file_exists(f"{PROJECT}/{chroot}/{location}")
    return False


def state(name: str, nvr: str) -> str:
    """Return published, stale or absent for one NVR, across every chroot."""
    version_release = version_release_of(name, nvr)
    if all(chroot_serves(name, version_release, chroot) for chroot in chroots()):
        return "published"
    builds = build_states(name, version_release)
    return "stale" if builds else "absent"


def missing_names(chroot: str, names: list[str]) -> list[str]:
    """Return the names without a fetchable binary RPM in this chroot."""
    grouped = candidates_by_name(binary_packages(fetch_primary(chroot)))
    return [
        name
        for name in names
        if not any(
            file_exists(f"{PROJECT}/{chroot}/{location}")
            for location in grouped.get(name, [])
        )
    ]


def chroots() -> list[str]:
    """Return the Fedora chroots the project builds."""
    url = f"{API}/project?ownername=quadsdev&projectname=qiip"
    built = json.loads(fetch(url)).get("chroot_repos", {})
    return sorted(
        chroot for chroot in built if re.fullmatch(r"fedora-\d+-x86_64", chroot)
    )


def self_test() -> int:
    """Check the metadata parsing and NVR arithmetic without network access."""
    assert primary_href(REPOMD) == "repodata/abc-primary.xml.gz"
    assert binary_packages(PRIMARY) == [
        (
            "python3-click-8.5.0-1.fc43",
            "python3-click",
            "Packages/p/python3-click-8.5.0-1.fc43.noarch.rpm",
        )
    ]
    assert candidates_by_name(binary_packages(PRIMARY)) == {
        "python3-click": ["Packages/p/python3-click-8.5.0-1.fc43.noarch.rpm"]
    }
    assert dist_of("fedora-43-x86_64") == "fc43"
    assert dist_of("fedora-44-x86_64") == "fc44"
    assert version_release_of("qiip-dev", "qiip-dev-0.2.0~dev.15-1.fc43") == (
        "0.2.0~dev.15-1"
    )
    assert (
        version_release_of("python3-click", "python3-click-8.5.0-1.fc43") == "8.5.0-1"
    )
    for bad in ("qiip-dev-1.0.0", "qiip-dev-1.0.0-1", "python3-click-1.0.0-1.fc43"):
        try:
            version_release_of("qiip-dev", bad)
        except SystemExit:
            continue
        raise AssertionError(f"a foreign NVR must be rejected: {bad}")
    print("copr_nvr_state self-test ok")
    return 0


def main(argv: list[str]) -> int:
    """Dispatch the requested mode."""
    if argv[1:2] == ["--self-test"]:
        return self_test()
    if argv[1:2] == ["chroots"] and len(argv) == 2:
        print("\n".join(chroots()))
        return 0
    if argv[1:2] == ["state"] and len(argv) == 4:
        name, nvr = argv[2], argv[3]
        result = state(name, nvr)
        print(result)
        if result == "stale":
            builds = build_states(name, version_release_of(name, nvr))
            print(f"copr builds for {nvr}: {', '.join(builds)}", file=sys.stderr)
        return 0
    if argv[1:2] == ["names"] and len(argv) >= 4:
        missing = missing_names(argv[2], argv[3:])
        if missing:
            print(" ".join(missing))
            return 1
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
