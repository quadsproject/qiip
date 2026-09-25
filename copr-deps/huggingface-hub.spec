%global project huggingface_hub

Name:           python3-huggingface-hub
Version:        1.25.0
Release:        1%{?dist}
Summary:        Client library to download and publish models and other files on the Hugging Face Hub

License:        Apache-2.0
URL:            https://huggingface.co/docs/huggingface_hub
Source0:        https://files.pythonhosted.org/packages/source/h/%{project}/%{project}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  pyproject-rpm-macros

# Runtime Requires come from the wheel metadata (click, httpx, filelock,
# fsspec, packaging, pyyaml, tqdm, typing-extensions). hf-xet (a Rust
# extension, unconditional on x86_64 upstream) is deliberately filtered out:
# no Fedora provider, and QIIP only downloads models, for which
# huggingface_hub falls back to plain HTTPS with a warning when hf_xet is
# absent. The Xet upload path is not used by QIIP.
%global __requires_exclude ^python3(\.\d+)?dist\(hf-xet\)

%description
huggingface-hub is the client library used to download, upload, and manage
models, datasets, and spaces on the Hugging Face Hub.

%prep
%autosetup -n %{project}-%{version}

%generate_buildrequires
%pyproject_buildrequires

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files -l %{project}

%files -f %{pyproject_files}
%license LICENSE
%doc README.md
%{_bindir}/hf
%{_bindir}/huggingface-cli
%{_bindir}/tiny-agents

%changelog
* Fri Sep 25 2026 quads project maintainers <noreply@github.com> - 1.25.0-1
- QIIP dependency build: Fedora 43/44 ship huggingface-hub below the 1.25 floor.
