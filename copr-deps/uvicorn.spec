%global project uvicorn

Name:           python3-uvicorn
Version:        0.54.0
Release:        1%{?dist}
Summary:        ASGI web server, for Python

License:        BSD-3-Clause
URL:            https://www.uvicorn.org/
Source0:        https://files.pythonhosted.org/packages/source/u/%{project}/%{project}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  pyproject-rpm-macros

%description
Uvicorn is an ASGI web server, for Python. QIIP runs the gateway with
`python3 -m uvicorn inference_proxy.main:create_app --factory`.

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
%license LICENSE.md
%doc README.md
%{_bindir}/uvicorn

%changelog
* Fri Sep 25 2026 quads project maintainers <noreply@github.com> - 0.54.0-1
- QIIP dependency build: Fedora 43-45 ship uvicorn below the 0.45 floor.
