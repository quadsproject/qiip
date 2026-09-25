%global project fastapi

Name:           python3-fastapi
Version:        0.135.0
Release:        1%{?dist}
Summary:        FastAPI framework, high performance, easy to learn, fast to code, ready for production

License:        MIT
URL:            https://fastapi.tiangolo.com/
Source0:        https://files.pythonhosted.org/packages/source/f/%{project}/%{project}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  pyproject-rpm-macros

%description
FastAPI is a modern, fast (high-performance), web framework for building
APIs with Python 3.8+ based on standard Python type hints.

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
%{_bindir}/fastapi

%changelog
* Fri Sep 25 2026 quads project maintainers <noreply@github.com> - 0.135.0-1
- QIIP dependency build: Fedora 43 ships fastapi below the 0.135 floor
  (0.135 introduced fastapi.sse, used by the gateway).
