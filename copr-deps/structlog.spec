%global project structlog

Name:           python3-%{project}
Version:        26.1.0
Release:        1%{?dist}
Summary:        Structured logging for Python

License:        Apache-2.0 OR MIT
URL:            https://www.structlog.org/
# PyPI sdist, checked out by the COPR build (or vendored below).
Source0:        https://files.pythonhosted.org/packages/source/s/%{project}/%{project}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  pyproject-rpm-macros

%description
structlog makes structured logging in Python easy: it renders key-value pairs
instead of string interpolation, supports a POSIX-inspired logger, is free of
magic, and works out of the box with stdlib logging.

%prep
%autosetup -n %{project}-%{version}

%generate_buildrequires
%pyproject_buildrequires

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files -l %{project}

# The structlog sdist ships no LICENSE file; the license is declared in metadata.
%files -f %{pyproject_files}
%doc README.md CHANGELOG.md

%changelog
* Fri Sep 25 2026 quads project maintainers <noreply@github.com> - 26.1.0-1
- QIIP dependency build: structlog is not packaged in Fedora 43-45.
