%global project click

Name:           python3-click
Version:        8.5.0
Release:        1%{?dist}
Summary:        Composable command line interface toolkit

License:        BSD-3-Clause
URL:            https://palletsprojects.com/p/click/
Source0:        https://files.pythonhosted.org/packages/source/c/%{project}/%{project}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  pyproject-rpm-macros

%description
Click is a Python package for creating beautiful command line interfaces in a
composable way with as little code as necessary. Built here because
huggingface-hub >= 1.25 requires click >= 8.4.2, which is newer than the
Fedora 43/44 package.

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
%license LICENSE.txt
%doc README.md

%changelog
* Fri Sep 25 2026 quads project maintainers <noreply@github.com> - 8.5.0-1
- QIIP dependency build: huggingface-hub 1.25 requires click >= 8.4.2.
