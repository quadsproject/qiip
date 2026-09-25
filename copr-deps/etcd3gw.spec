%global project etcd3gw

Name:           python3-etcd3gw
Version:        2.7.0
Release:        1%{?dist}
Summary:        Thin Python client for etcd v3

License:        Apache-2.0
URL:            https://github.com/kragniz/python-etcd3gw
Source0:        https://files.pythonhosted.org/packages/source/e/%{project}/%{project}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  pyproject-rpm-macros

%description
etcd3gw is a thin, dependency-light client for etcd v3. QIIP uses it for
service discovery, registration, and the raw watch/revision/lease adapter.

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
%doc README.rst

%changelog
* Fri Sep 25 2026 quads project maintainers <noreply@github.com> - 2.7.0-1
- QIIP dependency build: Fedora 43/44 ship etcd3gw below the 2.7 floor
  (the watch/revision/lease API QIIP exercises requires 2.7).
