%global org quadsproject
%global sum QUADS Idle Inference Proxy - routes LLM inference requests to vLLM and llama.cpp nodes
%global desc QIIP is a QUADS-native inference framework that automates \
installation, drivers, setup, and presentation of free or idle NVIDIA GPU \
systems through one OpenAI-compatible inference API. It provides a gateway \
service that proxies requests to inference nodes, discovers backends via \
etcd, health-checks them, and routes with automatic failover. nginx \
terminates TLS in front of the gateway.

Name:           @NAME@
Version:        @VERSION@
Release:        @RELEASE@%{?dist}
Summary:        %{sum}

License:        GPL-3.0-or-later and MIT and Apache-2.0
URL:            https://github.com/%{org}/qiip
# GitHub tag archive. The Makefile builds the same tarball locally and embeds
# it in the SRPM, so COPR never fetches this URL; it is the canonical source.
Source0:        %{url}/archive/refs/tags/v@GIT_VERSION@.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  pyproject-rpm-macros
BuildRequires:  systemd-rpm-macros

Requires:       python3 >= 3.12
# Bundled nginx stack: nginx.conf uses `http2 on;` (nginx >= 1.25.1);
# gen-cert.sh shells out to openssl.
Requires:       nginx >= 1.25.1
Requires:       openssl
@CONFLICTS@

%description
%{desc}

%prep
%autosetup -n qiip-v@GIT_VERSION@

%generate_buildrequires
%pyproject_buildrequires

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files -l inference_proxy

# Node-side engine bundles and shared helpers (uploaded to GPU nodes by the
# gateway; resolved relative to the working directory at runtime).
install -d -m 0755 %{buildroot}%{_datadir}/qiip
cp -a auto-vllm auto-llamacpp common conf nginx %{buildroot}%{_datadir}/qiip/
chmod 0755 %{buildroot}%{_datadir}/qiip/nginx/gen-cert.sh

install -d -m 0755 %{buildroot}%{_unitdir}
install -m 0644 rpm/inference-proxy.service %{buildroot}%{_unitdir}/inference-proxy.service

# Writable runtime data location (matches the shipped unit's overrides).
install -d -m 0755 %{buildroot}%{_localstatedir}/lib/qiip
# Admin config: examples are never loaded (the loader matches .yml/.yaml),
# and qiip.env is the secrets file the unit reads.
install -d -m 0755 %{buildroot}%{_sysconfdir}/qiip/conf
install -m 0644 conf/qiip.yml.example conf/auth.yml.example conf/plugins.yml.example \
    %{buildroot}%{_sysconfdir}/qiip/conf/
: > %{buildroot}%{_sysconfdir}/qiip/qiip.env

%check
# The full suite runs in CI (Node 24 + pinned dev deps); %check is empty so
# the RPM build stays buildable from system packages only.

%files -f %{pyproject_files}
%license LICENSE
%doc README.md UPGRADING.md docs/releases.md
%dir %{_datadir}/qiip
%{_datadir}/qiip/auto-vllm
%{_datadir}/qiip/auto-llamacpp
%{_datadir}/qiip/common
%{_datadir}/qiip/conf
%{_datadir}/qiip/nginx
%{_unitdir}/inference-proxy.service
%dir %{_localstatedir}/lib/qiip
%dir %{_sysconfdir}/qiip
%dir %{_sysconfdir}/qiip/conf
%config(noreplace) %{_sysconfdir}/qiip/conf/qiip.yml.example
%config(noreplace) %{_sysconfdir}/qiip/conf/auth.yml.example
%config(noreplace) %{_sysconfdir}/qiip/conf/plugins.yml.example
%config(noreplace) %{_sysconfdir}/qiip/qiip.env

%post
# nginx needs its temp dirs; the nginx package does not create them.
if [ ! -d /var/cache/nginx ]; then
    install -d -o nginx -g nginx -m 0755 /var/cache/nginx
fi
restorecon -R /var/cache/nginx 2>/dev/null || :
%systemd_post inference-proxy.service

%preun
%systemd_preun inference-proxy.service

%postun
%systemd_postun inference-proxy.service

%changelog
* @DATE@ quads project maintainers <noreply@github.com> - @VERSION@-@RELEASE@
- package generated from upstream, changelog not tracked
