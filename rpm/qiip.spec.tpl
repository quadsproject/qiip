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
# QUADS-style serving: nginx terminates TLS (nginx.conf uses `http2 on;`,
# nginx >= 1.25.1; gen-cert.sh shells out to openssl), uvicorn serves the
# app with explicit worker tuning, and setsebool lives in policycoreutils.
Requires:       nginx >= 1.25.1
Requires:       openssl
Requires:       policycoreutils
# etcd is the discovery/registry backend (inference_proxy/discovery/).
# etcd3gw (the thin client) declares no server floor, so name the supported
# branch here: 3.5 is the maintained line (3.4 is EOL) and the current docs
# pin 3.5.21. Installed but not started/required at runtime for a remote
# cluster, matching the "gateway starts during an etcd outage" contract.
Requires:       etcd >= 3.5.0
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
chmod 0755 %{buildroot}%{_datadir}/qiip/nginx/gen-cert.sh \
    %{buildroot}%{_datadir}/qiip/nginx/nginx-deploy-conf.sh

install -d -m 0755 %{buildroot}%{_unitdir}
install -m 0644 rpm/inference-proxy.service %{buildroot}%{_unitdir}/inference-proxy.service

# Writable runtime data location (matches the shipped unit's overrides).
install -d -m 0755 %{buildroot}%{_localstatedir}/lib/qiip
# Admin config: examples are never loaded (the loader matches .yml/.yaml),
# and qiip.env is the secrets file the unit reads. Server tuning lives in
# the server: YAML block (or INFERENCE_PROXY_SERVER__* overrides here).
install -d -m 0755 %{buildroot}%{_sysconfdir}/qiip/conf
install -m 0644 conf/qiip.yml.example conf/auth.yml.example conf/plugins.yml.example \
    %{buildroot}%{_sysconfdir}/qiip/conf/
cat > %{buildroot}%{_sysconfdir}/qiip/qiip.env <<'EOF'
# Secrets and INFERENCE_PROXY_* overrides go here. Server tuning is set in
# the server: YAML block; override per host with e.g.
# INFERENCE_PROXY_SERVER__WORKERS=1
EOF

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
# QUADS-style nginx integration: nginx is a hard dependency and is managed
# here. Deploy the bundled config once (do not clobber a user's edited
# /etc/nginx/nginx.conf on upgrade), generate a cert if none exists, allow
# nginx to proxy to the gateway over SELinux, then enable and start it.
FQDN="$(hostname -f 2>/dev/null || hostname)"
# The nginx package ships /etc/nginx/nginx.conf (%config noreplace). rpm -V
# flags it only when the operator edited it, so deploy the bundled config when
# the stock file is unmodified (or absent); a user-modified config is never
# clobbered. nginx.conf.default is the upstream sample, not the pristine
# distro file, so it is not a usable comparison target.
if [ -f /usr/share/qiip/nginx/nginx.conf ] && \
   /usr/share/qiip/nginx/nginx-deploy-conf.sh; then
    sed -e "s/{FQDN}/$FQDN/g" /usr/share/qiip/nginx/nginx.conf > /etc/nginx/nginx.conf
fi
QIIP_FQDN="$FQDN" /usr/share/qiip/nginx/gen-cert.sh >/dev/null 2>&1 || \
    echo "qiip: cert generation failed; install a pair in /etc/pki/tls/certs before starting nginx" >&2
setsebool httpd_can_network_connect 1 -P 2>/dev/null || :
# nginx needs its temp dirs; the nginx package does not create them.
if [ ! -d /var/cache/nginx ]; then
    install -d -o nginx -g nginx -m 0755 /var/cache/nginx
fi
restorecon -R /var/cache/nginx 2>/dev/null || :
if /usr/sbin/nginx -t >/dev/null 2>&1; then
    # Start on first install only; upgrades just (re)enable, never restart a
    # stopped nginx behind the operator's back.
    if [ "${1:-1}" -eq 1 ]; then
        systemctl enable --now nginx >/dev/null 2>&1 || :
    else
        systemctl enable nginx >/dev/null 2>&1 || :
        systemctl is-active --quiet nginx 2>/dev/null || \
            echo "qiip: nginx is enabled but not running; start it with systemctl start nginx" >&2
    fi
else
    echo "qiip: nginx -t failed; fix /etc/nginx/nginx.conf then run systemctl enable --now nginx" >&2
fi
%systemd_post inference-proxy.service
# First install: also start the gateway (the macro above only enables), so
# nginx does not sit on a dead upstream. Upgrades leave the running state alone.
if [ "${1:-1}" -eq 1 ] && systemctl is-enabled --quiet inference-proxy.service 2>/dev/null; then
    systemctl start inference-proxy.service >/dev/null 2>&1 || \
        echo "qiip: inference-proxy failed to start; check systemctl status inference-proxy" >&2
fi

%preun
%systemd_preun inference-proxy.service

%postun
%systemd_postun inference-proxy.service

%changelog
* @DATE@ quads project maintainers <noreply@github.com> - @VERSION@-@RELEASE@
- package generated from upstream, changelog not tracked
