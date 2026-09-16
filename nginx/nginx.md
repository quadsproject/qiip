# QIIP nginx reverse proxy

An optional TLS-terminating reverse proxy for the QIIP gateway. The gateway
itself listens on port 5000; nginx exposes it on 80/443 with settings tuned
for LLM inference: 50 MiB request bodies, unbuffered SSE streaming, upstream
keepalive, HTTP/2, and long generation-friendly timeouts.

Two deployment methods are documented here, both using the same
`nginx/nginx.conf`:

1. **Podman container** - rootless systemd user service (Quadlet), pasta
   networking, first-run self-signed certificate generation.
2. **RPM (dnf) install** - distribution nginx on the same host, systemd unit.

This is strictly optional. The gateway runs fine without it, and the two
methods are alternatives on one host (both bind 80/443).

## Table of contents

- [Method 1: Podman container](#method-1-podman-container)
  - [Build the image](#build-the-image)
  - [Prepare config, certificates and unit](#prepare-config-certificates-and-unit)
  - [Install and start](#install-and-start)
- [Method 2: RPM install](#method-2-rpm-install)
  - [Install nginx](#install-nginx)
  - [Deploy the config](#deploy-the-config)
  - [SELinux and firewall](#selinux-and-firewall)
  - [Install and start](#install-and-start-1)
- [Certificates](#certificates)
  - [Generate a self-signed pair](#generate-a-self-signed-pair)
  - [First-run generation in the container](#first-run-generation-in-the-container)
- [Certificate rotation](#certificate-rotation)
  - [RPM: ansible-sslcerts](#rpm-ansible-sslcerts)
  - [Container: ansible-sslcerts](#container-ansible-sslcerts)
  - [Self-signed rotation](#self-signed-rotation)
  - [Expiry checks](#expiry-checks)
- [Teardown](#teardown)
- [Verify the proxy](#verify-the-proxy)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Gateway running and healthy on port 5000
  (`curl -s http://localhost:5000/health` returns `{"status": "ok", ...}`).
- Fedora 40+ host with **nginx >= 1.25.1** (the config uses `http2 on;`,
  which older nginx rejects; RHEL 9 ships nginx 1.20-1.24 and does not
  satisfy this unless the `nginx:1.26` module stream is enabled). IPv6 is
  optional: on an IPv4-only host, remove the two `listen [::]:...` lines
  from the config (see [Troubleshooting](#troubleshooting)) - nginx fails
  to start with "Address family not supported" otherwise.
- For Method 1: podman (tested on 5.8.4) and a rootless user.
- For Method 2: sudo.

## Method 1: Podman container

Rootless podman runs the proxy as a systemd **user** unit (Quadlet) with
`pasta` networking: dual-stack, no shared host netns, no root.

### Build the image

```bash
cd /path/to/qiip
podman build -t qiip/nginx:latest -f nginx/Containerfile .
podman images --digests | grep qiip/nginx
# localhost/qiip/nginx  latest  sha256:...  <size>  <created>
```

Copy the `sha256:` digest into `Image=` in `systemd/qiip-nginx.container`
(the file ships with a `REPLACE_WITH_BUILT_IMAGE_DIGEST` placeholder).

### Prepare config, certificates and unit

Substitute the FQDN, swap the upstream to the container hostname, and plant
the config. One-time certificate generation is covered in
[Certificates](#certificates); you can skip it here because the entrypoint
creates a self-signed pair on first start (if none exists).

```bash
FQDN=$(hostname -f)                 # e.g. lab-proxy.example.com
mkdir -p ~/.config/qiip-nginx/certs ~/.config/qiip-nginx/logs
sed -E -e 's/\{FQDN\}/'"$FQDN"'/g' \
    -e 's|^([[:space:]]*server )127\.0\.0\.1:5000;|\1host.containers.internal:5000;|' \
    nginx/nginx.conf > ~/.config/qiip-nginx/nginx.conf

sed -i "s/REPLACE_WITH_HOST_FQDN/$FQDN/" systemd/qiip-nginx.container
mkdir -p ~/.config/containers/systemd
cp systemd/qiip-nginx.container ~/.config/containers/systemd/
```

### Install and start

Rootless podman cannot publish ports below 1024 by default
(`net.ipv4.ip_unprivileged_port_start=1024` on Fedora). Either allow it
(host-wide, once) or use the high-port fallback below.

```bash
sudo sysctl -w net.ipv4.ip_unprivileged_port_start=0
echo 'net.ipv4.ip_unprivileged_port_start=0' | sudo tee /etc/sysctl.d/99-qiip-nginx.conf
sudo sysctl --system

systemctl --user daemon-reload
systemctl --user enable --now qiip-nginx
sudo loginctl enable-linger "$USER"     # start at boot without a login session

systemctl --user status qiip-nginx --no-pager
podman exec qiip-nginx nginx -t         # configuration file test is successful
```

High-port fallback (no sysctl): change the two `PublishPort=` lines in the
Quadlet to `PublishPort=8443:443` (and `PublishPort=8080:80` only if you also
serve plain HTTP) and use `https://localhost:8443/...` below. Note the
config's port-80 308 redirect targets 443, so with the fallback either drop
the HTTP listener or edit the redirect target, and the [verify
commands](#verify-the-proxy) use 443 (switch to 8443).

#### Logging

nginx logs land on the container host: the Quadlet bind-mounts
`~/.config/qiip-nginx/logs` to `/var/log/nginx`, so
`access.log`/`error.log` are plain host files (same layout as an RPM
install, minus the package's logrotate). Inspect them directly and rotate
with host logrotate:

```bash
tail -f ~/.config/qiip-nginx/logs/access.log
```

```ini
# /etc/logrotate.d/qiip-nginx
/home/<deploy_user>/.config/qiip-nginx/logs/*.log {
    weekly
    rotate 4
    compress
    missingok
    notifempty
    copytruncate
}
```

`copytruncate` is used instead of a logrotate reopen signal because nginx
keeps the file descriptors open; copytruncate is simplest and loses at most
access-log buffering (flush=5s).

## Method 2: RPM install

The same `nginx/nginx.conf`, installed into the distribution nginx. Good for
hosts that already run nginx as a system service, and the path where the
[ansible-sslcerts](#rpm-ansible-sslcerts) playbook's normal handler
(restart of the system `nginx` unit) works unchanged.

Steps for RPM install:

1. Install nginx
2. Deploy the config
3. SELinux and firewall
4. Install and start

### Install nginx

```bash
sudo dnf install -y nginx openssl
# nginx needs its temp dirs; the package does not create /var/cache/nginx.
sudo install -d -o nginx -g nginx /var/cache/nginx
```

### Deploy the config

Fetch the repo config (or copy it from a checkout), substitute the FQDN, and
remove the package's stock virtual host so its `listen 80` does not collide
with the config's port-80 redirect. The upstream stays `127.0.0.1:5000` for
bare-metal nginx.

```bash
FQDN=$(hostname -f)

sudo curl -fsSL -o /etc/nginx/nginx.conf \
  https://raw.githubusercontent.com/quadsproject/qiip/main/nginx/nginx.conf
sudo sed -i "s/{FQDN}/$FQDN/g" /etc/nginx/nginx.conf

sudo rm -f /etc/nginx/conf.d/*.conf /etc/nginx/default.d/*.conf
```

From a checkout, replace the `curl` line with
`sudo cp nginx/nginx.conf /etc/nginx/nginx.conf`.

### SELinux and firewall

nginx is confined to `httpd_t`; proxying to the gateway port needs the
network-connect boolean. Open 80/443 and keep 5000 closed externally (the
`/v1/*` inference endpoints are unauthenticated).

```bash
sudo setsebool -P httpd_can_network_connect on
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### Install and start

```bash
sudo nginx -t
sudo systemctl enable --now nginx
```

## Certificates

Certificate and key live in the same directory the
[sadsfae/ansible-sslcerts](https://github.com/sadsfae/ansible-sslcerts#apache-and-nginx-files-locations)
playbook uses for nginx, so playbook-pushed certs work without config edits:

- Path inside the container: `/etc/pki/tls/certs/<servername>.pem` and
  `/etc/pki/tls/certs/<servername>.key` (the Quadlet mounts
  `~/.config/qiip-nginx/certs` there).
- Path for RPM: `/etc/pki/tls/certs/<servername>.pem` and `.key`.

`<servername>` in the config is the `{FQDN}` placeholder; it must match the
file names (see the naming contract in [rotation](#container-ansible-sslcerts)).

Note: `/etc/pki/tls/certs` is also the system CA trust-store directory on
Fedora. These cert/key files are a web-server pair placed there for
(ansible-sslcerts-compatible) convenience only; `update-ca-trust` does not
add them to the bundle and they are not part of the system trust store.

### Generate a self-signed pair

One shared script does both methods: it is the container entrypoint and a
host one-shot. Install it on an RPM host with `cat` (visible) or `curl`
(the block below is a copy of `nginx/gen-cert.sh`; keep it in sync):

```bash
sudo tee /usr/local/sbin/gen-cert.sh >/dev/null <<'EOF'
#!/bin/sh
# QIIP self-signed certificate generator and container entrypoint.
#
# Container: ENTRYPOINT ["/gen-cert.sh"], CMD ["/usr/sbin/nginx","-g","daemon off;"]
#            generates the cert iff missing, then execs the command.
# Host one-shot: QIIP_FQDN=<fqdn> CERTS_DIR=<dir> /usr/local/sbin/gen-cert.sh
#            generates the cert iff missing, then exits 0.
#
# Idempotent: an existing non-empty pair is never touched, so certs pushed by
# an internal CA (or the ansible-sslcerts playbook) survive container restarts.
set -eu
FQDN="${QIIP_FQDN:?set QIIP_FQDN (<fqdn>)}"
CERTS_DIR="${CERTS_DIR:-/etc/pki/tls/certs}"
CERT="$CERTS_DIR/$FQDN.pem"
KEY="$CERTS_DIR/$FQDN.key"
if [ ! -s "$CERT" ] || [ ! -s "$KEY" ]; then
    export TMPDIR="${TMPDIR:-/var/cache/nginx}"
    openssl req -x509 -newkey rsa:4096 -quiet \
        -keyout "$KEY" -out "$CERT" \
        -days 3650 -nodes \
        -subj "/CN=$FQDN" -addext "subjectAltName=DNS:$FQDN"
    chmod 600 "$KEY"
    chmod 644 "$CERT"
fi
if [ "$#" -gt 0 ]; then
    # Container mode: the /var/cache/nginx tmpfs starts empty and root-owned;
    # nginx master (uid 0) creates the temp dirs, workers (nginx user) must be
    # able to write into them. Matches the RPM package ownership.
    mkdir -p /var/cache/nginx/client_temp /var/cache/nginx/proxy_temp \
             /var/cache/nginx/fastcgi_temp /var/cache/nginx/uwsgi_temp \
             /var/cache/nginx/scgi_temp
    chown -R nginx:nginx /var/cache/nginx
    exec "$@"
fi
exit 0
EOF
sudo chmod 0555 /usr/local/sbin/gen-cert.sh
```

(Or once: `sudo curl -fsSL -o /usr/local/sbin/gen-cert.sh https://raw.githubusercontent.com/quadsproject/qiip/main/nginx/gen-cert.sh`.)

Run it. The script is idempotent: an existing pair is never touched, so it is
safe to re-run after the ansible playbook has pushed an internal-CA pair.

```bash
FQDN=$(hostname -f)
sudo env QIIP_FQDN="$FQDN" CERTS_DIR=/etc/pki/tls/certs /usr/local/sbin/gen-cert.sh
openssl x509 -in /etc/pki/tls/certs/$FQDN.pem -noout -subject -enddate
# subject=CN = <fqdn>; notAfter = 10 years out
```

The generated pair is valid 10 years (3650 days), RSA 4096, CN and SAN set to
the FQDN, key 0600, cert 0644.

### First-run generation in the container

The container entrypoint runs the same script before nginx starts: with no
certs present it generates the self-signed pair into the mounted cert dir
(persisted on the host as the podman user); with a pair present (self-signed
or internal CA) it is left untouched.

```bash
podman exec qiip-nginx ls -l /etc/pki/tls/certs/
```

## Certificate rotation

### RPM: ansible-sslcerts

For distribution nginx the upstream playbook works as shipped: drop
`<hostname>.pem`/`.key` into `install/roles/sslcerts/files/`, run the
playbook, and its `Reload Nginx` handler restarts the system `nginx` service.
Host must be in an inventory group named `nginx` and files must be named
after `ansible_nodename`.

```bash
cp new.pem install/roles/sslcerts/files/$(hostname -f).pem
cp new.key install/roles/sslcerts/files/$(hostname -f).key
ansible-playbook -i hosts install/sslcerts.yml
```

### Container: ansible-sslcerts

The same playbook pushes certs to the container using per-host overrides;
no new inventory group or wrapper playbook is needed. The role's default
destination (`/etc/pki/tls/certs`, root:root 0600) is not readable by a
rootless container, and its reload handler restarts the system `nginx` unit
(which does not exist here). Both are solved by the upstream
`sslcerts_nginx_reload_command` variable
([sadsfae/ansible-sslcerts#18](https://github.com/sadsfae/ansible-sslcerts/pull/18))
plus `host_vars` overrides.

1. List the host under the existing `[nginx]` group in the playbook's
   `hosts` inventory, exactly like a normal nginx host (the shipped
   inventory has a commented `#host04` placeholder).

2. Create `install/host_vars/<inventory-hostname>.yml` in the playbook
   checkout, named after the host as written in the inventory (the shipped
   example is `install/host_vars/host04.yaml`):

 ```yaml
 ---
 sslcerts_owner: <deploy_user>
 sslcerts_group: <deploy_user>
 sslcerts_mode: "0600"
 nginx_cert_path: /home/<deploy_user>/.config/qiip-nginx/certs
 nginx_key_path: /home/<deploy_user>/.config/qiip-nginx/certs
 sslcerts_nginx_reload_command: systemctl --user --machine=<deploy_user>@.host restart qiip-nginx
 ```

 Quote the mode: an unquoted YAML `0600` is octal and renders as a decimal
 integer.

The reload command restarts the user unit when the key changes. It must be a
single command with arguments (the role's handler uses `ansible.builtin.command`,
no shell operators). Restart, not
`nginx -s reload`: the restart re-mounts the cert dir so the SELinux `:Z`
label is re-applied to files the playbook wrote after the container started.
`systemctl --user --machine=<user>@.host` runs from a root context (the
playbook connects as root); the equivalent from a root shell is
`runuser -u <deploy_user> -- systemctl --user restart qiip-nginx`.

Naming contract: the cert/key files in the role's `files/` directory are
named after `ansible_nodename` (the system hostname), so `{FQDN}` in the
nginx config (and `QIIP_FQDN` in the Quadlet) must equal it. The `host_vars`
file is keyed by the inventory hostname, which may differ. If the system
hostname and FQDN differ, nginx fails with
`cannot load certificate ... BIO_new_file() failed`.

Gotcha: the role notifies the reload handler only on the **key** copy
(`tasks/main.yml`). A cert-only renewal with an unchanged key does not
trigger the restart; re-run the wrapper or restart the unit once.

### Self-signed rotation

The 10-year pair outlives the lab; when it needs replacing, delete it and let
the entrypoint regenerate (container), or re-run the host one-shot (RPM):

```bash
# container
rm ~/.config/qiip-nginx/certs/$FQDN.pem ~/.config/qiip-nginx/certs/$FQDN.key
systemctl --user restart qiip-nginx

# RPM
sudo rm /etc/pki/tls/certs/$FQDN.pem /etc/pki/tls/certs/$FQDN.key
sudo env QIIP_FQDN="$FQDN" CERTS_DIR=/etc/pki/tls/certs /usr/local/sbin/gen-cert.sh
sudo systemctl reload nginx
```

### Expiry checks

```bash
openssl x509 -in /etc/pki/tls/certs/$FQDN.pem -noout -enddate
# container host-side copy:
openssl x509 -in ~/.config/qiip-nginx/certs/$FQDN.pem -noout -enddate
```

Set an alert 90 days before expiry (typical CA practice; self-signed runs 10
years so this is a lifetime away).

## Teardown

Container method:

```bash
systemctl --user disable --now qiip-nginx
rm ~/.config/containers/systemd/qiip-nginx.container
systemctl --user daemon-reload
loginctl disable-linger "$USER"          # optional
sudo rm /etc/sysctl.d/99-qiip-nginx.conf # only if you set the sysctl
sudo sysctl --system
rm -rf ~/.config/qiip-nginx              # config, certs, logs
podman rmi localhost/qiip/nginx:latest   # optional image cleanup
```

RPM method:

```bash
sudo systemctl disable --now nginx
sudo dnf remove -y nginx                 # or: sudo dnf reinstall nginx
# reinstall restores the stock /etc/nginx/nginx.conf and the conf.d/default.d
# files that this setup replaced/removed
```

## Verify the proxy

```bash
# TLS handshake and identity
openssl s_client -connect localhost:443 -servername $FQDN </dev/null 2>/dev/null | openssl x509 -noout -subject -issuer -dates

# Gateway health through the proxy (port 80 is a 308; use HTTPS, -k for the
# lab self-signed cert)
curl -k -s https://localhost/health

# Dashboard redirect
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' -k https://localhost/
# 302 https://localhost/dashboard

# Streaming API (chunks arrive immediately; proves proxy_buffering off + gzip off)
curl -N -k https://localhost/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model>","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# HTTP redirects to HTTPS (308 preserves method and body)
curl -sI http://localhost/       # 308 Location: https://localhost/
```

For the container, nginx logs are host files under
`~/.config/qiip-nginx/logs/` (see [Logging](#logging)); for RPM they are
`/var/log/nginx/` (rotated by the package's logrotate).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `413 Request Entity Too Large` | Body over the deliberate 50 MiB cap (OMP archived image frames fit under it) | Raise `client_max_body_size` in the config (bounded step, e.g. 64m), `nginx -t`, restart. |
| `502 Bad Gateway` | Gateway down, or wrong upstream | Confirm the gateway is up: `curl -s http://localhost:5000/health`, and restart how you run it (README Quick Start). Container: confirm the planted config has `host.containers.internal:5000`. |
| `504 Gateway Timeout` | `proxy_read_timeout 3600` exceeded or upstream wedged | Fix/hold the node; only raise the timeout for legitimately very long generations. |
| SSE arrives in bursts | `proxy_buffering` on or `gzip` on | Confirm `proxy_buffering off`, `X-Accel-Buffering no always`, `gzip off`. |
| `Permission denied` on the cert | SELinux label missing | Ensure `:Z` on the `Volume=`/mount; `sudo chcon -R -t container_file_t ~/.config/qiip-nginx/certs`, then restart. |
| `bind: address already in use` on 80/443 | Bare-metal nginx still runs | This proxy replaces it: `sudo systemctl stop nginx && sudo systemctl disable nginx`, then restart the new unit. |
| Container exits: `mkdir() /var/lib/nginx/tmp/client_body failed (13: Permission denied)` | Read-only rootfs without the nginx temp dirs | The Quadlet ships `Tmpfs=/var/cache/nginx`; the config sets temp paths under `/var/cache/nginx`. |
| `nginx -t`/start: `socket() [::]:443 failed (97: Address family not supported by protocol)` | Host has no IPv6; the `listen [::]:...` lines are fatal without IPv6 | Strip the `[::]` lines: `sudo sed -i '/listen \[::\]/d' <config>` (`/etc/nginx/nginx.conf` or `~/.config/qiip-nginx/nginx.conf`), then `nginx -t` and restart. |
| `curl http://host/health` returns 308 | Intentional: port 80 redirects to HTTPS | Use `https://host/health` (with `-k` for self-signed). |

## References

- Config source: `nginx/nginx.conf` (same file for both methods; the
  container deploy substitutes only `{FQDN}` and the upstream hostname).
- Certificate layout: [sadsfae/ansible-sslcerts](https://github.com/sadsfae/ansible-sslcerts#apache-and-nginx-files-locations).
- RFE: https://github.com/quadsproject/qiip/issues/116
