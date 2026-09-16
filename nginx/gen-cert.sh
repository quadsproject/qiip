#!/bin/sh
# QIIP self-signed certificate generator and container entrypoint.
#
# Container: ENTRYPOINT ["/gen-cert.sh"], CMD ["/usr/sbin/nginx","-g","daemon off;"]
#            generates the cert iff missing, then execs the command.
# Host one-shot: QIIP_FQDN=<fqdn> CERTS_DIR=~/.config/qiip-nginx/certs \
#                /usr/local/sbin/gen-cert.sh
#            generates the cert iff missing, then exits 0.
#
# Idempotent: an existing non-empty pair is never touched, so certs pushed by
# an internal CA (or the ansible-sslcerts playbook) survive container restarts.
set -eu
FQDN="${QIIP_FQDN:?set QIIP_FQDN (<fqdn>)}"
case "$FQDN" in
    *REPLACE_*) echo "QIIP_FQDN is still the placeholder '$FQDN' - set it to the real hostname" >&2; exit 1 ;;
esac
CERTS_DIR="${CERTS_DIR:-/etc/pki/tls/certs}"
CERT="$CERTS_DIR/$FQDN.pem"
KEY="$CERTS_DIR/$FQDN.key"
if [ ! -s "$CERT" ] || [ ! -s "$KEY" ]; then
    echo "generating self-signed pair for $FQDN into $CERTS_DIR" >&2
    export TMPDIR="${TMPDIR:-/var/cache/nginx}"   # writable tmpfs under ReadOnly=true
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
