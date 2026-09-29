#!/bin/sh
# %post guard: exit 0 when the bundled nginx.conf may replace the stock
# /etc/nginx/nginx.conf (absent, or unmodified per the nginx-core rpmdb);
# exit 1 when the operator edited it (never clobber a user config).
# nginx.conf.default is the upstream sample, not the pristine distro file,
# so it is not a usable comparison target.
NGINX_CONF="${NGINX_CONF:-/etc/nginx/nginx.conf}"
if [ ! -f "$NGINX_CONF" ]; then
    exit 0
fi
rpm -V nginx-core 2>/dev/null | grep -q " $NGINX_CONF$" && exit 1
exit 0
