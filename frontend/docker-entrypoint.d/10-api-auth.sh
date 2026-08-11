#!/bin/sh
# Write the nginx snippet that authenticates proxied /v1 requests.
#
# The bearer token stays in the nginx container: it is read from the environment
# at start-up and written to a config file that is never served as a static
# asset. Putting it in the JS bundle would publish it to every visitor, and the
# browser EventSource API cannot send an Authorization header at all, so the
# reverse proxy is the only place this can live.
set -eu

SNIPPET_DIR=/etc/nginx/snippets
SNIPPET="$SNIPPET_DIR/api-auth.conf"

mkdir -p "$SNIPPET_DIR"

if [ -n "${AGENT_EVALS_API__API_KEY:-}" ]; then
    # Escape the two characters nginx treats specially inside a quoted string.
    escaped=$(printf '%s' "$AGENT_EVALS_API__API_KEY" | sed 's/\\/\\\\/g; s/"/\\"/g')
    printf 'proxy_set_header Authorization "Bearer %s";\n' "$escaped" > "$SNIPPET"
    echo "api-auth: proxying /v1 with a bearer token from AGENT_EVALS_API__API_KEY"
else
    # Clear any inherited header rather than leaving the file absent, which
    # would make nginx fail to start on the `include` directive.
    printf 'proxy_set_header Authorization "";\n' > "$SNIPPET"
    echo "api-auth: no AGENT_EVALS_API__API_KEY set; proxying /v1 unauthenticated" >&2
fi

chmod 600 "$SNIPPET"
