#!/usr/bin/env bash
# Start/stop a fresh svs-enabled PostgreSQL in /dev/shm for one benchmark run,
# serving the benchmark client over the private network. Runs on the server
# from a directory holding this script, tests/pgsvs-server.sh, the server conf
# and server.env (see README.md).
#
# Usage: server.sh start|stop
set -euo pipefail

dir=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
source "$dir/server.env"
: "${SERVER_IP:?set SERVER_IP (private IP of this host) in server.env}"
: "${CLIENT_IP:?set CLIENT_IP (private IP of the client) in server.env}"
PG_INSTALL=${PG_INSTALL:-/dev/shm/pgsvs/pgsql_install}

export PGSVS_BIN=$PG_INSTALL/bin
# The bundled ICU 70 (and anything else not in the OS) lives in lib/.
export LD_LIBRARY_PATH=$PG_INSTALL/lib
export PGSVS_DATA=${PGSVS_DATA:-/dev/shm/pgsvs/data}
export PGSVS_PORT=${PGSVS_PORT:-5433}
export PGSVS_LISTEN=$SERVER_IP
export PGSVS_TRUST_CIDR=$CLIENT_IP/32
export PGSVS_SEARCH_THREADS=${SEARCH_THREADS:-13}
export PGSVS_CONF=${PGSVS_CONF:-$dir/pgsvs-server-bench-16vcpu.conf}

mkdir -p "$dir/logs"
"$dir/pgsvs-server.sh" "$1"
# Keep each run's server log (the data dir is removed on stop).
if [ "$1" = stop ] && [ -f "$PGSVS_DATA.log" ]; then
    mv "$PGSVS_DATA.log" "$dir/logs/server-$(date +%Y%m%dT%H%M%S).log"
fi
