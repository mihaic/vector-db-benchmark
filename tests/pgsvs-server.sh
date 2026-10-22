#!/usr/bin/env bash
# Start/stop a throwaway PostgreSQL with the svs extension for the pgvector SVS
# integration tests (the stock pgvector Docker image has no svs).
#
# Usage: PGSVS_BIN=<postgres bin dir> tests/pgsvs-server.sh start|stop
#
# Optional: PGSVS_DATA (data dir, recreated on start and removed on stop),
# PGSVS_PORT (default 5433), PGSVS_CONF (a file appended to postgresql.conf,
# e.g. experiments/pgsvs-server-bench.conf), PGSVS_SEARCH_THREADS (the
# database's vamana_databases.search_num_threads; the extension's default
# grants one SVS search thread), PGSVS_LISTEN (an address to listen on besides
# 127.0.0.1) and PGSVS_TRUST_CIDR (a client network allowed in with trust
# auth), to serve a benchmark client on another host.
#
# PGSVS_BIN is the bin directory of a PostgreSQL install that has both pgvector
# and svs installed, e.g. the one produced by the SVS extension's
# docs/build_guide/build_all.sh (<build_guide>/pgsql_install/bin).
set -euo pipefail

: "${PGSVS_BIN:?set PGSVS_BIN to the bin directory of a PostgreSQL install with pgvector + svs}"
PGSVS_DATA=${PGSVS_DATA:-${TMPDIR:-/tmp}/vdb-bench-pgsvs-data}
PGSVS_PORT=${PGSVS_PORT:-5433}
PGSVS_LOG=${PGSVS_DATA}.log

case "${1:-}" in
start)
    rm -rf "$PGSVS_DATA"
    # Trust auth on a throwaway cluster: loopback, plus PGSVS_TRUST_CIDR if set.
    "$PGSVS_BIN/initdb" -D "$PGSVS_DATA" -U postgres --auth=trust >/dev/null
    if [ -n "${PGSVS_TRUST_CIDR:-}" ]; then
        echo "host all all $PGSVS_TRUST_CIDR trust" >>"$PGSVS_DATA/pg_hba.conf"
    fi
    cat >>"$PGSVS_DATA/postgresql.conf" <<EOF
listen_addresses = '127.0.0.1${PGSVS_LISTEN:+,$PGSVS_LISTEN}'
port = $PGSVS_PORT
# The svs launcher and per-database workers start only when preloaded, and the
# launcher requires logical WAL.
shared_preload_libraries = 'svs'
wal_level = logical
max_connections = 300
EOF
    if [ -n "${PGSVS_CONF:-}" ]; then
        cat "$PGSVS_CONF" >>"$PGSVS_DATA/postgresql.conf"
    fi
    "$PGSVS_BIN/pg_ctl" -D "$PGSVS_DATA" -l "$PGSVS_LOG" -w start >/dev/null
    # svs creates vamana_databases; enrolling the database starts its worker.
    "$PGSVS_BIN/psql" -h 127.0.0.1 -p "$PGSVS_PORT" -U postgres -d postgres -q -v ON_ERROR_STOP=1 <<EOF
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS svs;
INSERT INTO vamana_databases (datname, search_num_threads)
    VALUES ('postgres', ${PGSVS_SEARCH_THREADS:-NULL}) ON CONFLICT (datname) DO NOTHING;
EOF
    echo "svs-enabled PostgreSQL running on ${PGSVS_LISTEN:-127.0.0.1}:$PGSVS_PORT (log: $PGSVS_LOG)"
    ;;
stop)
    "$PGSVS_BIN/pg_ctl" -D "$PGSVS_DATA" -m fast -w stop >/dev/null || true
    rm -rf "$PGSVS_DATA"
    ;;
*)
    echo "usage: $0 start|stop" >&2
    exit 2
    ;;
esac
