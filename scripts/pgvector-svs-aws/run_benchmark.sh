#!/usr/bin/env bash
# pgvector HNSW vs SVS Vamana on DBpedia: 3 modes x 3 engines, with a fresh
# server per run. Runs on the client from the benchmark directory (see
# README.md); start it detached so it survives logout:
#
#   SERVER_IP=<server private IP> setsid nohup ./run_benchmark.sh \
#       > logs/driver.out 2>&1 < /dev/null &
set -uo pipefail

: "${SERVER_IP:?set SERVER_IP to the private IP of the server}"
# Server directory with server.sh; relative paths are from the server home.
SERVER_DIR=${SERVER_DIR:-pgsvs}
DS=dbpedia-openai-1M-1536-angular-100neighbors_insert_100000_42_cosine
# ENGINES="a b" runs only those engines.
read -r -a ENGINES <<<"${ENGINES:-dbpedia-cal-pgvector-hnsw-float32 \
dbpedia-cal-pgvector-svs-noquant dbpedia-cal-pgvector-svs-LeanVec4x8-div4}"
MODES=(
    "duration|--search-duration 55"
    "mixed|--search-duration 55 --update-search-ratio 100:100"
    "insert|--update-search-ratio 100:100 --insert"
)

mkdir -p logs results
for mode in "${MODES[@]}"; do
    tag=${mode%%|*}
    flags=${mode#*|}
    for e in "${ENGINES[@]}"; do
        ssh "$SERVER_IP" "$SERVER_DIR/server.sh stop; $SERVER_DIR/server.sh start" \
            >>logs/server-control.log 2>&1
        sleep 5
        echo "$(date '+%F %T') START $tag $e" | tee -a logs/driver.log
        # shellcheck disable=SC2086
        PGVECTOR_PORT=5433 ./vector-db-benchmark --engines "$e" --datasets "$DS" \
            --host "$SERVER_IP" --skip-if-exists false --repetitions 1 $flags \
            >"logs/$tag-$e.log" 2>&1
        rc=$?
        echo "$(date '+%F %T') END $tag $e exit=$rc" | tee -a logs/driver.log
        ssh "$SERVER_IP" "$SERVER_DIR/server.sh stop" >>logs/server-control.log 2>&1
    done
done
echo "$(date '+%F %T') ALL DONE" | tee -a logs/driver.log
