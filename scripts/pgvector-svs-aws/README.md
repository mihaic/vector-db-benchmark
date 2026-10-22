# pgvector HNSW vs SVS Vamana on AWS (DBpedia 1M × 1536)

How to reproduce the pgvector benchmark that compares HNSW (pgvector) with
SVS Vamana (the `svs` PostgreSQL extension) on two EC2 instances: a database
server and a separate benchmark client.

## What runs

Three engines from
[`pgvector-dbpedia-calibration.json`](../../experiments/configurations/pgvector-dbpedia-calibration.json),
all float32, inner product (`distance: dot`) on the normalized cosine dataset,
top-100, 100 client connections:

| Engine | Index |
|---|---|
| `dbpedia-cal-pgvector-hnsw-float32` | HNSW, m = 16, ef_construction = 200 |
| `dbpedia-cal-pgvector-svs-noquant` | Vamana, graph_degree = 32, build_window_size = 200, no compression |
| `dbpedia-cal-pgvector-svs-LeanVec4x8-div4` | as above, LeanVec 4×8 reduced to 384 dims |

Each engine runs in three modes, in this order:

1. `--search-duration 55`: search only, for 55 s.
2. `--search-duration 55 --update-search-ratio 100:100`: searches interleaved
   with re-upserts of existing vectors, for 55 s.
3. `--update-search-ratio 100:100 --insert`: searches interleaved with inserts
   of the 100,000 held-out vectors until all are in, then one pass scored
   against `allneighbors` (`post_insert_mean_recall`).

Every run starts from a fresh PostgreSQL cluster: load 875,000 vectors with
COPY, build the index after the load (build time counts in `upload_time`),
then calibrate the search knob (`ef` for HNSW, `svs.search_window_size` for
Vamana) to precision 0.95 before the measured search. All runs use
`--repetitions 1`. The 9 runs take several hours.

Dataset: `dbpedia-openai-1M-1536-angular-100neighbors_insert_100000_42_cosine`
(HDF5: 875,000 `train`, 100,000 `insert`, 5,000 `test` queries with 100
`neighbors` and `allneighbors` each; MD5
`99f9400d738e959a271906cf746e8423`).

## Hosts

| Role | Instance | vCPUs | RAM | Notes |
|---|---|---|---|---|
| Server | r8i.4xlarge (Xeon 6975P-C) | 16 | 123 GB | Ubuntu 26.04; 8 GB root disk, so the install and data live in `/dev/shm` (62 GB tmpfs) |
| Client | c8i.4xlarge | 16 | 30 GB | Ubuntu 24.04; holds the 6 GB dataset |

Both are in the same availability zone (us-west-2b), and the client reaches
the server only on its private IP. You need:

- passwordless SSH from the client to the server's private IP (the client
  script restarts the server before every run);
- the server's security group to allow TCP 5433 from the client.

Below, `SERVER_IP` and `CLIENT_IP` are the instances' private IPs.

## 1. Build PostgreSQL, pgvector and svs

The server has no compiler and a tiny root disk, so the install is built on
another Ubuntu 22.04 machine with the SVS extension's build guide and copied
over. The benchmark used:

- PostgreSQL REL_18_1
- pgvector v0.8.2
- SVS-Extension-for-PostgreSQL at commit `ccdc9e1` (svs 0.1.0)
- the SVS nightly shared library pinned in the build guide's `config`
  (`svs-shared-library-nightly-2026-09-15-1529`, SVS commit `1cd0df27`)

```bash
cd SVS-Extension-for-PostgreSQL/docs/build_guide
./build_all.sh        # PostgreSQL, SVS C API, pgvector, svs into ./pgsql_install and ./svs_install
```

Package the install. This bundles `libsvs_c_api` and ICU 70, which Ubuntu
26.04 doesn't ship:

```bash
vector-db-benchmark/scripts/pgvector-svs-aws/package_pg.sh \
    SVS-Extension-for-PostgreSQL/docs/build_guide pgsvs-bundle.tgz
```

Alternatively, build natively on the server (the build guide's requirements:
`build-essential cmake git pkg-config bison flex libreadline-dev zlib1g-dev
libicu-dev libgomp1`) with the same versions, into `/dev/shm/pgsvs`.

## 2. Set up the server

```bash
sudo apt install -y libgomp1

# Install (PostgreSQL installs are relocatable).
scp pgsvs-bundle.tgz ubuntu@<server>:/dev/shm/
ssh ubuntu@<server> 'cd /dev/shm && tar xzf pgsvs-bundle.tgz && rm pgsvs-bundle.tgz'

# Scripts and config, kept on the root disk.
ssh ubuntu@<server> mkdir -p pgsvs
scp vector-db-benchmark/scripts/pgvector-svs-aws/server.sh \
    vector-db-benchmark/tests/pgsvs-server.sh \
    vector-db-benchmark/experiments/pgsvs-server-bench-16vcpu.conf \
    ubuntu@<server>:pgsvs/
ssh ubuntu@<server> "printf 'SERVER_IP=%s\nCLIENT_IP=%s\n' $SERVER_IP $CLIENT_IP > pgsvs/server.env"
```

**Keep a login session open on the server for the whole benchmark**, for
example a `screen` session. systemd-logind defaults to `RemoveIPC=yes`: when
the last session of `ubuntu` ends, it deletes everything `ubuntu` owns in
`/dev/shm`, which here means the PostgreSQL install, the data directory, and
PostgreSQL's shared memory. `sudo loginctl enable-linger ubuntu` avoids this
too.

Check the server once by hand:

```bash
~/pgsvs/server.sh start
LD_LIBRARY_PATH=/dev/shm/pgsvs/pgsql_install/lib /dev/shm/pgsvs/pgsql_install/bin/psql \
    -h 127.0.0.1 -p 5433 -U postgres \
    -c "SELECT worker_state, search_threads_granted FROM pg_stat_vamana_worker"
# Expect: running | 13
~/pgsvs/server.sh stop
```

`server.sh` creates a new cluster in `/dev/shm/pgsvs/data` on every `start`
(trust auth on loopback and `CLIENT_IP/32` only). It enrolls the `postgres`
database in `vamana_databases` with 13 SVS search threads, and removes the
cluster on `stop`, keeping its log in `~/pgsvs/logs/`.

### Server settings for 16 vCPUs

[`pgsvs-server-bench-16vcpu.conf`](../../experiments/pgsvs-server-bench-16vcpu.conf)
is scaled from the 112-thread development host's
[`pgsvs-server-bench.conf`](../../experiments/pgsvs-server-bench.conf) and the
pgv-svs-dev-scripts 16-vCPU SVS config. The settings that matter:

| Setting | 16 vCPU | Dev host (2 × 28 cores) | Why |
|---|---|---|---|
| SVS search threads (`vamana_databases.search_num_threads`) | 13 | 48 | The default is 1 thread per database, which serializes all searches. 13 follows the pgv-svs-dev-scripts setting for this instance type. |
| `max_parallel_maintenance_workers` | 16 | 48 | Build threads, for both HNSW and Vamana. |
| `max_parallel_workers` | 32 | 112 | SVS keeps its search threads permanently parked in this pool, so it must hold them plus the build workers. |
| `max_worker_processes` | 48 | 128 | At least the pool plus the svs launcher and worker. |
| `maintenance_work_mem` | 16 GB | 32 GB | The HNSW graph fits in memory; SVS doesn't use it yet. |
| `shared_buffers` | 24 GB | 32 GB | The data directory is RAM too. |
| `svs.max_residency_memory`, `svs.default_residency_memory` | 32 GB | 96 GB | The defaults (100 MB) are far below a 1M × 1536 index. |
| `max_wal_size` | 16 GB | 64 GB | WAL is in tmpfs. |

To run with the pgv-svs-dev-scripts server settings instead
(`postgresql_svs_r8i.4xl.conf`), put
`PGSVS_CONF=$HOME/pgsvs/pgsvs-server-devscripts-16vcpu.conf` in `server.env`
and copy
[`pgsvs-server-devscripts-16vcpu.conf`](../../experiments/pgsvs-server-devscripts-16vcpu.conf)
to the server. It keeps only this file's svs memory budgets, which the
dev-scripts file predates. Expect slower builds: its `max_parallel_workers = 16`
leaves 3 workers beside the 13 search threads. It also logs no ERROR or
WARNING lines (`log_min_messages = log`).

Also required by svs: `shared_preload_libraries = 'vector, svs'` and
`wal_level = logical` (without logical WAL the svs launcher exits and every
Vamana `CREATE INDEX` fails waiting for a build-thread grant). `autovacuum` is
off so nothing competes with the measured phases.

Peak `/dev/shm` use is about 30 GB, during the HNSW build: table, WAL, and the
parallel build's 16 GB `maintenance_work_mem` segment.

## 3. Set up the client

Build the benchmark binary on Ubuntu 22.04 or newer (it needs
`libhdf5-dev` to build and `libhdf5-103` to run), from the commit being
benchmarked:

```bash
make vector-db-benchmark      # target/release/vector-db-benchmark
```

Lay out a benchmark directory on the client:

```bash
B=~/pgsvs-bench
ssh ubuntu@<client> "mkdir -p $B/datasets/dbpedia-openai-1M-1536-angular-100neighbors $B/experiments/configurations $B/results $B/logs"
scp target/release/vector-db-benchmark scripts/pgvector-svs-aws/run_benchmark.sh ubuntu@<client>:$B/
scp datasets/datasets.json ubuntu@<client>:$B/datasets/
scp experiments/configurations/pgvector-dbpedia-calibration.json ubuntu@<client>:$B/experiments/configurations/
```

Put the dataset at
`$B/datasets/dbpedia-openai-1M-1536-angular-100neighbors/dbpedia_openai_1M_insert_100000_42_cosine.hdf5`
(copy it, or symlink an existing copy) and check its MD5 (above).

## 4. Run

On the client, detached so it survives logout:

```bash
cd ~/pgsvs-bench
SERVER_IP=<server private IP> setsid nohup ./run_benchmark.sh > logs/driver.out 2>&1 < /dev/null &
```

Progress is in `logs/driver.log` (one START/END line per run, with the exit
code), and each run's output is in `logs/<mode>-<engine>.log`. To re-run some
engines only, set `ENGINES`, for example
`ENGINES=dbpedia-cal-pgvector-svs-noquant SERVER_IP=... ./run_benchmark.sh`. To check that a
build is parallel, run `ps -eo cmd | grep -c "[p]arallel worker for PID"` on the
server during an HNSW build (expect 16), or look for one
`CREATE INDEX` backend at about 1,600% CPU during a Vamana build.

## 5. Results

Result JSON files go to `~/pgsvs-bench/results/` on the client:

- `<engine>-<dataset>-upload-*.json`: `upload_time` (COPY plus index build).
- `<engine>-<dataset>-search-0-*.json`: search-only runs: `rps`, `mean_recall`,
  `p50_time`, `p95_time`, `p99_time`, and the calibrated knob in
  `params.search_params`.
- `<engine>-<dataset>-search-0-mixed-100x100-*.json`: mixed runs, which add
  `update_count`, `update_rps`, `update_p50_time`/`update_p95_time`/`update_p99_time`
  and `update_retries` (see Notes). The `--insert` run also has `post_insert_mean_recall` and
  `post_insert_recall_p10`.

To tabulate them, run the binary in report mode from the benchmark directory
(rows follow the `--engines` order):

```bash
cd ~/pgsvs-bench
./vector-db-benchmark --report report.md \
    --datasets dbpedia-openai-1M-1536-angular-100neighbors_insert_100000_42_cosine \
    --engines dbpedia-cal-pgvector-hnsw-float32 \
    --engines dbpedia-cal-pgvector-svs-noquant \
    --engines dbpedia-cal-pgvector-svs-LeanVec4x8-div4
```

## Notes

- **Vamana inserts can be rejected under load.** svs 0.1.0 accepts at most 64
  in-flight insert batches per database. Beyond that it rejects the write with
  "too many concurrently pending SVS insert batches". The engine retries those
  writes with backoff (for up to 30 s each) and reports the count as
  `update_retries`; any other write error fails the run.
- **A new Vamana index is briefly unusable.** Right after `CREATE INDEX`, the
  svs worker loads the index without updating its heartbeat. If that takes
  more than 3 s (as for the 1536-d float32 `svs-noquant` index), every scan and
  insert fails at once with "vamana background worker unavailable" until the
  load ends. After a Vamana build the engine waits, with a probe query, until
  the worker answers ("Vamana worker ready after ... s"); the wait counts in
  `upload_time`.
- **A failed calibration fails the run** rather than measuring with the
  default knob, and so does any failed query during calibration. The report
  marks older runs that fell back to the default with `†`, and lists the
  engines that have no result for a workload under its table.
- **A failed `SVSAddPoints` only shows on the server.** The client sees
  success, and the server log has `WARNING: SVS dynamic add points failed`.
  Check `~/pgsvs/logs/` after the insert runs.
- **Don't cap `max_parallel_workers` in a session** (for example with a
  benchmark tool's "max parallel workers" option). The parked SVS search threads
  already fill most of the pool, so a session cap at the build size leaves no
  workers and the build runs on one core.
- **A Vamana `CREATE INDEX` ignores cancel and terminate.** To abort one,
  restart the server with `pg_ctl -m immediate stop`.
