//! PgVector engine implementation.
//!
//! Uses the `postgres` crate for PostgreSQL connectivity with the pgvector extension.
//! Supports two vector indexes, selected by `algorithm` (top-level or in
//! `collection_params`):
//! - `hnsw` (default): pgvector HNSW with configurable m/ef_construction.
//! - `svs-vamana`: SVS Vamana from the `svs` extension, configured by the
//!   `collection_params.svs-vamana_config` block (passed through as index
//!   `WITH (...)` options) and searched with `svs.search_window_size`.
//!   Options use the extension's encoding, e.g. `alpha` as an integer x100 and
//!   `compression_type` 0 = none, 1 = LeanVec, 2 = LVQ. The server needs `svs`
//!   in `shared_preload_libraries`, `wal_level = logical`, and the database
//!   enrolled in `vamana_databases` (see tests/pgsvs-server.sh).
//!
//! COPY bulk upload, distance operators <-> (L2), <=> (cosine) and <#> (inner
//! product), and the mixed search+update / `--insert` workload (upserts by id).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use postgres::types::ToSql;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{closed_loop_duration, Engine, SearchResults, UpdateSearchRatio, UploadStats};
use vector_db_benchmark::readers::metadata::{
    is_multivalued_keyword_field, MetadataItem, MetadataValue,
};

/// Map a dataset schema field type to a Postgres column type. Returns None for
/// types pgvector can't filter on with a plain scalar column (e.g. geo).
fn pg_column_type(field_type: &str) -> Option<&'static str> {
    match field_type {
        // A `uuid` is an exact-match opaque string; store it in a TEXT column
        // (exact `=` match, like keyword). Without this arm it hit `_ => None`,
        // so no column was created and every uuid filter silently broke.
        "keyword" | "text" | "uuid" => Some("TEXT"),
        "int" => Some("BIGINT"),
        "float" => Some("DOUBLE PRECISION"),
        // Bools/datetimes arrive from the reader as strings ("true"/"false",
        // ISO-8601). COPY-text coerces those into a BOOLEAN column and parses
        // ISO into a TIMESTAMPTZ column, so a plain scalar column filters fine.
        "bool" => Some("BOOLEAN"),
        "datetime" => Some("TIMESTAMPTZ"),
        // Geo point stored as a "lat,lon" TEXT scalar; filtered with the
        // earthdistance extension (great-circle metres). A dedicated earth/cube
        // column would need binary COPY, so the text form keeps the COPY path.
        "geo" => Some("TEXT"),
        _ => None,
    }
}

/// Escape a string for a COPY ... FORMAT text field.
fn escape_copy_text(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Format the value of `column` from a row's metadata as a COPY text field,
/// returning `\N` (NULL) when the field is absent or unsupported.
fn copy_field(meta: Option<&MetadataItem>, column: &str) -> String {
    let value = meta.and_then(|m| m.fields.iter().find(|(k, _)| k == column).map(|(_, v)| v));
    match value {
        Some(MetadataValue::String(s)) => escape_copy_text(s),
        // Numeric columns receive the literal digits; COPY coerces them into the
        // declared BIGINT/DOUBLE column. No escaping needed (digits/./-/e only).
        Some(MetadataValue::Int(n)) => n.to_string(),
        Some(MetadataValue::Float(f)) => f.to_string(),
        Some(MetadataValue::Labels(labels)) => escape_copy_text(&labels.join(";")),
        // Geo point -> "lat,lon" TEXT (earthdistance filter splits it back out).
        Some(MetadataValue::Geo { lon, lat }) => escape_copy_text(&format!("{},{}", lat, lon)),
        // missing → NULL
        _ => "\\N".to_string(),
    }
}

/// `collection_params` key holding the SVS Vamana build options (same key as
/// the Redis SVS configs).
const SVS_CONFIG_KEY: &str = "svs-vamana_config";

/// `collection_params` keys for index build session settings (both indexes).
const BUILD_MAINTENANCE_WORK_MEM: &str = "maintenance_work_mem";
const BUILD_MAX_PARALLEL_WORKERS: &str = "max_parallel_workers";

/// Vector index built on `items.embedding`.
#[derive(Debug, Clone, PartialEq)]
enum PgIndex {
    /// pgvector HNSW. Created on the empty table in `configure` and filled
    /// incrementally by the COPY upload, unless `index_after_load` is set.
    Hnsw { m: i64, ef_construction: i64 },
    /// SVS Vamana, always built in one pass over the loaded table at the end
    /// of `upload` (graph construction over the full corpus).
    Vamana(VamanaConfig),
}

#[derive(Debug, Clone, Default, PartialEq)]
struct VamanaConfig {
    /// `WITH (...)` index options as `(name, SQL literal)`. Keys absent from the
    /// config are omitted so the extension's defaults apply.
    with_options: Vec<(String, String)>,
}

/// Session settings applied before an index build after the load.
#[derive(Debug, Clone, Default, PartialEq)]
struct BuildSettings {
    /// `maintenance_work_mem`, as a SQL literal. pgvector HNSW builds much
    /// faster when the graph fits; SVS does not use it yet.
    maintenance_work_mem: Option<String>,
    /// Parallel build workers (`max_parallel_maintenance_workers` and the
    /// table's `parallel_workers`). The pool itself, `max_parallel_workers`,
    /// is left to the server: SVS keeps its search threads parked in it.
    max_parallel_workers: Option<u64>,
}

/// Storage type of `items.embedding`, from `data_type` (`FLOAT32`, the
/// default, or `FLOAT16`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum VectorType {
    Vector,
    HalfVec,
}

impl VectorType {
    fn sql_type(self) -> &'static str {
        match self {
            VectorType::Vector => "vector",
            VectorType::HalfVec => "halfvec",
        }
    }

    /// Bound parameter `$n`, sent as a `vector` and cast to the column type.
    fn param(self, n: usize) -> String {
        match self {
            VectorType::Vector => format!("${}", n),
            VectorType::HalfVec => format!("${}::vector::halfvec", n),
        }
    }

    /// Ops class for this type from a `vector_*_ops` class.
    fn ops_class(self, vector_ops: &str) -> String {
        vector_ops.replacen("vector", self.sql_type(), 1)
    }
}

/// Re-upserts an existing point or inserts a new one during the mixed workload.
/// Only the embedding is written: re-upserted points keep their payload, and
/// `--insert` vectors carry none (their payload columns stay NULL).
fn upsert_sql(vector_type: VectorType) -> String {
    format!(
        "INSERT INTO items (id, embedding) VALUES ($1, {}) \
         ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding",
        vector_type.param(2)
    )
}

pub struct PgVectorEngine {
    name: String,
    host: String,
    port: u16,
    dbname: String,
    user: String,
    password: String,
    index: PgIndex,
    /// Build the index after the COPY load rather than before (always for
    /// Vamana; `collection_params.index_after_load` for HNSW).
    index_after_load: bool,
    build: BuildSettings,
    vector_type: VectorType,
    /// `collection_params.distance`: index with this metric instead of the
    /// dataset's. Only valid for cosine datasets, whose vectors and queries
    /// are normalized on read, so `l2` and `dot` rank identically to cosine.
    /// `normalize_cosine: true` is shorthand for `distance: l2`.
    distance: Option<String>,
    batch_size: usize,
    parallel: usize,
    search_params: Vec<SearchParams>,
    /// Payload columns (name, pg type) derived from the dataset schema, set in
    /// `configure` and written by `upload` so filtered queries have real columns.
    schema_columns: Vec<(String, String)>,
}

impl PgVectorEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("PGVECTOR_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5432);

        let dbname = std::env::var("PGVECTOR_DB").unwrap_or_else(|_| "postgres".to_string());
        let user = std::env::var("PGVECTOR_USER").unwrap_or_else(|_| "postgres".to_string());
        let password = std::env::var("PGVECTOR_PASSWORD").unwrap_or_else(|_| "passwd".to_string());

        let index = parse_pg_index(engine_config)?;
        let cp_extra = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.extra.as_ref());
        let cp_bool = |key: &str| {
            cp_extra
                .and_then(|e| e.get(key))
                .map(|v| {
                    v.as_bool()
                        .ok_or_else(|| format!("{} must be a boolean, got {}", key, v))
                })
                .transpose()
        };
        let index_after_load =
            matches!(index, PgIndex::Vamana(_)) || cp_bool("index_after_load")?.unwrap_or(false);
        let build = parse_build_settings(cp_extra)?;
        let vector_type = parse_vector_type(engine_config)?;

        let distance = cp_extra
            .and_then(|e| e.get("distance"))
            .map(|v| {
                v.as_str()
                    .map(str::to_lowercase)
                    .ok_or_else(|| format!("distance must be a string, got {}", v))
            })
            .transpose()?;
        if let Some(d) = &distance {
            map_pg_distance_ops(d)?;
        }
        let distance = match (cp_bool("normalize_cosine")?.unwrap_or(false), distance) {
            (false, d) => d,
            (true, None) => Some("l2".to_string()),
            (true, Some(_)) => {
                return Err("set either normalize_cosine or distance, not both".to_string())
            }
        };

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1024) as usize;

        Ok(Self {
            name: engine_config.name.clone(),
            host: host.to_string(),
            port,
            dbname,
            user,
            password,
            index,
            index_after_load,
            build,
            vector_type,
            distance,
            batch_size,
            parallel,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            schema_columns: Vec::new(),
        })
    }

    /// `(operator, ops class)` for the indexed distance. Resolved per call
    /// rather than cached in `configure`, which `--skip-upload` runs skip.
    fn distance_ops(&self, dataset: &Dataset) -> Result<(&'static str, String), String> {
        let dataset_ops = map_pg_distance_ops(dataset.distance())?;
        let (op, vector_ops) = match self.distance.as_deref() {
            None => dataset_ops,
            Some(d) => {
                let ops = map_pg_distance_ops(d)?;
                if ops != dataset_ops && !dataset.needs_normalization() {
                    return Err(format!(
                        "distance '{}' differs from the dataset's '{}', which ranks \
                         differently unless vectors are normalized (cosine datasets only)",
                        d,
                        dataset.distance()
                    ));
                }
                ops
            }
        };
        Ok((op, self.vector_type.ops_class(vector_ops)))
    }

    /// `CREATE INDEX` statement for the configured index.
    fn index_sql(&self, ops_class: &str) -> String {
        match &self.index {
            PgIndex::Hnsw { m, ef_construction } => format!(
                "CREATE INDEX ON items USING hnsw(embedding {}) WITH (m = {}, ef_construction = {})",
                ops_class, m, ef_construction
            ),
            PgIndex::Vamana(cfg) => vamana_index_sql(ops_class, cfg),
        }
    }

    fn connection_string(&self) -> String {
        format!(
            "host={} port={} dbname={} user={} password={}",
            self.host, self.port, self.dbname, self.user, self.password
        )
    }

    fn connect(&self) -> Result<postgres::Client, String> {
        postgres::Client::connect(&self.connection_string(), postgres::NoTls)
            .map_err(|e| format!("PostgreSQL connection failed: {}", e))
    }

    fn create_progress_bar(&self, total: usize) -> ProgressBar {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec_int}/s)")
                .unwrap()
                .with_key("per_sec_int", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    write!(w, "{}", HumanCount(state.per_sec() as u64)).unwrap()
                })
                .progress_chars("#>-"),
        );
        pb
    }

    /// Recall of one pass over every query against `allneighbors`, after a
    /// `--insert` run. Returns `(mean, p10, per-query recalls)`.
    #[allow(clippy::too_many_arguments)]
    fn post_insert_recall(
        &self,
        dataset: &Dataset,
        queries: &[Vec<f32>],
        parsed_filters: &[Option<(String, Vec<PgValue>)>],
        explicit_top: Option<usize>,
        knob: Option<i64>,
        distance_op: &str,
        parallel: usize,
    ) -> Result<(f64, f64, Vec<f64>), String> {
        let all_neighbors = dataset.read_all_neighbors()?;
        if all_neighbors.len() < queries.len() {
            return Err(format!(
                "allneighbors has {} rows, fewer than the {} queries",
                all_neighbors.len(),
                queries.len()
            ));
        }
        println!(
            "\tVerifying recall against the full post-insert corpus ({} queries)...",
            queries.len()
        );

        let query_idx = Arc::new(AtomicUsize::new(0));
        let conn_str = self.connection_string();
        let index = &self.index;
        let vector_type = self.vector_type;
        let pb = self.create_progress_bar(queries.len());
        let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel.max(1));
            for _ in 0..parallel.max(1) {
                let conn_str = conn_str.clone();
                let all_neighbors = &all_neighbors;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut local: Vec<f64> = Vec::new();
                    let mut conn = match postgres::Client::connect(&conn_str, postgres::NoTls) {
                        Ok(c) => c,
                        Err(_) => return local,
                    };
                    if let Err(e) = apply_search_session(&mut conn, index, knob) {
                        eprintln!("{}", e);
                        return local;
                    }
                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= queries.len() {
                            break;
                        }
                        let top = explicit_top.unwrap_or_else(|| {
                            let n = all_neighbors[idx].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });
                        let (results, _) = knn_query(
                            &mut conn,
                            distance_op,
                            vector_type,
                            &queries[idx],
                            top,
                            parsed_filters[idx].as_ref(),
                        );
                        match results {
                            Ok(ordered_ids) => {
                                let m = crate::metrics::compute_metrics(
                                    &ordered_ids,
                                    &all_neighbors[idx],
                                    top,
                                );
                                local.push(m.recall);
                            }
                            Err(e) => {
                                eprintln!("Post-insert recall query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    local
                }));
            }
            for h in handles {
                recalls.extend(h.join().unwrap());
            }
        });
        pb.finish_and_clear();

        if recalls.is_empty() {
            return Err("No post-insert recall queries completed".to_string());
        }
        let mean_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
        let mut sorted = recalls.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let recall_p10 = crate::engine::percentile_linear(&sorted, 0.10);
        Ok((mean_recall, recall_p10, recalls))
    }
}

/// Per-thread samples of the mixed workload, merged on join.
#[derive(Default)]
struct MixedWorkerSamples {
    times: Vec<f64>,
    precs: Vec<f64>,
    recs: Vec<f64>,
    mrs: Vec<f64>,
    nds: Vec<f64>,
    u_times: Vec<f64>,
    update_failures: usize,
    first_update_error: Option<String>,
    update_retries: usize,
}

impl Engine for PgVectorEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let vector_size = dataset.vector_size();
        let (_, ops_class) = self.distance_ops(dataset)?;

        let mut conn = self.connect()?;

        if matches!(self.index, PgIndex::Vamana(_)) {
            println!("Checking svs extension...");
            // A no-op when already installed; creating it needs a superuser.
            conn.execute("CREATE EXTENSION IF NOT EXISTS svs", &[])
                .map_err(|e| {
                    format!(
                        "svs extension unavailable ({}). Install it and run \
                         'CREATE EXTENSION svs;' as a PostgreSQL superuser",
                        pg_error_message(&e)
                    )
                })?;
        }

        // Ensure vector extension exists
        println!("Creating vector extension...");
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .map_err(|e| format!("Failed to create vector extension: {}", e))?;
        // cube + earthdistance back the geo-radius filter (great-circle metres).
        // Best-effort: only needed when a dataset has a geo field.
        let _ = conn.execute("CREATE EXTENSION IF NOT EXISTS cube", &[]);
        let _ = conn.execute("CREATE EXTENSION IF NOT EXISTS earthdistance", &[]);

        // Drop existing table
        println!("Dropping existing items table...");
        conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
            .map_err(|e| format!("Failed to drop table: {}", e))?;

        // Derive payload columns from the dataset schema so filtered queries have
        // real columns to reference (otherwise every filter is a SQL error).
        self.schema_columns = dataset
            .config
            .schema
            .as_ref()
            .and_then(|s| s.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(field, ftype)| {
                        pg_column_type(ftype.as_str().unwrap_or(""))
                            .map(|t| (field.clone(), t.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Create table (id, embedding, + one column per schema field)
        println!("Creating items table (vector dimension {})...", vector_size);
        let mut columns = vec![
            "id SERIAL PRIMARY KEY".to_string(),
            format!(
                "embedding {}({}) NOT NULL",
                self.vector_type.sql_type(),
                vector_size
            ),
        ];
        for (name, pg_type) in &self.schema_columns {
            columns.push(format!("\"{}\" {}", name, pg_type));
        }
        let create_sql = format!("CREATE TABLE items ({})", columns.join(", "));
        conn.execute(&create_sql, &[])
            .map_err(|e| format!("Failed to create table: {}", e))?;

        // Set storage to PLAIN for better performance
        conn.execute(
            "ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN",
            &[],
        )
        .map_err(|e| format!("Failed to alter storage: {}", e))?;

        if self.index_after_load {
            println!("Vector index will be built after upload.");
        } else {
            let index_sql = self.index_sql(&ops_class);
            println!("Creating vector index: {}", index_sql);
            conn.execute(&index_sql, &[])
                .map_err(|e| format!("Failed to create vector index: {}", pg_error_message(&e)))?;
        }

        println!("PgVector configured successfully.");

        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();
        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors, metadata) = dataset.read_vectors(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();

        println!(
            "Read {} vectors ({}d) in {:.3}s",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
        );

        // PgVector uses COPY for bulk upload
        // Use batched INSERT for parallel upload since COPY requires exclusive connection
        let pb = self.create_progress_bar(ids.len());
        let upload_start = Instant::now();

        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let conn_str = self.connection_string();
        let schema_columns = self.schema_columns.clone();

        // COPY target column list: id, embedding, then one per schema field.
        let copy_sql = {
            let mut cols = vec!["id".to_string(), "embedding".to_string()];
            for (name, _) in &schema_columns {
                cols.push(format!("\"{}\"", name));
            }
            format!(
                "COPY items ({}) FROM STDIN WITH (FORMAT text)",
                cols.join(", ")
            )
        };

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let conn_str = conn_str.clone();
                let copy_sql = copy_sql.clone();
                let schema_columns = &schema_columns;
                let batches = &batches;
                let ids = &ids;
                let vectors = &vectors;
                let metadata = &metadata;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match postgres::Client::connect(&conn_str, postgres::NoTls) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };

                    loop {
                        let idx = batch_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= total_batches {
                            break;
                        }
                        if error.lock().unwrap().is_some() {
                            break;
                        }

                        let (batch_start, batch_end) = batches[idx];

                        // Use COPY for bulk insert
                        let copy_result = (|| -> Result<(), String> {
                            let mut writer = conn
                                .copy_in(copy_sql.as_str())
                                .map_err(|e| format!("COPY start failed: {}", e))?;

                            use std::io::Write;
                            for i in batch_start..batch_end {
                                let vec_str: String = vectors[i]
                                    .iter()
                                    .map(|v| v.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",");
                                let mut line = format!("{}\t[{}]", ids[i], vec_str);
                                for (name, _) in schema_columns {
                                    line.push('\t');
                                    line.push_str(&copy_field(metadata[i].as_ref(), name));
                                }
                                writeln!(writer, "{}", line)
                                    .map_err(|e| format!("COPY write failed: {}", e))?;
                            }

                            writer
                                .finish()
                                .map_err(|e| format!("COPY finish failed: {}", e))?;
                            Ok(())
                        })();

                        if let Err(e) = copy_result {
                            *error.lock().unwrap() = Some(e);
                            break;
                        }

                        pb.inc((batch_end - batch_start) as u64);
                    }
                });
            }
        });

        pb.finish_with_message("Upload complete");

        if let Some(e) = error.lock().unwrap().take() {
            return Err(e);
        }

        // A build after the load is part of the upload time, as an incremental
        // build is part of the COPY time.
        if self.index_after_load {
            let copy_time = upload_start.elapsed().as_secs_f64();
            let (_, ops_class) = self.distance_ops(dataset)?;
            let mut conn = self.connect()?;
            for stmt in build_session_sql(&self.build) {
                conn.execute(&stmt, &[])
                    .map_err(|e| format!("Failed to apply '{}': {}", stmt, pg_error_message(&e)))?;
            }
            let index_sql = self.index_sql(&ops_class);
            println!("Building vector index: {}", index_sql);
            let build_start = Instant::now();
            conn.execute(&index_sql, &[])
                .map_err(|e| format!("Failed to create vector index: {}", pg_error_message(&e)))?;
            println!(
                "COPY time: {:.3}s, index build time: {:.3}s",
                copy_time,
                build_start.elapsed().as_secs_f64()
            );
        }

        let upload_time = upload_start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        let total_time = read_time + upload_time;

        Ok(UploadStats {
            upload_time,
            total_time,
            drain_time: None,
            upload_count: vectors.len(),
            parallel: self.parallel,
            batch_size: self.batch_size,
            memory_usage: None,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;
        let knob = search_knob(&self.index, params);
        let (distance_op, _) = self.distance_ops(dataset)?;
        // Fail the run up front if the session settings are rejected.
        apply_search_session(&mut self.connect()?, &self.index, knob)?;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;
        if queries.is_empty() {
            return Err("dataset contains no search queries".to_string());
        }
        // Duration mode recycles queries, indexing `neighbors` in lock-step.
        if neighbors.len() < queries.len() {
            return Err("dataset ground-truth shorter than query set".into());
        }

        // Parse each query's filter ONCE (outside the timed region) into a
        // parameterized template + a list of typed values to bind. The template
        // SQL text is stable per filter shape (values are `$N` placeholders, not
        // inlined), so the server reuses one prepared plan across queries in a
        // run. Filter placeholders start at $3 ($1 = query vector, $2 = LIMIT).
        let parsed_filters: Vec<Option<(String, Vec<PgValue>)>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(|v| parse_pg_conditions(v, 3)))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        // `--search-duration`: run until the deadline, recycling queries.
        let closed_loop_duration = closed_loop_duration(params)?;
        let num_to_run = if closed_loop_duration.is_some() {
            usize::MAX
        } else if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(if closed_loop_duration.is_some() {
            0
        } else {
            num_to_run
        });
        let conn_str = self.connection_string();
        let index = &self.index;
        let vector_type = self.vector_type;

        // Barrier-synchronized start so per-worker connection setup AND the cold
        // first query fall OUTSIDE the measured window (mirrors redis/vertex).
        // Every worker connects + primes, then blocks on `ready`; the main thread
        // stamps the shared start instant into `start_cell` and releases `go`, so
        // the measurement clock starts only once all workers are warm and poised.
        // A worker that fails to connect MUST still pass both barriers before
        // returning, or the run would deadlock.
        let ready = Arc::new(std::sync::Barrier::new(parallel + 1));
        let go = Arc::new(std::sync::Barrier::new(parallel + 1));
        let start_cell = Arc::new(std::sync::OnceLock::<Instant>::new());

        let capacity = num_to_run.min(queries.len());
        let mut times: Vec<f64> = Vec::with_capacity(capacity);
        let mut precs: Vec<f64> = Vec::with_capacity(capacity);
        let mut recs: Vec<f64> = Vec::with_capacity(capacity);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(capacity);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(capacity);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let conn_str = conn_str.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let query_idx = Arc::clone(&query_idx);
                let ready = Arc::clone(&ready);
                let go = Arc::clone(&go);
                let start_cell = Arc::clone(&start_cell);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let mut conn = match postgres::Client::connect(&conn_str, postgres::NoTls) {
                        Ok(c) => c,
                        Err(_) => {
                            // Still cross both barriers so peers aren't stranded.
                            ready.wait();
                            go.wait();
                            return (t, p, r, mr, nd);
                        }
                    };

                    if let Err(e) = apply_search_session(&mut conn, index, knob) {
                        eprintln!("{}", e);
                        ready.wait();
                        go.wait();
                        return (t, p, r, mr, nd);
                    }

                    // Prime this connection with ONE discarded query (index 0) so
                    // the cold first round-trip + plan build is not inside the
                    // measured window. Best effort: errors ignored, no sample
                    // recorded. Reuses the exact per-query request path below.
                    if !queries.is_empty() {
                        let prime_top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[0].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });
                        let _ = knn_query(
                            &mut conn,
                            distance_op,
                            vector_type,
                            &queries[0],
                            prime_top,
                            parsed_filters[0].as_ref(),
                        );
                    }

                    // Signal "connected + primed", then block until the main thread
                    // stamps the shared measurement start and releases everyone.
                    ready.wait();
                    go.wait();
                    let start_time = *start_cell.get().unwrap();

                    loop {
                        if closed_loop_duration
                            .map(|duration| Instant::now() >= start_time + duration)
                            .unwrap_or(false)
                        {
                            break;
                        }
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }
                        let idx = idx % queries.len();

                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        let (results, query_time) = knn_query(
                            &mut conn,
                            distance_op,
                            vector_type,
                            &queries[idx],
                            top,
                            parsed_filters[idx].as_ref(),
                        );

                        match results {
                            Ok(ordered_ids) => {
                                let m = crate::metrics::compute_metrics(
                                    &ordered_ids,
                                    &neighbors[idx],
                                    top,
                                );
                                t.push(query_time);
                                p.push(m.precision);
                                r.push(m.recall);
                                mr.push(m.mrr);
                                nd.push(m.ndcg);
                            }
                            Err(e) => {
                                eprintln!("Search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                }));
            }

            // All workers spawned. Wait until every one is connected + primed,
            // stamp the shared start instant, then release them together so the
            // measurement clock excludes connection setup and the cold first query.
            ready.wait();
            let st = Instant::now();
            start_cell.set(st).ok();
            go.wait();

            for h in handles {
                let (t, p, r, mr, nd) = h.join().unwrap();
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
            }
        });

        pb.finish_and_clear();
        // total_time excludes connection setup and the cold first query: it is
        // measured from the barrier release stamped into `start_cell`.
        let total_time = start_cell
            .get()
            .map(|st| st.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        // Dispatched queries; equals `num_to_run` unless duration-bounded.
        let attempted_queries = query_idx.load(Ordering::Relaxed).min(num_to_run);
        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times,
            &precs,
            &recs,
            &mrr_vals,
            &ndcg_vals,
            total_time,
            top,
            parallel,
            attempted_queries,
        )
    }

    fn search_mixed(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
        ratio: &UpdateSearchRatio,
        use_insert: bool,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;
        let knob = search_knob(&self.index, params);
        let (distance_op, _) = self.distance_ops(dataset)?;
        apply_search_session(&mut self.connect()?, &self.index, knob)?;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;
        if queries.is_empty() {
            return Err("dataset contains no search queries".to_string());
        }
        // Workers index `neighbors[query_pos]` in lock-step with `queries`.
        if neighbors.len() < queries.len() {
            return Err("dataset ground-truth shorter than query set".into());
        }
        let parsed_filters: Vec<Option<(String, Vec<PgValue>)>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(|v| parse_pg_conditions(v, 3)))
            .collect();

        // Updates come from the dedicated `insert` set (new points) with
        // --insert, otherwise from the corpus (re-upserting existing points).
        let normalize = dataset.needs_normalization();
        let (upd_ids, upd_vectors, _) = if use_insert {
            println!("\tReading insert vectors for updates...");
            dataset.read_insert_vectors(normalize)?
        } else {
            println!("\tReading vectors for updates...");
            dataset.read_vectors(normalize)?
        };
        if upd_ids.is_empty() {
            return Err("dataset contains no vectors to update".to_string());
        }
        let mut update_seq: Vec<usize> = (0..upd_ids.len()).collect();
        update_seq.shuffle(&mut rand::rngs::StdRng::seed_from_u64(42));

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let closed_loop_duration = closed_loop_duration(params)?;
        // With --insert the run stops once every insert vector has been used
        // exactly once; queries recycle until then, as in duration mode.
        let unbounded = closed_loop_duration.is_some() || use_insert;
        let num_to_run = if unbounded {
            usize::MAX
        } else if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        let search_idx = Arc::new(AtomicUsize::new(0));
        let update_idx = Arc::new(AtomicUsize::new(0));
        let ratio_searches = ratio.searches as usize;
        let ratio_updates = ratio.updates as usize;
        let update_seq_len = update_seq.len();
        let conn_str = self.connection_string();
        let index = &self.index;
        let vector_type = self.vector_type;
        let upsert_sql = upsert_sql(self.vector_type);
        let upsert_sql = upsert_sql.as_str();

        let pb = self.create_progress_bar(if unbounded { 0 } else { num_to_run });
        let start_time = Instant::now();

        let mut times: Vec<f64> = Vec::new();
        let mut precs: Vec<f64> = Vec::new();
        let mut recs: Vec<f64> = Vec::new();
        let mut mrs: Vec<f64> = Vec::new();
        let mut nds: Vec<f64> = Vec::new();
        let mut u_times: Vec<f64> = Vec::new();
        let mut update_failures = 0usize;
        let mut first_update_error: Option<String> = None;
        let mut update_retries = 0usize;

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let conn_str = conn_str.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let upd_ids = &upd_ids;
                let upd_vectors = &upd_vectors;
                let update_seq = &update_seq;
                let search_idx = Arc::clone(&search_idx);
                let update_idx = Arc::clone(&update_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut out = MixedWorkerSamples::default();

                    let mut conn = match postgres::Client::connect(&conn_str, postgres::NoTls) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("PostgreSQL connection failed: {}", e);
                            return out;
                        }
                    };
                    if let Err(e) = apply_search_session(&mut conn, index, knob) {
                        eprintln!("{}", e);
                        return out;
                    }
                    let upsert = match conn.prepare(upsert_sql) {
                        Ok(stmt) => stmt,
                        Err(e) => {
                            eprintln!("Failed to prepare upsert: {}", pg_error_message(&e));
                            return out;
                        }
                    };
                    let mut pb_pending: u64 = 0;

                    'outer: loop {
                        // Checked once per round so a round is never cut
                        // half-searched/half-updated.
                        if closed_loop_duration
                            .map(|duration| Instant::now() >= start_time + duration)
                            .unwrap_or(false)
                        {
                            break 'outer;
                        }

                        for _ in 0..ratio_searches {
                            let idx = search_idx.fetch_add(1, Ordering::Relaxed);
                            if idx >= num_to_run {
                                break 'outer;
                            }
                            let query_pos = idx % queries.len();
                            let top = explicit_top.unwrap_or_else(|| {
                                let n = neighbors[query_pos].len();
                                if n > 0 {
                                    n
                                } else {
                                    10
                                }
                            });

                            let (results, query_time) = knn_query(
                                &mut conn,
                                distance_op,
                                vector_type,
                                &queries[query_pos],
                                top,
                                parsed_filters[query_pos].as_ref(),
                            );
                            match results {
                                Ok(ordered_ids) => {
                                    let m = crate::metrics::compute_metrics(
                                        &ordered_ids,
                                        &neighbors[query_pos],
                                        top,
                                    );
                                    out.times.push(query_time);
                                    out.precs.push(m.precision);
                                    out.recs.push(m.recall);
                                    out.mrs.push(m.mrr);
                                    out.nds.push(m.ndcg);
                                }
                                Err(e) => {
                                    eprintln!("Search query {} failed: {}", idx, e);
                                }
                            }
                            pb_pending += 1;
                            if pb_pending >= 256 {
                                pb.inc(pb_pending);
                                pb_pending = 0;
                            }
                        }

                        // With --insert each insert vector is used exactly once,
                        // so the run stops when the insert set is exhausted.
                        for _ in 0..ratio_updates {
                            let uidx = update_idx.fetch_add(1, Ordering::Relaxed);
                            if use_insert && uidx >= update_seq_len {
                                break 'outer;
                            }
                            let data_idx = update_seq[uidx % update_seq_len];

                            let update_start = Instant::now();
                            let result = upsert_vector(
                                &mut conn,
                                &upsert,
                                upd_ids[data_idx],
                                &upd_vectors[data_idx],
                            );
                            let update_time = update_start.elapsed().as_secs_f64();
                            match result {
                                Ok(retries) => {
                                    out.u_times.push(update_time);
                                    out.update_retries += retries;
                                }
                                Err(e) => {
                                    out.update_failures += 1;
                                    out.first_update_error.get_or_insert(e);
                                }
                            }
                        }
                    }
                    if pb_pending > 0 {
                        pb.inc(pb_pending);
                    }
                    out
                }));
            }

            for h in handles {
                let out = h.join().unwrap();
                times.extend(out.times);
                precs.extend(out.precs);
                recs.extend(out.recs);
                mrs.extend(out.mrs);
                nds.extend(out.nds);
                u_times.extend(out.u_times);
                update_failures += out.update_failures;
                update_retries += out.update_retries;
                if first_update_error.is_none() {
                    first_update_error = out.first_update_error;
                }
            }
        });

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        // A failed write means the index (e.g. an SVS build without write
        // support) did not take the workload; don't report it as measured.
        if update_failures > 0 {
            return Err(format!(
                "{} of {} updates failed; first error: {}",
                update_failures,
                update_failures + u_times.len(),
                first_update_error.unwrap_or_default()
            ));
        }
        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        let attempted_queries = search_idx.load(Ordering::Relaxed).min(num_to_run);
        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        let mut results = crate::engine::compute_search_stats(
            &times,
            &precs,
            &recs,
            &mrs,
            &nds,
            total_time,
            top,
            parallel,
            attempted_queries,
        )?;
        if !u_times.is_empty() {
            let mut u_sorted = u_times.clone();
            u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            results.update_count = Some(u_times.len());
            results.update_rps = Some(u_times.len() as f64 / total_time);
            results.update_mean_time = Some(u_times.iter().sum::<f64>() / u_times.len() as f64);
            results.update_p50_time = Some(crate::engine::percentile_linear(&u_sorted, 0.50));
            results.update_p95_time = Some(crate::engine::percentile_linear(&u_sorted, 0.95));
            results.update_p99_time = Some(crate::engine::percentile_linear(&u_sorted, 0.99));
        }
        results.update_latencies = Some(u_times);
        results.update_search_ratio = Some(format!("{}:{}", ratio.updates, ratio.searches));
        results.update_retries = Some(update_retries);
        if update_retries > 0 {
            println!(
                "\tUpdates retried {} times (svs pending-insert slots exhausted)",
                update_retries
            );
        }

        // `neighbors` is ground truth against `train` alone; once every insert
        // vector is in, score one fresh pass against `allneighbors` (the full
        // merged corpus). Skipped when the dataset has no `allneighbors`.
        if use_insert {
            match self.post_insert_recall(
                dataset,
                &queries,
                &parsed_filters,
                explicit_top,
                knob,
                distance_op,
                parallel,
            ) {
                Ok((mean_recall, recall_p10, recalls)) => {
                    println!(
                        "\tPost-insert recall (allneighbors): mean={:.4}, p10={:.4}",
                        mean_recall, recall_p10
                    );
                    results.post_insert_mean_recall = Some(mean_recall);
                    results.post_insert_recall_p10 = Some(recall_p10);
                    results.post_insert_recalls = Some(recalls);
                }
                Err(e) => {
                    println!("\tSkipping post-insert recall check: {}", e);
                }
            }
        }

        Ok(results)
    }

    fn delete(&mut self) -> Result<(), String> {
        let mut conn = self.connect()?;
        conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
            .map_err(|e| format!("Failed to drop table: {}", e))?;
        Ok(())
    }
}

// ── Index config ──────────────────────────────────────────────────

/// Resolve the vector index from `algorithm` (top-level, then
/// `collection_params.algorithm`; default `hnsw`).
fn parse_pg_index(engine_config: &EngineConfig) -> Result<PgIndex, String> {
    let cp = engine_config.collection_params.as_ref();
    let cp_extra = cp.and_then(|cp| cp.extra.as_ref());
    let algorithm = engine_config
        .algorithm
        .clone()
        .or_else(|| {
            cp_extra
                .and_then(|e| e.get("algorithm"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "hnsw".to_string());

    match algorithm.to_lowercase().as_str() {
        "hnsw" => {
            let (m, ef_construction) = cp
                .and_then(|cp| cp.hnsw_config.as_ref())
                .map(|h| (h.m.unwrap_or(16), h.ef_construction.unwrap_or(128)))
                .unwrap_or((16, 128));
            Ok(PgIndex::Hnsw { m, ef_construction })
        }
        "svs-vamana" | "vamana" => {
            let empty = serde_json::Map::new();
            let block = match cp_extra.and_then(|e| e.get(SVS_CONFIG_KEY)) {
                None => &empty,
                Some(v) => v
                    .as_object()
                    .ok_or_else(|| format!("{} must be an object, got {}", SVS_CONFIG_KEY, v))?,
            };
            Ok(PgIndex::Vamana(parse_vamana_config(block)?))
        }
        other => Err(format!(
            "pgvector does not support algorithm '{}' (expected 'hnsw' or 'svs-vamana')",
            other
        )),
    }
}

/// Parse the `svs-vamana_config` block. Every key other than the build session
/// settings is passed through as a `WITH (...)` index option, so the
/// extension's own option names apply (e.g. `graph_degree`,
/// `build_window_size`, `compression_type`) and it rejects unknown ones.
fn parse_vamana_config(
    block: &serde_json::Map<String, serde_json::Value>,
) -> Result<VamanaConfig, String> {
    let mut cfg = VamanaConfig::default();
    for (key, value) in block {
        if key == BUILD_MAINTENANCE_WORK_MEM || key == BUILD_MAX_PARALLEL_WORKERS {
            return Err(format!(
                "{} is a build setting: set it in collection_params, not {}",
                key, SVS_CONFIG_KEY
            ));
        }
        // Option names are spliced into the DDL, so allow only plain lowercase
        // identifiers (which also flags Redis-style keys like GRAPH_MAX_DEGREE
        // copied over by mistake).
        let valid = key.starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
            && key
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !valid {
            return Err(format!(
                "{} option '{}' is not a lowercase SVS index option name",
                SVS_CONFIG_KEY, key
            ));
        }
        let name = format!("{}.{}", SVS_CONFIG_KEY, key);
        cfg.with_options
            .push((key.clone(), sql_literal(&name, value)?));
    }
    Ok(cfg)
}

/// Parse the build settings from the `collection_params` extras.
fn parse_build_settings(
    cp_extra: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Result<BuildSettings, String> {
    let get = |key: &str| cp_extra.and_then(|e| e.get(key));
    let maintenance_work_mem = get(BUILD_MAINTENANCE_WORK_MEM)
        .map(|value| match value {
            // A string with a unit ('64GB') or a number of kB, as in Postgres.
            serde_json::Value::String(_) | serde_json::Value::Number(_) => {
                sql_literal(BUILD_MAINTENANCE_WORK_MEM, value)
            }
            _ => Err(format!(
                "{} must be a string or number, got {}",
                BUILD_MAINTENANCE_WORK_MEM, value
            )),
        })
        .transpose()?;
    let max_parallel_workers = get(BUILD_MAX_PARALLEL_WORKERS)
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                format!(
                    "{} must be a non-negative integer, got {}",
                    BUILD_MAX_PARALLEL_WORKERS, value
                )
            })
        })
        .transpose()?;
    Ok(BuildSettings {
        maintenance_work_mem,
        max_parallel_workers,
    })
}

/// `data_type` from `collection_params`, else `upload_params` (as in the Redis
/// configs): `FLOAT32` (default) or `FLOAT16`.
fn parse_vector_type(engine_config: &EngineConfig) -> Result<VectorType, String> {
    let data_type = engine_config
        .collection_params
        .as_ref()
        .and_then(|cp| cp.extra.as_ref())
        .and_then(|e| e.get("data_type"))
        .or_else(|| {
            engine_config
                .upload_params
                .as_ref()
                .and_then(|p| p.get("data_type"))
        });
    match data_type {
        None => Ok(VectorType::Vector),
        Some(v) => match v.as_str().map(str::to_uppercase).as_deref() {
            Some("FLOAT32") => Ok(VectorType::Vector),
            Some("FLOAT16") => Ok(VectorType::HalfVec),
            _ => Err(format!(
                "pgvector data_type must be FLOAT32 or FLOAT16, got {}",
                v
            )),
        },
    }
}

/// Render a JSON scalar as a SQL literal for DDL / SET.
fn sql_literal(name: &str, value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::Bool(b) => Ok(b.to_string()),
        serde_json::Value::String(s) => Ok(format!("'{}'", s.replace('\'', "''"))),
        _ => Err(format!(
            "{} must be a number, boolean or string, got {}",
            name, value
        )),
    }
}

fn vamana_index_sql(ops_class: &str, cfg: &VamanaConfig) -> String {
    let mut sql = format!(
        "CREATE INDEX ON items USING vamana(embedding {})",
        ops_class
    );
    if !cfg.with_options.is_empty() {
        let options: Vec<String> = cfg
            .with_options
            .iter()
            .map(|(k, v)| format!("{} = {}", k, v))
            .collect();
        sql.push_str(&format!(" WITH ({})", options.join(", ")));
    }
    sql
}

/// Session settings applied on the build connection before `CREATE INDEX`.
fn build_session_sql(build: &BuildSettings) -> Vec<String> {
    let mut stmts = Vec::new();
    if let Some(mem) = &build.maintenance_work_mem {
        stmts.push(format!("SET maintenance_work_mem = {}", mem));
    }
    if let Some(n) = build.max_parallel_workers {
        // Not `SET max_parallel_workers`: the pool also holds the SVS search
        // threads (parked parallel workers), so capping it at the build's
        // size leaves no workers for the build, which then runs serially.
        stmts.push(format!("SET max_parallel_maintenance_workers = {}", n));
        stmts.push(format!("ALTER TABLE items SET (parallel_workers = {})", n));
    }
    stmts
}

// ── Search session + queries ──────────────────────────────────────

/// The index's search-time knob: HNSW `ef_search` (`ef`, else `hnsw_ef`;
/// default 128) or Vamana `search_window_size` (`search_window_size`, else
/// `ef`; `None` keeps the extension default). Calibration sets either key.
fn search_knob(index: &PgIndex, params: &SearchParams) -> Option<i64> {
    let sp = params.search_params.as_ref();
    let extra_i64 = |key: &str| {
        sp.and_then(|sp| sp.extra.as_ref())
            .and_then(|e| e.get(key))
            .and_then(|v| v.as_i64())
    };
    match index {
        PgIndex::Hnsw { .. } => Some(
            sp.and_then(|sp| sp.ef)
                .or_else(|| extra_i64("hnsw_ef"))
                .unwrap_or(128),
        ),
        PgIndex::Vamana(_) => extra_i64("search_window_size")
            .or_else(|| extra_i64("SEARCH_WINDOW_SIZE"))
            .or_else(|| sp.and_then(|sp| sp.ef)),
    }
}

/// Apply the index's search settings to a connection.
fn apply_search_session(
    conn: &mut postgres::Client,
    index: &PgIndex,
    knob: Option<i64>,
) -> Result<(), String> {
    match index {
        PgIndex::Hnsw { .. } => {
            if let Some(ef) = knob {
                let _ = conn.execute(&format!("SET hnsw.ef_search = {}", ef), &[]);
            }
            // Enable iterative index scans (pgvector >= 0.8) so that
            // non-sargable filters (e.g. array contains-any) keep pulling
            // from the HNSW index until LIMIT is satisfied in exact order,
            // instead of post-filtering a bounded candidate set.
            let _ = conn.execute("SET hnsw.iterative_scan = strict_order", &[]);
        }
        PgIndex::Vamana(_) => {
            // Not best effort: an ignored window would silently search with
            // the default and make calibration meaningless.
            if let Some(window) = knob {
                conn.execute(&format!("SET svs.search_window_size = {}", window), &[])
                    .map_err(|e| {
                        format!(
                            "Failed to set svs.search_window_size: {}",
                            pg_error_message(&e)
                        )
                    })?;
            }
        }
    }
    Ok(())
}

/// Run one KNN query, returning the ordered ids and the query round-trip time.
/// `top` (LIMIT) and every filter value are BOUND params, not inlined, so the
/// statement text is stable per filter shape and the plan is reused across
/// queries.
fn knn_query(
    conn: &mut postgres::Client,
    distance_op: &str,
    vector_type: VectorType,
    query: &[f32],
    top: usize,
    filter: Option<&(String, Vec<PgValue>)>,
) -> (Result<Vec<i64>, postgres::Error>, f64) {
    let query_vec = pgvector::Vector::from(query.to_vec());
    let top_param = top as i64;
    let where_clause = filter
        .map(|(tmpl, _)| format!(" WHERE {}", tmpl))
        .unwrap_or_default();

    // $1 = query vector, $2 = LIMIT (bigint), $3.. = filter values.
    let query_sql = format!(
        "SELECT id, embedding {} {} AS _score FROM items{} ORDER BY _score LIMIT $2::bigint",
        distance_op,
        vector_type.param(1),
        where_clause
    );

    let mut query_params: Vec<&(dyn ToSql + Sync)> =
        Vec::with_capacity(2 + filter.map(|(_, v)| v.len()).unwrap_or(0));
    query_params.push(&query_vec);
    query_params.push(&top_param);
    if let Some((_, values)) = filter {
        for v in values {
            query_params.push(v.as_sql());
        }
    }

    let query_start = Instant::now();
    let results = conn.query(&query_sql, &query_params);
    let query_time = query_start.elapsed().as_secs_f64();

    let ids = results.map(|rows| {
        rows.iter()
            .map(|row| {
                let id: i32 = row.get(0);
                id as i64
            })
            .collect()
    });
    (ids, query_time)
}

/// Give up retrying a rejected write after this long.
const UPSERT_RETRY_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// Upsert one vector, returning how many times it was retried. Writes that svs
/// rejects for lack of a pending-insert slot are retried with backoff, as an
/// application would; any other error fails the write.
fn upsert_vector(
    conn: &mut postgres::Client,
    upsert: &postgres::Statement,
    id: i64,
    vector: &[f32],
) -> Result<usize, String> {
    // `items.id` is SERIAL (int4).
    let id = i32::try_from(id).map_err(|_| format!("id {} does not fit items.id (int4)", id))?;
    let embedding = pgvector::Vector::from(vector.to_vec());
    let start = Instant::now();
    let mut backoff = std::time::Duration::from_micros(100);
    let mut retries = 0;
    loop {
        match conn.execute(upsert, &[&id, &embedding]) {
            Ok(_) => return Ok(retries),
            Err(e)
                if is_svs_insert_slot_exhausted(&e) && start.elapsed() < UPSERT_RETRY_DEADLINE =>
            {
                retries += 1;
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(std::time::Duration::from_millis(10));
            }
            Err(e) => {
                return Err(format!(
                    "upsert of id {} failed after {} retries: {}",
                    id,
                    retries,
                    pg_error_message(&e)
                ))
            }
        }
    }
}

/// svs rejects an index insert with `program_limit_exceeded` when every one of
/// its per-database pending-insert reservation slots is in use
/// (`SVS_MAX_PENDING_INSERT_RESERVATIONS`, 64 in svs 0.1.0), i.e. when more
/// writes are in flight than that. The write can be retried.
fn is_svs_insert_slot_exhausted(e: &postgres::Error) -> bool {
    e.code() == Some(&postgres::error::SqlState::PROGRAM_LIMIT_EXCEEDED)
        && e.as_db_error()
            .is_some_and(|db| db.message().contains("pending SVS insert batches"))
}

/// The server's message for a database error (`Display` only says "db error").
fn pg_error_message(e: &postgres::Error) -> String {
    e.as_db_error()
        .map(|db| db.message().to_string())
        .unwrap_or_else(|| e.to_string())
}

// ── PgVector condition parser ─────────────────────────────────────

/// Map a dataset distance name to the pgvector `(operator, ops_class)` pair.
/// Unknown metrics error. A wrong arm here (e.g. L2 op with a cosine ops class)
/// would silently change ranking, so every arm is unit-tested. `<#>` is the
/// negative inner product, so ascending order ranks by descending similarity.
fn map_pg_distance_ops(distance: &str) -> Result<(&'static str, &'static str), String> {
    match distance.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok(("<->", "vector_l2_ops")),
        "cosine" | "angular" => Ok(("<=>", "vector_cosine_ops")),
        "dot" | "ip" => Ok(("<#>", "vector_ip_ops")),
        other => Err(format!(
            "PgVector does not support distance metric: {}",
            other
        )),
    }
}

/// A filter value bound as a query parameter (instead of inlined into the SQL
/// text) so the prepared-statement text stays stable across queries in a run and
/// the server reuses one plan. Each variant owns its data and lends a
/// `&(dyn ToSql + Sync)` for binding via `as_sql`.
#[derive(Debug, Clone, PartialEq)]
enum PgValue {
    Text(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    TextArray(Vec<String>),
    IntArray(Vec<i64>),
    FloatArray(Vec<f64>),
}

impl PgValue {
    fn as_sql(&self) -> &(dyn ToSql + Sync) {
        match self {
            PgValue::Text(v) => v,
            PgValue::Int(v) => v,
            PgValue::Float(v) => v,
            PgValue::Bool(v) => v,
            PgValue::TextArray(v) => v,
            PgValue::IntArray(v) => v,
            PgValue::FloatArray(v) => v,
        }
    }
}

/// Accumulates the ordered list of bound values and hands out sequential `$N`
/// placeholders so the emitted template and the values line up positionally.
struct FilterBuilder {
    next_param: usize,
    values: Vec<PgValue>,
}

impl FilterBuilder {
    /// Record `v` as the next bound value and return its `$N` placeholder text.
    fn bind(&mut self, v: PgValue) -> String {
        let n = self.next_param;
        self.next_param += 1;
        self.values.push(v);
        format!("${}", n)
    }
}

/// Parse a dataset filter into a parameterized SQL template + the ordered typed
/// values to bind. `first_param` is the placeholder number the filter's first
/// value takes (callers pass 3: $1 = query vector, $2 = LIMIT). The template
/// text depends only on the filter *shape* (fields/operators), never on the
/// values, so distinct queries in a run share one prepared statement.
fn parse_pg_conditions(
    conditions: &serde_json::Value,
    first_param: usize,
) -> Option<(String, Vec<PgValue>)> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut builder = FilterBuilder {
        next_param: first_param,
        values: Vec::new(),
    };

    let template = build_pg_group(obj, &mut builder)?;
    Some((template, builder.values))
}

/// Build the SQL for a `{and:[...], or:[...]}` group. Each `and`/`or` array
/// becomes a parenthesised sub-expression joined by SQL `AND`/`OR`; the two
/// groups (when both present) are combined with `AND`. Entries that are
/// themselves nested groups recurse through `build_pg_entry`, so an arbitrarily
/// deep tree like `(a AND b) OR (c AND d)` maps to correctly parenthesised SQL.
fn build_pg_group(
    obj: &serde_json::Map<String, serde_json::Value>,
    builder: &mut FilterBuilder,
) -> Option<String> {
    let mut clauses = Vec::new();

    if let Some(and_entries) = obj.get("and").and_then(|v| v.as_array()) {
        let sub: Vec<String> = and_entries
            .iter()
            .filter_map(|e| build_pg_entry(e, builder))
            .collect();
        if !sub.is_empty() {
            clauses.push(format!("({})", sub.join(" AND ")));
        }
    }

    if let Some(or_entries) = obj.get("or").and_then(|v| v.as_array()) {
        let sub: Vec<String> = or_entries
            .iter()
            .filter_map(|e| build_pg_entry(e, builder))
            .collect();
        if !sub.is_empty() {
            clauses.push(format!("({})", sub.join(" OR ")));
        }
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
}

/// Build one entry of an `and`/`or` array. An entry is EITHER a nested group
/// (it carries an `and`/`or` key) — built as its own parenthesised sub-clause so
/// the engine's native SQL grouping preserves precedence — OR a field leaf,
/// dispatched to `build_pg_clause`. This is the recursion that keeps a
/// mis-flattening builder from collapsing `(a AND b) OR (c AND d)` into a flat
/// disjunction.
fn build_pg_entry(entry: &serde_json::Value, builder: &mut FilterBuilder) -> Option<String> {
    if let Some(entry_obj) = entry.as_object() {
        if entry_obj.contains_key("and") || entry_obj.contains_key("or") {
            return build_pg_group(entry_obj, builder).map(|s| format!("({})", s));
        }
    }
    build_pg_clause(entry, builder)
}

fn build_pg_clause(entry: &serde_json::Value, builder: &mut FilterBuilder) -> Option<String> {
    let entry_obj = entry.as_object()?;
    let mut parts = Vec::new();
    for (field_name, field_filters) in entry_obj {
        let filter_obj = field_filters.as_object()?;
        for (condition_type, criteria) in filter_obj {
            match condition_type.as_str() {
                "match" => {
                    // match_any: OR-of-values, mirroring qdrant's
                    // Condition::matches(field, Vec).
                    if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                        if !any.is_empty() && any.iter().all(|v| v.is_number()) {
                            // Numeric contains-any on a scalar numeric column. The
                            // WHOLE list is bound as ONE array param and tested with
                            // `= ANY(...)` (equivalent to `IN`), so the SQL text is
                            // independent of the list length — a stable statement.
                            if any.iter().all(|v| v.is_i64() || v.is_u64()) {
                                let nums: Vec<i64> = any.iter().map(json_as_i64).collect();
                                let ph = builder.bind(PgValue::IntArray(nums));
                                parts.push(format!("\"{}\" = ANY({}::bigint[])", field_name, ph));
                            } else {
                                let nums: Vec<f64> =
                                    any.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
                                let ph = builder.bind(PgValue::FloatArray(nums));
                                parts.push(format!(
                                    "\"{}\" = ANY({}::double precision[])",
                                    field_name, ph
                                ));
                            }
                        } else {
                            // Keyword contains-any. Multi-valued keyword payloads
                            // are stored as a single ';'-joined TEXT scalar (see
                            // copy_field), so a scalar `IN` can never match an
                            // array-valued doc. Split the column on ';' and test
                            // set intersection with the query values — correct for
                            // both single-valued ('red' -> {red}) and multi-valued
                            // ('red;green' -> {red,green}) keyword fields, mirroring
                            // qdrant's array-contains-any semantics. The value set is
                            // a single bound text[] param (no manual escaping, stable
                            // statement text regardless of list length).
                            let elems: Vec<String> = any
                                .iter()
                                .filter_map(|v| v.as_str())
                                .map(|s| s.to_string())
                                .collect();
                            if elems.is_empty() {
                                // Empty (or no representable) IN-set matches
                                // NOTHING; a bare `false` keeps that in both AND
                                // and OR contexts instead of dropping the clause
                                // (which as the sole condition would return every
                                // row — the inverse of the filter).
                                parts.push("false".to_string());
                            } else {
                                let ph = builder.bind(PgValue::TextArray(elems));
                                parts.push(format!(
                                    "string_to_array(\"{}\", ';') && {}::text[]",
                                    field_name, ph
                                ));
                            }
                        }
                    } else if let Some(value) = criteria.get("value") {
                        if let Some(s) = value.as_str() {
                            let ph = builder.bind(PgValue::Text(s.to_string()));
                            if is_multivalued_keyword_field(field_name) {
                                // Multi-valued keyword: the ';'-joined column holds
                                // a set, so exact-match means "contains this value".
                                // Test set membership, mirroring the match_any arm
                                // above (issue #88) — a scalar `=` compares the whole
                                // joined string and never matches one element.
                                parts.push(format!(
                                    "{} = ANY(string_to_array(\"{}\", ';'))",
                                    ph, field_name
                                ));
                            } else {
                                parts.push(format!("\"{}\" = {}", field_name, ph));
                            }
                        } else if let Some(b) = value.as_bool() {
                            let ph = builder.bind(PgValue::Bool(b));
                            parts.push(format!("\"{}\" = {}", field_name, ph));
                        } else if value.is_i64() || value.is_u64() {
                            let ph = builder.bind(PgValue::Int(json_as_i64(value)));
                            parts.push(format!("\"{}\" = {}", field_name, ph));
                        } else if let Some(f) = value.as_f64() {
                            let ph = builder.bind(PgValue::Float(f));
                            parts.push(format!("\"{}\" = {}", field_name, ph));
                        } else {
                            // Non-scalar match value (a JSON array/object/null)
                            // is malformed input — the canonical model uses
                            // `match.any` for lists. Drop the clause (emit
                            // nothing) instead of inlining `"f" = [1,2]`
                            // verbatim, matching qdrant/redis/valkey/vectorsets.
                            // As the sole condition this leaves `parts` empty →
                            // the builder returns None.
                        }
                    } else if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                        // Full-text match: a `{match:{text}}` clause over a TEXT
                        // column. Postgres FTS token containment via
                        // `to_tsvector(col) @@ plainto_tsquery(term)`. Dropping the
                        // clause would run the kNN query UNFILTERED while recall is
                        // scored against the filtered ground truth.
                        let ph = builder.bind(PgValue::Text(text.to_string()));
                        parts.push(format!(
                            "to_tsvector('english', \"{}\") @@ plainto_tsquery('english', {})",
                            field_name, ph
                        ));
                    }
                }
                "range" => {
                    if let Some(co) = criteria.as_object() {
                        for (op, val) in co {
                            let sql_op = match op.as_str() {
                                "lt" => "<",
                                "gt" => ">",
                                "lte" => "<=",
                                "gte" => ">=",
                                _ => continue,
                            };
                            if val.is_null() {
                                continue;
                            }
                            if val.is_i64() || val.is_u64() {
                                let ph = builder.bind(PgValue::Int(json_as_i64(val)));
                                parts.push(format!("\"{}\" {} {}", field_name, sql_op, ph));
                            } else if let Some(f) = val.as_f64() {
                                let ph = builder.bind(PgValue::Float(f));
                                parts.push(format!("\"{}\" {} {}", field_name, sql_op, ph));
                            } else if let Some(s) = val.as_str() {
                                // A string bound is an ISO-8601 datetime range over a
                                // TIMESTAMPTZ column. Inline as a single-quoted SQL
                                // literal + cast: binding `$n::timestamptz` makes
                                // Postgres infer the param type as timestamptz and
                                // reject a text value ("error serializing parameter").
                                // Single-quote-escape the benchmark-controlled ISO
                                // string (was previously inlined bare, i.e. rendered
                                // with double quotes → a broken SQL identifier).
                                let esc = s.replace('\'', "''");
                                parts.push(format!(
                                    "\"{}\" {} '{}'::timestamptz",
                                    field_name, sql_op, esc
                                ));
                            } else {
                                // Non-scalar range bound (degenerate) — inline.
                                parts.push(format!("\"{}\" {} {}", field_name, sql_op, val));
                            }
                        }
                    }
                }
                "geo" => {
                    // Geo-radius over the "lat,lon" TEXT column via earthdistance
                    // (great-circle metres). lat/lon/radius are dataset numbers
                    // (not injectable), so inline them; the stored point is split
                    // back into (lat, lon) for ll_to_earth.
                    if let (Some(lat), Some(lon), Some(radius)) = (
                        criteria.get("lat").and_then(|v| v.as_f64()),
                        criteria.get("lon").and_then(|v| v.as_f64()),
                        criteria.get("radius").and_then(|v| v.as_f64()),
                    ) {
                        parts.push(format!(
                            "earth_distance(ll_to_earth(split_part(\"{f}\", ',', 1)::float8, \
                             split_part(\"{f}\", ',', 2)::float8), ll_to_earth({lat}, {lon})) <= {radius}",
                            f = field_name,
                            lat = lat,
                            lon = lon,
                            radius = radius
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}

/// Coerce a JSON number to i64, tolerating u64 values that fit (falling back to
/// a saturating cast for the rare > i64::MAX case).
fn json_as_i64(v: &serde_json::Value) -> i64 {
    v.as_i64()
        .or_else(|| v.as_u64().map(|u| u as i64))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};

    #[test]
    fn maps_schema_types_to_pg_columns() {
        assert_eq!(pg_column_type("keyword"), Some("TEXT"));
        assert_eq!(pg_column_type("text"), Some("TEXT"));
        assert_eq!(pg_column_type("int"), Some("BIGINT"));
        assert_eq!(pg_column_type("float"), Some("DOUBLE PRECISION"));
        assert_eq!(pg_column_type("bool"), Some("BOOLEAN"));
        assert_eq!(pg_column_type("datetime"), Some("TIMESTAMPTZ"));
        // Geo is now stored as a "lat,lon" TEXT scalar (earthdistance filter).
        assert_eq!(pg_column_type("geo"), Some("TEXT"));
        assert_eq!(pg_column_type("uuid"), Some("TEXT"));
        assert_eq!(pg_column_type("unknown"), None);
    }

    #[test]
    fn escapes_copy_text_specials() {
        assert_eq!(escape_copy_text("a\tb\nc\\d"), "a\\tb\\nc\\\\d");
    }

    #[test]
    fn copy_field_reads_value_or_null() {
        let meta = MetadataItem {
            fields: vec![
                (
                    "category".to_string(),
                    MetadataValue::String("shoes".to_string()),
                ),
                (
                    "labels".to_string(),
                    MetadataValue::Labels(vec!["a".to_string(), "b".to_string()]),
                ),
            ],
        };
        assert_eq!(copy_field(Some(&meta), "category"), "shoes");
        assert_eq!(copy_field(Some(&meta), "labels"), "a;b");
        assert_eq!(copy_field(Some(&meta), "missing"), "\\N");
        assert_eq!(copy_field(None, "category"), "\\N");
    }

    // Native numeric fields (issue #87) must COPY the literal digits into the
    // BIGINT/DOUBLE column — NOT fall through to the `_ => NULL` arm.
    #[test]
    fn copy_field_emits_numeric_literals() {
        let meta = MetadataItem {
            fields: vec![
                ("size".to_string(), MetadataValue::Int(42)),
                ("price".to_string(), MetadataValue::Float(3.5)),
            ],
        };
        assert_eq!(copy_field(Some(&meta), "size"), "42");
        assert_eq!(copy_field(Some(&meta), "price"), "3.5");
    }

    // Parse with the production base offset ($1 = vector, $2 = LIMIT, filter
    // values start at $3) and return the (template, values) pair.
    fn parse(cond: &serde_json::Value) -> Option<(String, Vec<PgValue>)> {
        parse_pg_conditions(cond, 3)
    }

    #[test]
    fn filter_clause_quotes_columns_and_binds_values() {
        // AND of a keyword match and a numeric range → quoted columns, values as
        // sequential placeholders (NOT inlined), values captured in bind order.
        let cond = json!({
            "and": [
                {"category": {"match": {"value": "shoes"}}},
                {"year": {"range": {"gte": 2020}}}
            ]
        });
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"category\" = $3 AND \"year\" >= $4)");
        assert_eq!(
            vals,
            vec![PgValue::Text("shoes".to_string()), PgValue::Int(2020)]
        );
    }

    #[test]
    fn match_value_binds_string_without_inline_escaping() {
        // The value is a bound param — no SQL text escaping, no value in the text.
        let cond = json!({"and": [{"name": {"match": {"value": "O'Brien"}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"name\" = $3)");
        assert_eq!(vals, vec![PgValue::Text("O'Brien".to_string())]);
    }

    #[test]
    fn statement_text_is_stable_across_differing_values() {
        // Same filter SHAPE, different values (tenant_7 vs tenant_3) → identical
        // statement text so the server reuses one prepared plan. Only the bound
        // values differ.
        let (sql7, v7) =
            parse(&json!({"and":[{"tenant":{"match":{"value":"tenant_7"}}}]})).unwrap();
        let (sql3, v3) =
            parse(&json!({"and":[{"tenant":{"match":{"value":"tenant_3"}}}]})).unwrap();
        assert_eq!(
            sql7, sql3,
            "statement text must be identical across queries"
        );
        assert_eq!(sql7, "(\"tenant\" = $3)");
        assert_eq!(v7, vec![PgValue::Text("tenant_7".to_string())]);
        assert_eq!(v3, vec![PgValue::Text("tenant_3".to_string())]);
    }

    #[test]
    fn first_param_offset_is_honored() {
        // Placeholder numbering starts at the caller-supplied base.
        let (sql, _) =
            parse_pg_conditions(&json!({"and":[{"a":{"match":{"value":"x"}}}]}), 5).unwrap();
        assert_eq!(sql, "(\"a\" = $5)");
    }

    #[test]
    fn match_any_string_list_binds_text_array_overlap() {
        // Contains-any via set intersection, so it matches both single-valued and
        // ';'-joined multi-valued keyword columns. The whole set is one text[]
        // param, so the statement text is independent of the list length.
        let cond = json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(string_to_array(\"color\", ';') && $3::text[])");
        assert_eq!(
            vals,
            vec![PgValue::TextArray(vec![
                "red".to_string(),
                "blue".to_string()
            ])]
        );
    }

    #[test]
    fn labels_exact_match_uses_set_membership() {
        // #88: exact-match on the multi-valued `labels` column means "contains
        // this value" — test set membership, not whole-string equality. A
        // single-valued keyword (`name`) keeps scalar `=`.
        let cond = json!({"and": [{"labels": {"match": {"value": "red"}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "($3 = ANY(string_to_array(\"labels\", ';')))");
        assert_eq!(vals, vec![PgValue::Text("red".to_string())]);

        let (sql_name, _) =
            parse(&json!({"and": [{"name": {"match": {"value": "red"}}}]})).unwrap();
        assert_eq!(sql_name, "(\"name\" = $3)");
    }

    #[test]
    fn match_any_int_list_binds_bigint_array_any() {
        // `= ANY($N::bigint[])` is equivalent to `IN`, with the list bound as one
        // array param (stable statement text regardless of length).
        let cond = json!({"and": [{"size": {"match": {"any": [1, 2, 3]}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"size\" = ANY($3::bigint[]))");
        assert_eq!(vals, vec![PgValue::IntArray(vec![1, 2, 3])]);
    }

    #[test]
    fn match_any_float_list_binds_double_array_any() {
        let cond = json!({"and": [{"score": {"match": {"any": [1.5, 2.5]}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"score\" = ANY($3::double precision[]))");
        assert_eq!(vals, vec![PgValue::FloatArray(vec![1.5, 2.5])]);
    }

    #[test]
    fn match_any_string_values_are_bound_not_escaped() {
        let cond = json!({"and": [{"name": {"match": {"any": ["O'Brien"]}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(string_to_array(\"name\", ';') && $3::text[])");
        assert_eq!(vals, vec![PgValue::TextArray(vec!["O'Brien".to_string()])]);
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // An empty IN-set must match NOTHING (never invert to returning all
        // rows), so the clause is a bare `false` (no bound value) rather than
        // being dropped.
        let cond = json!({"and": [{"color": {"match": {"any": []}}}]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(false)");
        assert!(vals.is_empty());
    }

    // #121: a non-scalar `value` (a JSON array) is malformed input — the
    // canonical model uses `match.any` for lists. The clause is dropped (no SQL,
    // no bound value) rather than inlining `"n" = [1,2]` verbatim. As the sole
    // condition the whole filter is None. Matches qdrant/redis/valkey/vectorsets.
    #[test]
    fn match_non_scalar_value_dropped() {
        assert!(parse(&json!({"and": [{"n": {"match": {"value": [1, 2]}}}]})).is_none());
        assert!(parse(&json!({"and": [{"n": {"match": {"value": {"x": 1}}}}]})).is_none());
        assert!(parse(&json!({"and": [{"n": {"match": {"value": null}}}]})).is_none());
        // A non-scalar clause is dropped but a sibling scalar clause survives.
        let cond = json!({"and": [
            {"n": {"match": {"value": [1, 2]}}},
            {"c": {"match": {"value": "red"}}},
        ]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"c\" = $3)");
        assert_eq!(vals, vec![PgValue::Text("red".to_string())]);
    }

    // ── OR-branch of the condition parser ──────────────────────────────────

    #[test]
    fn or_only_emits_or_joined_group() {
        let cond = json!({"or":[
            {"a":{"match":{"value":"x"}}},
            {"b":{"match":{"value":"y"}}},
        ]});
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"a\" = $3 OR \"b\" = $4)");
        assert_eq!(
            vals,
            vec![
                PgValue::Text("x".to_string()),
                PgValue::Text("y".to_string())
            ]
        );
    }

    #[test]
    fn and_plus_or_keeps_both_groups() {
        // AND group binds first ($3), then OR group ($4).
        let cond = json!({
            "and":[{"a":{"match":{"value":"x"}}}],
            "or":[{"b":{"match":{"value":"y"}}}],
        });
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(sql, "(\"a\" = $3) AND (\"b\" = $4)");
        assert_eq!(
            vals,
            vec![
                PgValue::Text("x".to_string()),
                PgValue::Text("y".to_string())
            ]
        );
    }

    // ── Nested / grouped boolean tree ──────────────────────────────────────

    #[test]
    fn nested_or_of_and_groups_parenthesises_each_subgroup() {
        // (color==red AND size>=50) OR (color==blue AND size<10). Each nested
        // AND-group must become its OWN parenthesised sub-expression so the top
        // OR does not mis-flatten into `red AND size>=50 OR color==blue AND ...`.
        let cond = json!({
            "or": [
                {"and": [
                    {"color": {"match": {"value": "red"}}},
                    {"size": {"range": {"gte": 50}}}
                ]},
                {"and": [
                    {"color": {"match": {"value": "blue"}}},
                    {"size": {"range": {"lt": 10}}}
                ]}
            ]
        });
        let (sql, vals) = parse(&cond).unwrap();
        assert_eq!(
            sql,
            "(((\"color\" = $3 AND \"size\" >= $4)) OR ((\"color\" = $5 AND \"size\" < $6)))"
        );
        assert_eq!(
            vals,
            vec![
                PgValue::Text("red".to_string()),
                PgValue::Int(50),
                PgValue::Text("blue".to_string()),
                PgValue::Int(10),
            ]
        );
    }

    // ── Range operators ────────────────────────────────────────────────────

    fn range_sql(criteria: serde_json::Value) -> Option<(String, Vec<PgValue>)> {
        parse(&json!({"and":[{"n":{"range":criteria}}]}))
    }

    #[test]
    fn range_lt_lte_gt_gte() {
        assert_eq!(
            range_sql(json!({"lt":5})).unwrap(),
            ("(\"n\" < $3)".to_string(), vec![PgValue::Int(5)])
        );
        assert_eq!(
            range_sql(json!({"lte":5})).unwrap(),
            ("(\"n\" <= $3)".to_string(), vec![PgValue::Int(5)])
        );
        assert_eq!(
            range_sql(json!({"gt":5})).unwrap(),
            ("(\"n\" > $3)".to_string(), vec![PgValue::Int(5)])
        );
        assert_eq!(
            range_sql(json!({"gte":5})).unwrap(),
            ("(\"n\" >= $3)".to_string(), vec![PgValue::Int(5)])
        );
    }

    #[test]
    fn range_two_sided_gte_lt() {
        // Criteria object keys iterate in sorted order (BTreeMap): gte, then lt.
        let (sql, vals) = range_sql(json!({"gte":10,"lt":20})).unwrap();
        assert_eq!(sql, "(\"n\" >= $3 AND \"n\" < $4)");
        assert_eq!(vals, vec![PgValue::Int(10), PgValue::Int(20)]);
    }

    #[test]
    fn range_unknown_op_is_none() {
        assert!(range_sql(json!({"foo":5})).is_none());
    }

    #[test]
    fn range_null_bound_is_none() {
        assert!(range_sql(json!({"gte":serde_json::Value::Null})).is_none());
    }

    // ── Geo filter (unsupported → None) ────────────────────────────────────

    #[test]
    fn geo_builds_earthdistance_clause() {
        // Geo-radius is filtered via earthdistance over the "lat,lon" TEXT column:
        // earth_distance(ll_to_earth(lat, lon), ll_to_earth(qlat, qlon)) <= radius.
        let cond = json!({"and":[{"loc":{"geo":{"lat":20.0,"lon":10.0,"radius":5}}}]});
        let (sql, binds) = parse(&cond).expect("geo clause should be built");
        assert!(sql.contains("earth_distance"), "sql: {sql}");
        assert!(sql.contains("ll_to_earth(20"), "query point inlined: {sql}");
        assert!(sql.contains("<= 5"), "radius inlined: {sql}");
        assert!(binds.is_empty(), "geo inlines its numeric args, no binds");
    }

    // ── Distance-metric mapping ────────────────────────────────────────────

    #[test]
    fn distance_ops_mapping_covers_all_arms() {
        assert_eq!(map_pg_distance_ops("l2").unwrap(), ("<->", "vector_l2_ops"));
        assert_eq!(
            map_pg_distance_ops("euclidean").unwrap(),
            ("<->", "vector_l2_ops")
        );
        assert_eq!(
            map_pg_distance_ops("cosine").unwrap(),
            ("<=>", "vector_cosine_ops")
        );
        assert_eq!(
            map_pg_distance_ops("angular").unwrap(),
            ("<=>", "vector_cosine_ops")
        );
        assert_eq!(
            map_pg_distance_ops("COSINE").unwrap(),
            ("<=>", "vector_cosine_ops")
        );
        assert_eq!(
            map_pg_distance_ops("dot").unwrap(),
            ("<#>", "vector_ip_ops")
        );
        assert_eq!(map_pg_distance_ops("ip").unwrap(), ("<#>", "vector_ip_ops"));
        assert!(map_pg_distance_ops("nope").is_err());
    }

    // ── Index config ───────────────────────────────────────────────────────

    fn engine_config(v: serde_json::Value) -> EngineConfig {
        serde_json::from_value(v).unwrap()
    }

    fn search_params(v: serde_json::Value) -> SearchParams {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn index_defaults_to_hnsw() {
        let cfg = engine_config(json!({"name": "pg", "engine": "pgvector"}));
        assert_eq!(
            parse_pg_index(&cfg).unwrap(),
            PgIndex::Hnsw {
                m: 16,
                ef_construction: 128
            }
        );
        let cfg = engine_config(json!({
            "name": "pg", "engine": "pgvector",
            "collection_params": {"hnsw_config": {"m": 32, "ef_construct": 256}}
        }));
        assert_eq!(
            parse_pg_index(&cfg).unwrap(),
            PgIndex::Hnsw {
                m: 32,
                ef_construction: 256
            }
        );
    }

    #[test]
    fn index_selects_vamana_from_collection_params_or_top_level() {
        let nested = engine_config(json!({
            "name": "pg", "engine": "pgvector",
            "collection_params": {"algorithm": "svs-vamana"}
        }));
        assert_eq!(
            parse_pg_index(&nested).unwrap(),
            PgIndex::Vamana(VamanaConfig::default())
        );
        let top = engine_config(json!({
            "name": "pg", "engine": "pgvector", "algorithm": "SVS-VAMANA"
        }));
        assert!(matches!(parse_pg_index(&top).unwrap(), PgIndex::Vamana(_)));
        let bad = engine_config(json!({"name": "pg", "engine": "pgvector", "algorithm": "ivf"}));
        assert!(parse_pg_index(&bad).is_err());
    }

    #[test]
    fn vamana_config_passes_options_through() {
        let cfg = engine_config(json!({
            "name": "pg", "engine": "pgvector",
            "collection_params": {
                "algorithm": "svs-vamana",
                "svs-vamana_config": {
                    "graph_degree": 32,
                    "alpha": 120,
                    "use_search_history": true,
                    "compression_type": "it's"
                }
            }
        }));
        let PgIndex::Vamana(v) = parse_pg_index(&cfg).unwrap() else {
            panic!("expected Vamana");
        };
        let mut opts = v.with_options.clone();
        opts.sort();
        assert_eq!(
            opts,
            vec![
                ("alpha".to_string(), "120".to_string()),
                ("compression_type".to_string(), "'it''s'".to_string()),
                ("graph_degree".to_string(), "32".to_string()),
                ("use_search_history".to_string(), "true".to_string()),
            ]
        );
    }

    #[test]
    fn vamana_config_rejects_unsafe_or_malformed_entries() {
        let parse = |block: serde_json::Value| parse_vamana_config(block.as_object().unwrap());
        // Redis-style uppercase key, and anything that could escape the DDL.
        assert!(parse(json!({"GRAPH_MAX_DEGREE": 32})).is_err());
        assert!(parse(json!({"graph_degree = 1); DROP TABLE items; --": 1})).is_err());
        assert!(parse(json!({"graph_degree": [32]})).is_err());
        // Build settings belong in collection_params.
        assert!(parse(json!({"max_parallel_workers": 4})).is_err());
        assert!(parse(json!({"maintenance_work_mem": "8GB"})).is_err());
    }

    fn cp_extra(v: serde_json::Value) -> std::collections::HashMap<String, serde_json::Value> {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn build_settings_from_collection_params() {
        let b = parse_build_settings(Some(&cp_extra(
            json!({"maintenance_work_mem": "8GB", "max_parallel_workers": 4}),
        )))
        .unwrap();
        assert_eq!(b.maintenance_work_mem.as_deref(), Some("'8GB'"));
        assert_eq!(b.max_parallel_workers, Some(4));
        assert_eq!(
            build_session_sql(&b),
            vec![
                "SET maintenance_work_mem = '8GB'",
                "SET max_parallel_maintenance_workers = 4",
                "ALTER TABLE items SET (parallel_workers = 4)",
            ]
        );
        let kb = parse_build_settings(Some(&cp_extra(json!({"maintenance_work_mem": 65536}))));
        assert_eq!(kb.unwrap().maintenance_work_mem.as_deref(), Some("65536"));
        assert!(
            parse_build_settings(Some(&cp_extra(json!({"max_parallel_workers": -1})))).is_err()
        );
        assert!(
            parse_build_settings(Some(&cp_extra(json!({"maintenance_work_mem": true})))).is_err()
        );
        assert!(build_session_sql(&parse_build_settings(None).unwrap()).is_empty());
    }

    #[test]
    fn vector_type_from_data_type() {
        let parse = |extra: serde_json::Value| {
            let mut cfg = json!({"name": "pg", "engine": "pgvector"});
            cfg.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            parse_vector_type(&engine_config(cfg))
        };
        assert_eq!(parse(json!({})).unwrap(), VectorType::Vector);
        assert_eq!(
            parse(json!({"collection_params": {"data_type": "float16"}})).unwrap(),
            VectorType::HalfVec
        );
        // Redis configs also carry it in upload_params.
        assert_eq!(
            parse(json!({"upload_params": {"data_type": "FLOAT16"}})).unwrap(),
            VectorType::HalfVec
        );
        assert_eq!(
            parse(json!({"collection_params": {"data_type": "FLOAT32"}})).unwrap(),
            VectorType::Vector
        );
        assert!(parse(json!({"collection_params": {"data_type": "INT8"}})).is_err());
    }

    #[test]
    fn halfvec_casts_query_params_and_ops() {
        assert_eq!(VectorType::Vector.param(1), "$1");
        assert_eq!(VectorType::HalfVec.param(2), "$2::vector::halfvec");
        assert_eq!(
            VectorType::Vector.ops_class("vector_ip_ops"),
            "vector_ip_ops"
        );
        assert_eq!(
            VectorType::HalfVec.ops_class("vector_ip_ops"),
            "halfvec_ip_ops"
        );
        assert_eq!(
            upsert_sql(VectorType::HalfVec),
            "INSERT INTO items (id, embedding) VALUES ($1, $2::vector::halfvec) \
             ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
        );
    }

    fn dataset(distance: &str) -> Dataset {
        Dataset::new(
            serde_json::from_value(json!({"name": "d", "path": "d", "distance": distance}))
                .unwrap(),
        )
    }

    fn new_engine(cp: serde_json::Value) -> Result<PgVectorEngine, String> {
        PgVectorEngine::new(
            &engine_config(json!({"name": "pg", "engine": "pgvector", "collection_params": cp})),
            "localhost",
        )
    }

    #[test]
    fn distance_override_only_on_normalized_datasets() {
        let ip = new_engine(json!({"distance": "dot", "data_type": "FLOAT16"})).unwrap();
        assert_eq!(
            ip.distance_ops(&dataset("cosine")).unwrap(),
            ("<#>", "halfvec_ip_ops".to_string())
        );
        assert_eq!(
            ip.distance_ops(&dataset("dot")).unwrap(),
            ("<#>", "halfvec_ip_ops".to_string())
        );
        // L2 data is not normalized, so IP would rank differently.
        assert!(ip.distance_ops(&dataset("l2")).is_err());

        let l2 = new_engine(json!({"normalize_cosine": true})).unwrap();
        assert_eq!(
            l2.distance_ops(&dataset("angular")).unwrap(),
            ("<->", "vector_l2_ops".to_string())
        );
        assert_eq!(
            new_engine(json!({}))
                .unwrap()
                .distance_ops(&dataset("cosine"))
                .unwrap(),
            ("<=>", "vector_cosine_ops".to_string())
        );
        assert!(new_engine(json!({"normalize_cosine": true, "distance": "dot"})).is_err());
        assert!(new_engine(json!({"distance": "hamming"})).is_err());
    }

    #[test]
    fn index_after_load_is_forced_for_vamana() {
        assert!(!new_engine(json!({})).unwrap().index_after_load);
        assert!(
            new_engine(json!({"index_after_load": true}))
                .unwrap()
                .index_after_load
        );
        assert!(
            new_engine(json!({"algorithm": "svs-vamana"}))
                .unwrap()
                .index_after_load
        );
        assert_eq!(
            new_engine(json!({"hnsw_config": {"m": 8, "ef_construct": 64}}))
                .unwrap()
                .index_sql("halfvec_ip_ops"),
            "CREATE INDEX ON items USING hnsw(embedding halfvec_ip_ops) WITH (m = 8, ef_construction = 64)"
        );
    }

    #[test]
    fn vamana_index_sql_omits_with_clause_when_no_options() {
        assert_eq!(
            vamana_index_sql("vector_ip_ops", &VamanaConfig::default()),
            "CREATE INDEX ON items USING vamana(embedding vector_ip_ops)"
        );
        let cfg = VamanaConfig {
            with_options: vec![
                ("graph_degree".to_string(), "64".to_string()),
                ("build_window_size".to_string(), "128".to_string()),
            ],
        };
        assert_eq!(
            vamana_index_sql("vector_l2_ops", &cfg),
            "CREATE INDEX ON items USING vamana(embedding vector_l2_ops) \
             WITH (graph_degree = 64, build_window_size = 128)"
        );
    }

    #[test]
    fn search_knob_per_index() {
        let hnsw = PgIndex::Hnsw {
            m: 16,
            ef_construction: 128,
        };
        let vamana = PgIndex::Vamana(VamanaConfig::default());

        let none = search_params(json!({"parallel": 1}));
        assert_eq!(search_knob(&hnsw, &none), Some(128));
        assert_eq!(search_knob(&vamana, &none), None);

        let hnsw_ef = search_params(json!({"search_params": {"hnsw_ef": 64}}));
        assert_eq!(search_knob(&hnsw, &hnsw_ef), Some(64));
        assert_eq!(search_knob(&vamana, &hnsw_ef), None);

        // Calibration writes `ef` into InnerSearchParams.ef, any other
        // calibration_param into its extras.
        let ef = search_params(json!({"search_params": {"ef": 200}}));
        assert_eq!(search_knob(&hnsw, &ef), Some(200));
        assert_eq!(search_knob(&vamana, &ef), Some(200));

        let window = search_params(json!({"search_params": {"search_window_size": 48, "ef": 200}}));
        assert_eq!(search_knob(&vamana, &window), Some(48));
        let upper = search_params(json!({"search_params": {"SEARCH_WINDOW_SIZE": 96}}));
        assert_eq!(search_knob(&vamana, &upper), Some(96));
    }

    // ── Exact-match numeric / bool / non-scalar arms ───────────────────────

    #[test]
    fn exact_match_int_float_bool() {
        // Scalar match values become typed bound params (stable statement text).
        assert_eq!(
            parse(&json!({"and":[{"n":{"match":{"value":5}}}]})).unwrap(),
            ("(\"n\" = $3)".to_string(), vec![PgValue::Int(5)])
        );
        assert_eq!(
            parse(&json!({"and":[{"n":{"match":{"value":1.5}}}]})).unwrap(),
            ("(\"n\" = $3)".to_string(), vec![PgValue::Float(1.5)])
        );
        assert_eq!(
            parse(&json!({"and":[{"flag":{"match":{"value":true}}}]})).unwrap(),
            ("(\"flag\" = $3)".to_string(), vec![PgValue::Bool(true)])
        );
    }

    #[test]
    fn exact_match_array_value_is_dropped() {
        // #121: a non-scalar match value (a JSON array) is malformed input — the
        // canonical model uses `match.any` for lists. The clause is dropped (as
        // the sole condition → None), matching qdrant/redis/valkey/vectorsets
        // (was previously inlined verbatim as `"n" = [1,2]`).
        assert!(parse(&json!({"and":[{"n":{"match":{"value":[1,2]}}}]})).is_none());
    }
}
