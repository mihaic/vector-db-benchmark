//! Integration tests for the PgVector engine.
//!
//! Requires PostgreSQL with pgvector extension running on port 5433.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d pgvector
//! Run with:   PGVECTOR_PORT=5433 cargo test --test integration_pgvector -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PG_PORT: u16 = 5433;
const PG_HOST: &str = "127.0.0.1";
const PG_DB: &str = "postgres";
const PG_USER: &str = "postgres";
const PG_PASSWORD: &str = "passwd";

fn connect() -> postgres::Client {
    let conn_str = format!(
        "host={} port={} dbname={} user={} password={}",
        PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
    );
    postgres::Client::connect(&conn_str, postgres::NoTls).expect("Failed to connect to PostgreSQL")
}

fn wait_for_postgres() {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if connect_try().is_ok() {
            return;
        }
        if Instant::now() > deadline {
            panic!("PostgreSQL not available on port {} after 60s", PG_PORT);
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn connect_try() -> Result<postgres::Client, postgres::Error> {
    let conn_str = format!(
        "host={} port={} dbname={} user={} password={}",
        PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
    );
    postgres::Client::connect(&conn_str, postgres::NoTls)
}

fn cleanup(conn: &mut postgres::Client) {
    let _ = conn.execute("DROP TABLE IF EXISTS items CASCADE", &[]);
}

fn setup_table(conn: &mut postgres::Client, dim: usize) {
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
        .unwrap();
    conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
        .unwrap();
    conn.execute(
        &format!(
            "CREATE TABLE items (id SERIAL PRIMARY KEY, embedding vector({}) NOT NULL)",
            dim
        ),
        &[],
    )
    .unwrap();
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn brute_force_neighbors_l2(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            (i as i64, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

fn brute_force_neighbors_cosine(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let q_norm: f64 = query
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let v_norm: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let dot: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let sim = if q_norm > 0.0 && v_norm > 0.0 {
                dot / (q_norm * v_norm)
            } else {
                0.0
            };
            (i as i64, 1.0 - sim) // cosine distance
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_pgvector_create_table_and_extension() {
    wait_for_postgres();
    let mut conn = connect();
    cleanup(&mut conn);

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
        .unwrap();
    conn.execute(
        "CREATE TABLE items (id SERIAL PRIMARY KEY, embedding vector(4) NOT NULL)",
        &[],
    )
    .unwrap();

    // Verify table exists
    let row = conn
        .query_one(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'items'",
            &[],
        )
        .unwrap();
    let count: i64 = row.get(0);
    assert_eq!(count, 1, "items table should exist");

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_copy_upload() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 4;
    let count = 50;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    // COPY bulk upload
    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    // Verify count
    let row = conn.query_one("SELECT COUNT(*) FROM items", &[]).unwrap();
    let db_count: i64 = row.get(0);
    assert_eq!(db_count, count as i64, "All vectors should be uploaded");

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_knn_l2_search() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 8;
    let count = 100;
    let top = 10;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    // Upload with COPY
    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    // Create HNSW index for L2
    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_l2_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    // Search using pgvector::Vector parameterized query
    let query_vec = pgvector::Vector::from(vectors[0].clone());
    let rows = conn
        .query(
            &format!(
                "SELECT id FROM items ORDER BY embedding <-> $1 LIMIT {}",
                top
            ),
            &[&query_vec],
        )
        .unwrap();

    let result_ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();
    assert!(!result_ids.is_empty(), "Search should return results");
    assert_eq!(
        result_ids[0], 0,
        "Query vector (id=0) should be its own nearest neighbor"
    );

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_knn_cosine_search() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 4;
    let count = 20;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    // Create cosine index
    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    let query_vec = pgvector::Vector::from(vectors[0].clone());
    let rows = conn
        .query(
            "SELECT id FROM items ORDER BY embedding <=> $1 LIMIT 5",
            &[&query_vec],
        )
        .unwrap();

    let result_ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();
    assert!(
        !result_ids.is_empty(),
        "Cosine search should return results"
    );
    assert_eq!(result_ids[0], 0, "Self should be top-1 for cosine");

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_precision_l2() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 8;
    let count = 100;
    let top = 10;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_l2_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    // Set high ef_search for near-exact results
    conn.execute("SET hnsw.ef_search = 200", &[]).unwrap();

    let query_idx = 42;
    let expected = brute_force_neighbors_l2(&vectors[query_idx], &vectors, top);

    let query_vec = pgvector::Vector::from(vectors[query_idx].clone());
    let rows = conn
        .query(
            &format!(
                "SELECT id FROM items ORDER BY embedding <-> $1 LIMIT {}",
                top
            ),
            &[&query_vec],
        )
        .unwrap();

    let result_ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.into_iter().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.9,
        "L2 precision should be >= 0.9 for small dataset with high ef_search, got {}",
        precision
    );

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_precision_cosine() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 8;
    let count = 100;
    let top = 10;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    conn.execute("SET hnsw.ef_search = 200", &[]).unwrap();

    let query_idx = 42;
    let expected = brute_force_neighbors_cosine(&vectors[query_idx], &vectors, top);

    let query_vec = pgvector::Vector::from(vectors[query_idx].clone());
    let rows = conn
        .query(
            &format!(
                "SELECT id FROM items ORDER BY embedding <=> $1 LIMIT {}",
                top
            ),
            &[&query_vec],
        )
        .unwrap();

    let result_ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.into_iter().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.9,
        "Cosine precision should be >= 0.9, got {}",
        precision
    );

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_full_cycle() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 4;
    let count = 30;
    let top = 5;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    // 1. Upload
    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    // Verify count
    let row = conn.query_one("SELECT COUNT(*) FROM items", &[]).unwrap();
    let db_count: i64 = row.get(0);
    assert_eq!(db_count, count as i64);

    // 2. Create index
    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_l2_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    // 3. Search
    let query_vec = pgvector::Vector::from(vectors[0].clone());
    let rows = conn
        .query(
            &format!(
                "SELECT id FROM items ORDER BY embedding <-> $1 LIMIT {}",
                top
            ),
            &[&query_vec],
        )
        .unwrap();
    assert_eq!(rows.len(), top, "Should return top-{} results", top);

    // 4. Delete
    conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
        .unwrap();

    // Verify deleted
    let row = conn
        .query_one(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'items'",
            &[],
        )
        .unwrap();
    let count: i64 = row.get(0);
    assert_eq!(count, 0, "items table should be deleted");
}

/// End-to-end `match_any`: filter a single-valued keyword field to an OR-set and
/// assert the engine returns the filtered nearest neighbours (recall vs ground
/// truth brute-forced over only the matching docs). Covers the keyword
/// contains-any arm on scalar data.
#[test]
fn test_binary_pgvector_match_any() {
    wait_for_postgres();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-ma", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_project(
        "match-any-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    assert!(
        common::run_binary(
            &proj.root,
            "pg-ma",
            "match-any-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")],
        ),
        "pgvector match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "pg-ma");
    println!("pgvector match_any recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector match_any recall {:.3} < 0.9",
        recall
    );
}

/// UUID exact-match filter end-to-end. Regression for the schema-type bug:
/// `uuid` hit `pg_column_type`'s `_ => None`, so no column was created and the
/// filter referenced a missing column. With `uuid` -> TEXT the exact `=` match
/// selects the quarter of docs whose `uid` equals UUIDS[0].
#[test]
fn test_binary_pgvector_uuid() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-uuid", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_uuid_project("uuid-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-uuid",
            "uuid-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector uuid run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-uuid");
    println!("pgvector uuid recall={:.3}", recall);
    assert!(recall >= 0.9, "pgvector uuid recall {:.3} < 0.9", recall);
}

/// Bool-field equality filter end-to-end. Regression for the missing-column bug:
/// `pg_column_type("bool")` returned None so no column was created and the filter
/// referenced a non-existent column (SQL error). With a BOOLEAN column, COPY
/// coerces the reader's "true"/"false" string and `flag = $1` selects the evens.
#[test]
fn test_binary_pgvector_bool() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-bool", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_bool_project("bool-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-bool",
            "bool-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector bool run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-bool");
    println!("pgvector bool recall={:.3}", recall);
    assert!(recall >= 0.9, "pgvector bool recall {:.3} < 0.9", recall);
}

/// Datetime range filter end-to-end. Regression for two bugs: no TIMESTAMPTZ
/// column was created, and the range builder inlined the ISO bound as a bare
/// (identifier-quoted) token → SQL error. With a TIMESTAMPTZ column and a
/// `$n::timestamptz` bind, the `[day 100, day 300)` window is selected.
#[test]
fn test_binary_pgvector_datetime() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-dt", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_datetime_project("dt-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-dt",
            "dt-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector datetime run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-dt");
    println!("pgvector datetime recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector datetime recall {:.3} < 0.9",
        recall
    );
}

/// Full-text filter end-to-end. Regression: a `{match:{text}}` clause was dropped
/// (the match arm only handled `value`/`any`), so the kNN query ran UNFILTERED
/// while recall was scored against the filtered ground truth. Now the TEXT column
/// is filtered with Postgres FTS (`to_tsvector @@ plainto_tsquery`), selecting the
/// docs whose body contains the query token.
#[test]
fn test_binary_pgvector_fulltext() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-ft", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_fulltext_project("ft-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-ft",
            "ft-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector fulltext run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-ft");
    println!("pgvector fulltext recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector fulltext recall {:.3} < 0.9",
        recall
    );
}

/// Geo-radius filter end-to-end. Regression: `geo` had no column type and no
/// filter arm, so it was dropped and the search ran unfiltered. Now the point is
/// stored as a "lat,lon" TEXT scalar and filtered with the earthdistance
/// extension (`earth_distance(ll_to_earth(...)) <= radius`), selecting the docs
/// within the query radius.
#[test]
fn test_binary_pgvector_geo() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-geo", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_geo_project("geo-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-geo",
            "geo-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector geo run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-geo");
    println!("pgvector geo recall={:.3}", recall);
    assert!(recall >= 0.9, "pgvector geo recall {:.3} < 0.9", recall);
}

/// Multi-condition OR (`color == "red" OR size >= 90`) — verifies pgvector unions
/// two clauses with SQL `OR` (searching the union), not `AND`.
#[test]
fn test_binary_pgvector_or_filter() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-or", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_or_filter_project("or-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-or",
            "or-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector or run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-or");
    println!("pgvector or-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector or-filter recall {:.3} < 0.9",
        recall
    );
}

/// Nested/grouped boolean filter `(color==red AND size>=50) OR (color==blue AND
/// size<10)` — verifies pgvector recurses into each nested AND-group and emits a
/// parenthesised SQL sub-expression per group, instead of mis-flattening the tree
/// into a single flat disjunction (which selects a wildly different doc set and
/// collapses recall).
#[test]
fn test_binary_pgvector_nested_filter() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-nested", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_nested_filter_project(
        "nested-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-nested",
            "nested-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector nested run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-nested");
    println!("pgvector nested-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector nested-filter recall {:.3} < 0.9",
        recall
    );
}

/// Multi-condition AND (keyword match AND numeric range) — verifies pgvector
/// composes two clauses into one SQL WHERE (`"color" = $1 AND "size" >= $2`).
#[test]
fn test_binary_pgvector_and_filter() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-and", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_and_filter_project(
        "and-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-and",
            "and-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector and-filter run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-and");
    println!("pgvector and-filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector and-filter recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end `match_any` on a MULTI-VALUED keyword field (`labels`, #88). The
/// ';'-joined TEXT column is filtered with array-overlap (`match_any`) and set
/// membership (exact-match); this exercises the full binary path against a live
/// Postgres, complementing the direct-SQL overlap test.
#[test]
fn test_binary_pgvector_match_any_labels() {
    wait_for_postgres();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-mal", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj = common::write_match_any_labels_project(
        "match-any-labels-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
        common::GtMetric::L2,
    );
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    assert!(
        common::run_binary(
            &proj.root,
            "pg-mal",
            "match-any-labels-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")],
        ),
        "pgvector match_any labels run failed"
    );

    let recall = common::read_recall(&proj.root, "pg-mal");
    println!("pgvector match_any labels recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector multi-valued labels match_any recall {:.3} < 0.9",
        recall
    );
}

/// Direct-SQL correctness for the keyword contains-any clause the `match_any`
/// filter builder emits. Multi-valued keyword payloads are stored as a
/// ';'-joined TEXT scalar, so the array-overlap must match a row that contains
/// ANY listed value — precisely the case a plain scalar `IN` (the pre-fix
/// behaviour) would silently drop. This runs the exact clause shape asserted by
/// the `match_any_string_list_emits_array_overlap` unit test against Postgres,
/// so together they cover the fix end-to-end without HNSW-recall flakiness.
#[test]
fn test_pgvector_match_any_overlap_matches_multivalue() {
    wait_for_postgres();
    let mut conn = connect();
    conn.execute("DROP TABLE IF EXISTS ma_overlap_test", &[])
        .unwrap();
    conn.execute(
        "CREATE TABLE ma_overlap_test (id INT PRIMARY KEY, color TEXT)",
        &[],
    )
    .unwrap();
    // id 1: multi-valued 'red;green' (matches {red,yellow} via 'red');
    // id 2: 'yellow' (matches); id 3: 'blue' (no); id 4: 'green' (no).
    conn.execute(
        "INSERT INTO ma_overlap_test VALUES (1,'red;green'),(2,'yellow'),(3,'blue'),(4,'green')",
        &[],
    )
    .unwrap();

    // Exactly the clause build_pg_clause emits for match_any ["red","yellow"].
    let overlap: Vec<i32> = conn
        .query(
            "SELECT id FROM ma_overlap_test \
             WHERE string_to_array(\"color\", ';') && ARRAY['red', 'yellow']::text[] \
             ORDER BY id",
            &[],
        )
        .unwrap()
        .iter()
        .map(|r| r.get::<_, i32>(0))
        .collect();
    assert_eq!(
        overlap,
        vec![1, 2],
        "array-overlap must match the multi-valued 'red;green' (via red) and 'yellow'"
    );

    // Sanity: the pre-fix scalar `IN` misses the multi-valued row, so this data
    // genuinely distinguishes the correct implementation from a regression.
    let scalar_in: Vec<i32> = conn
        .query(
            "SELECT id FROM ma_overlap_test WHERE \"color\" IN ('red','yellow') ORDER BY id",
            &[],
        )
        .unwrap()
        .iter()
        .map(|r| r.get::<_, i32>(0))
        .collect();
    assert_eq!(
        scalar_in,
        vec![2],
        "scalar IN drops the multi-valued row (confirms the test discriminates)"
    );

    conn.execute("DROP TABLE IF EXISTS ma_overlap_test", &[])
        .unwrap();
}

/// Direct-SQL correctness for the EXACT-MATCH clause on a multi-valued keyword
/// field (#88). `build_pg_clause` emits `$1 = ANY(string_to_array(col, ';'))`
/// for a `match.value` on `labels`; this must match a row whose ';'-joined set
/// CONTAINS the value — the case a scalar `col = $1` (the pre-fix behaviour)
/// silently drops. Complements the unit test that pins the SQL shape.
#[test]
fn test_pgvector_exact_match_labels_set_membership() {
    wait_for_postgres();
    let mut conn = connect();
    conn.execute("DROP TABLE IF EXISTS ma_exact_test", &[])
        .unwrap();
    conn.execute(
        "CREATE TABLE ma_exact_test (id INT PRIMARY KEY, labels TEXT)",
        &[],
    )
    .unwrap();
    // id 1: multi-valued 'red;green' (contains 'red'); id 2: 'red' (scalar);
    // id 3: 'blue' (no).
    conn.execute(
        "INSERT INTO ma_exact_test VALUES (1,'red;green'),(2,'red'),(3,'blue')",
        &[],
    )
    .unwrap();

    // Exactly the clause build_pg_clause emits for match.value "red" on labels.
    let member: Vec<i32> = conn
        .query(
            "SELECT id FROM ma_exact_test \
             WHERE $1 = ANY(string_to_array(\"labels\", ';')) ORDER BY id",
            &[&"red"],
        )
        .unwrap()
        .iter()
        .map(|r| r.get::<_, i32>(0))
        .collect();
    assert_eq!(
        member,
        vec![1, 2],
        "set-membership must match the multi-valued 'red;green' (via red) and scalar 'red'"
    );

    // Sanity: the pre-fix scalar `=` misses the multi-valued row.
    let scalar_eq: Vec<i32> = conn
        .query(
            "SELECT id FROM ma_exact_test WHERE \"labels\" = $1 ORDER BY id",
            &[&"red"],
        )
        .unwrap()
        .iter()
        .map(|r| r.get::<_, i32>(0))
        .collect();
    assert_eq!(
        scalar_eq,
        vec![2],
        "scalar = drops the multi-valued row (confirms the test discriminates)"
    );

    conn.execute("DROP TABLE IF EXISTS ma_exact_test", &[])
        .unwrap();
}

// ---------------------------------------------------------------------------
// HDF5 end-to-end: inner product, mixed workload, SVS Vamana
// ---------------------------------------------------------------------------

/// SVS tests need a server with the `svs` extension on PG_PORT (the stock
/// pgvector image lacks it), so they run only with PGVECTOR_SVS=1.
fn svs_enabled(test: &str) -> bool {
    let enabled = std::env::var("PGVECTOR_SVS").is_ok_and(|v| v == "1");
    if !enabled {
        println!(
            "skipping {}: set PGVECTOR_SVS=1 with an svs-enabled server",
            test
        );
    }
    enabled
}

fn brute_force_neighbors_ip(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let mut sims: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dot: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            (i as i64, dot)
        })
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sims.iter().take(top).map(|(id, _)| *id).collect()
}

fn brute_force(distance: &str, query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    match distance {
        "l2" => brute_force_neighbors_l2(query, vectors, top),
        "cosine" => brute_force_neighbors_cosine(query, vectors, top),
        "dot" => brute_force_neighbors_ip(query, vectors, top),
        other => panic!("unsupported distance {}", other),
    }
}

/// HDF5 project with `train`/`insert`/`test` plus ground truth against `train`
/// (`neighbors`) and `train`+`insert` (`allneighbors`), for `engine_configs`.
fn insert_project(
    dataset_name: &str,
    engine_configs: &serde_json::Value,
    distance: &str,
    train_count: usize,
    insert_count: usize,
    top: usize,
) -> std::path::PathBuf {
    let dim = 8;
    let num_queries = 10;
    let (_, train) = generate_test_vectors(train_count, dim);
    let (_, insert) = generate_test_vectors(insert_count, dim);
    let (_, queries) = generate_test_vectors(num_queries, dim);
    let merged: Vec<Vec<f32>> = train.iter().chain(insert.iter()).cloned().collect();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force(distance, q, &train, top))
        .collect();
    let all_neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force(distance, q, &merged, top))
        .collect();
    common::write_insert_hdf5_project(
        dataset_name,
        &serde_json::to_string(engine_configs).unwrap(),
        &train,
        &insert,
        &queries,
        &neighbors,
        &all_neighbors,
        distance,
        dim,
    )
}

/// Run the binary and return its output (stdout is asserted on).
fn run_pg(
    root: &std::path::Path,
    engine: &str,
    dataset: &str,
    extra: &[&str],
) -> std::process::Output {
    let out = std::process::Command::new(common::binary_path())
        .args([
            "--engines",
            engine,
            "--datasets",
            dataset,
            "--host",
            PG_HOST,
            "--skip-if-exists",
            "false",
        ])
        .args(extra)
        .env("PGVECTOR_PORT", PG_PORT.to_string())
        .current_dir(root)
        .output()
        .expect("run vector-db-benchmark");
    assert!(
        out.status.success(),
        "{} run failed.\nstdout:\n{}\nstderr:\n{}",
        engine,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// `results` object of the engine's (single) search result JSON.
fn read_results(root: &std::path::Path, engine: &str) -> serde_json::Value {
    let pattern = glob::Pattern::new(&format!("{}-*-search-*.json", engine)).unwrap();
    let path = std::fs::read_dir(root.join("results"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| pattern.matches(&p.file_name().unwrap().to_string_lossy()))
        .unwrap_or_else(|| panic!("no search result for {}", engine));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    v["results"].clone()
}

/// Inner-product datasets map to `<#>` / vector_ip_ops (previously rejected).
#[test]
fn test_binary_pgvector_dot() {
    wait_for_postgres();
    let configs = serde_json::json!([{
        "name": "pg-dot", "engine": "pgvector",
        "search_params": [{"parallel": 1, "top": 5, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let root = insert_project("dot-test", &configs, "dot", 300, 1, 5);
    run_pg(&root, "pg-dot", "dot-test", &[]);
    let recall = read_results(&root, "pg-dot")["mean_recall"]
        .as_f64()
        .unwrap();
    assert!(recall >= 0.9, "pgvector dot recall {:.3} < 0.9", recall);
}

/// HNSW on a halfvec column, indexed with inner product over a cosine dataset
/// (normalized on read), built after the load with build settings applied.
#[test]
fn test_binary_pgvector_halfvec_ip_after_load() {
    wait_for_postgres();
    let configs = serde_json::json!([{
        "name": "pg-half-ip", "engine": "pgvector",
        "collection_params": {
            "data_type": "FLOAT16",
            "distance": "dot",
            "index_after_load": true,
            "maintenance_work_mem": "64MB",
            "max_parallel_workers": 2
        },
        "search_params": [{"parallel": 1, "top": 5, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let root = insert_project("half-ip-test", &configs, "cosine", 300, 1, 5);
    let out = run_pg(&root, "pg-half-ip", "half-ip-test", &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("hnsw(embedding halfvec_ip_ops)"),
        "expected a halfvec IP index built after load.\nstdout:\n{}",
        stdout
    );
    let recall = read_results(&root, "pg-half-ip")["mean_recall"]
        .as_f64()
        .unwrap();
    assert!(
        recall >= 0.9,
        "pgvector halfvec IP recall {:.3} < 0.9",
        recall
    );
}

/// `--search-duration` recycles queries until the deadline.
#[test]
fn test_binary_pgvector_search_duration() {
    wait_for_postgres();
    let configs = serde_json::json!([{
        "name": "pg-duration", "engine": "pgvector",
        "search_params": [{"parallel": 2, "top": 5, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let root = insert_project("duration-test", &configs, "l2", 300, 1, 5);
    run_pg(
        &root,
        "pg-duration",
        "duration-test",
        &["--search-duration", "2"],
    );
    let results = read_results(&root, "pg-duration");
    // 10 queries, recycled for 2 s: far more searches than queries.
    let searched = results["requested_queries"].as_u64().unwrap();
    assert!(
        searched > 100,
        "expected recycled queries, got {}",
        searched
    );
    assert!(results["total_time"].as_f64().unwrap() >= 2.0);
    let recall = results["mean_recall"].as_f64().unwrap();
    assert!(
        recall >= 0.9,
        "pgvector duration recall {:.3} < 0.9",
        recall
    );
}

/// Mixed workload re-upserting corpus vectors (`--update-search-ratio` alone).
#[test]
fn test_binary_pgvector_mixed_updates() {
    wait_for_postgres();
    let configs = serde_json::json!([{
        "name": "pg-mixed", "engine": "pgvector",
        "search_params": [{"parallel": 2, "top": 5, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let root = insert_project("mixed-test", &configs, "l2", 300, 1, 5);
    run_pg(
        &root,
        "pg-mixed",
        "mixed-test",
        &["--update-search-ratio", "1:2"],
    );
    let results = read_results(&root, "pg-mixed");
    assert!(
        results["update_count"].as_u64().unwrap_or(0) > 0,
        "expected updates in {}",
        results
    );
    assert_eq!(results["update_search_ratio"], "1:2");
    let recall = results["mean_recall"].as_f64().unwrap();
    assert!(recall >= 0.9, "pgvector mixed recall {:.3} < 0.9", recall);
}

/// Shared `--insert` check: every insert vector is upserted exactly once and
/// the post-insert pass is scored against `allneighbors`.
fn assert_insert_post_recall(
    engine: &str,
    collection_params: serde_json::Value,
    search: serde_json::Value,
    train_count: usize,
) {
    let insert_count = 50;
    let configs = serde_json::json!([{
        "name": engine, "engine": "pgvector",
        "collection_params": collection_params,
        "search_params": [search],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let dataset = format!("{}-ds", engine);
    let root = insert_project(&dataset, &configs, "l2", train_count, insert_count, 5);
    let out = run_pg(
        &root,
        engine,
        &dataset,
        &["--update-search-ratio", "1:1", "--insert"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Post-insert recall (allneighbors)"),
        "post-insert recall phase did not run.\nstdout:\n{}",
        stdout
    );
    let results = read_results(&root, engine);
    assert_eq!(results["update_count"].as_u64(), Some(insert_count as u64));
    let recall = results["post_insert_mean_recall"].as_f64().unwrap();
    assert!(
        recall >= 0.9,
        "{} post-insert recall {:.3} < 0.9",
        engine,
        recall
    );
    assert!(results["post_insert_recall_p10"].is_number());
}

#[test]
fn test_binary_pgvector_insert_post_recall() {
    wait_for_postgres();
    assert_insert_post_recall(
        "pg-insert",
        serde_json::json!({}),
        serde_json::json!({"parallel": 1, "top": 5, "search_params": {"hnsw_ef": 400}}),
        300,
    );
}

/// SVS Vamana build + search for each metric (cosine also indexed as L2 and IP,
/// and as halfvec), with the window calibrated to a target precision.
#[test]
fn test_binary_pgvector_svs_search_calibration() {
    if !svs_enabled("test_binary_pgvector_svs_search_calibration") {
        return;
    }
    wait_for_postgres();
    // (dataset distance, indexed distance override, data_type)
    for (distance, index_distance, data_type) in [
        ("l2", None, "FLOAT32"),
        ("cosine", None, "FLOAT32"),
        ("cosine", Some("l2"), "FLOAT32"),
        ("cosine", Some("dot"), "FLOAT32"),
        ("cosine", Some("dot"), "FLOAT16"),
        ("dot", None, "FLOAT32"),
    ] {
        let engine = format!(
            "pg-svs-{}-{}-{}",
            distance,
            index_distance.unwrap_or("same"),
            data_type
        );
        let mut collection_params = serde_json::json!({
            "algorithm": "svs-vamana",
            "data_type": data_type,
            "svs-vamana_config": {"graph_degree": 32, "build_window_size": 64}
        });
        if let Some(d) = index_distance {
            collection_params["distance"] = d.into();
        }
        let configs = serde_json::json!([{
            "name": engine, "engine": "pgvector",
            "collection_params": collection_params,
            "search_params": [{
                "parallel": 1, "top": 10,
                "calibration_param": "search_window_size",
                "calibration_precision": 0.95
            }],
            "upload_params": {"parallel": 1, "batch_size": 100}
        }]);
        let dataset = format!("{}-ds", engine);
        let root = insert_project(&dataset, &configs, distance, 500, 1, 10);
        let out = run_pg(&root, &engine, &dataset, &[]);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("Calibrated search_window_size="),
            "{}: calibration did not converge.\nstdout:\n{}",
            engine,
            stdout
        );
        let recall = read_results(&root, &engine)["mean_recall"]
            .as_f64()
            .unwrap();
        assert!(recall >= 0.9, "{} recall {:.3} < 0.9", engine, recall);
    }
}

/// Inserts into a built Vamana index; fails loudly if the extension rejects writes.
#[test]
fn test_binary_pgvector_svs_insert_post_recall() {
    if !svs_enabled("test_binary_pgvector_svs_insert_post_recall") {
        return;
    }
    wait_for_postgres();
    assert_insert_post_recall(
        "pg-svs-insert",
        serde_json::json!({
            "algorithm": "svs-vamana",
            "svs-vamana_config": {"graph_degree": 32, "build_window_size": 64}
        }),
        serde_json::json!({"parallel": 1, "top": 5, "search_params": {"search_window_size": 100}}),
        300,
    );
}

/// LVQ and LeanVec builds plus inserts into the compressed index. Both train
/// their compression on the rows present at build time (LVQ wants >= 10,000).
#[test]
fn test_binary_pgvector_svs_compression_insert() {
    if !svs_enabled("test_binary_pgvector_svs_compression_insert") {
        return;
    }
    wait_for_postgres();
    // (compression_type, compression_primary, compression_secondary):
    // LVQ 8-bit without residual, LVQ 4x8, LeanVec 8x8.
    for (name, ctype, primary, secondary) in [
        ("lvq8", 2, 8, 0),
        ("lvq4x8", 2, 4, 8),
        ("leanvec8x8", 1, 8, 8),
    ] {
        assert_insert_post_recall(
            &format!("pg-svs-{}", name),
            serde_json::json!({
                "algorithm": "svs-vamana",
                "svs-vamana_config": {
                    "graph_degree": 32,
                    "build_window_size": 64,
                    "compression_type": ctype,
                    "compression_primary": primary,
                    "compression_secondary": secondary,
                    // The fixture is isotropic random data, so LeanVec's default
                    // D/2 projection discards real signal. Ignored under LVQ.
                    "leanvec_dims": 6
                }
            }),
            serde_json::json!({"parallel": 1, "top": 5, "search_params": {"search_window_size": 200}}),
            10_000,
        );
    }
}

/// 100 concurrent writers exceed svs's 64 pending-insert slots per database;
/// rejected writes are retried and counted rather than failing the run.
#[test]
fn test_binary_pgvector_svs_concurrent_inserts_retry() {
    if !svs_enabled("test_binary_pgvector_svs_concurrent_inserts_retry") {
        return;
    }
    wait_for_postgres();
    let insert_count = 2000;
    let configs = serde_json::json!([{
        "name": "pg-svs-concurrent", "engine": "pgvector",
        "collection_params": {
            "algorithm": "svs-vamana",
            "svs-vamana_config": {"graph_degree": 32, "build_window_size": 64}
        },
        "search_params": [{"parallel": 100, "top": 5, "search_params": {"search_window_size": 100}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let root = insert_project("svs-concurrent-ds", &configs, "l2", 2000, insert_count, 5);
    run_pg(
        &root,
        "pg-svs-concurrent",
        "svs-concurrent-ds",
        &["--update-search-ratio", "1:1", "--insert"],
    );
    let results = read_results(&root, "pg-svs-concurrent");
    assert_eq!(results["update_count"].as_u64(), Some(insert_count as u64));
    println!("update_retries={}", results["update_retries"]);
    assert!(results["update_retries"].is_u64());
}
