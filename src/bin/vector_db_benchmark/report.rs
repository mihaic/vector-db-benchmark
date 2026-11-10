//! Benchmark report: a Markdown comparison of engines built from the per-run
//! result files in results/ (`*-search-*.json` and `*-upload-*.json`),
//! filtered by --engines and --datasets.
//!
//! One section per dataset, one table per workload (search-only, each mixed
//! update:search ratio, each `--insert` ratio), one row per engine: its newest
//! run of that workload, joined with the upload of the same invocation.
//! Engines with no run of a workload are listed under its table, and a run
//! whose calibration did not set its knob is marked.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use chrono::NaiveDateTime;
use serde_json::Value;

use crate::cli::Args;
use crate::config::{matches_pattern, project_root, read_engine_configs};

/// Result-file timestamp suffix, as written by `save_*_results`.
const TS_FORMAT: &str = "%Y-%m-%d-%H-%M-%S";
const TS_LEN: usize = 19;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Workload {
    Search,
    /// Re-upserts of existing points; duration in seconds when time-bounded.
    Mixed {
        ratio: String,
        duration: Option<u64>,
    },
    /// `--insert`: new points until the insert set is exhausted.
    Insert {
        ratio: String,
    },
}

impl Workload {
    fn title(&self) -> String {
        match self {
            Workload::Search => "Search only".to_string(),
            Workload::Mixed {
                ratio,
                duration: Some(d),
            } => format!("Mixed {} (updates:searches), {} s", ratio, d),
            Workload::Mixed {
                ratio,
                duration: None,
            } => format!("Mixed {} (updates:searches)", ratio),
            Workload::Insert { ratio } => {
                format!("Insert {} (inserts:searches) until all inserted", ratio)
            }
        }
    }

    /// Search first, then mixed by ratio, then inserts by ratio.
    fn sort_key(&self) -> (u8, f64, u64) {
        match self {
            Workload::Search => (0, 0.0, 0),
            Workload::Mixed { ratio, duration } => (
                1,
                crate::summary::ratio_sort_key(ratio),
                duration.unwrap_or(0),
            ),
            Workload::Insert { ratio } => (2, crate::summary::ratio_sort_key(ratio), 0),
        }
    }
}

/// One result file.
#[derive(Debug, Clone)]
struct RunFile {
    file: String,
    experiment: String,
    dataset: String,
    timestamp: NaiveDateTime,
    pid: Option<u64>,
    json: Value,
}

/// Entry point for `--report`.
pub fn export_report(args: &Args) -> Result<(), String> {
    let out_path = args
        .report
        .as_ref()
        .ok_or_else(|| "internal: --report not set".to_string())?;
    let results_dir = project_root().join("results");
    let (searches, uploads) = read_results(&results_dir, args)?;
    if searches.is_empty() {
        return Err(format!(
            "No search results in {} matched engines={:?} datasets={:?}.",
            results_dir.display(),
            args.engines,
            args.datasets
        ));
    }
    // Result files from before `params.calibration_param` existed: take the
    // calibrated knob from the engine's current config.
    let calibrated: HashMap<String, String> = read_engine_configs(args.engines_file.as_deref())?
        .into_iter()
        .filter_map(|(name, cfg)| {
            let param = cfg
                .search_params
                .iter()
                .flatten()
                .find_map(|p| p.calibration_param.clone())?;
            Some((name, param))
        })
        .collect();
    let report = render_report(&searches, &uploads, &args.engines, &calibrated);
    if out_path == "-" {
        print!("{}", report);
    } else {
        fs::write(out_path, &report).map_err(|e| format!("failed to write {}: {}", out_path, e))?;
        println!("Wrote report: {}", out_path);
    }
    Ok(())
}

/// Read every search and upload result file matching the engine and dataset
/// patterns. Unparseable files are skipped.
fn read_results(dir: &Path, args: &Args) -> Result<(Vec<RunFile>, Vec<RunFile>), String> {
    let pattern = dir.join("*.json").to_string_lossy().to_string();
    let mut searches = Vec::new();
    let mut uploads = Vec::new();
    for path in glob::glob(&pattern)
        .map_err(|e| format!("bad results glob: {}", e))?
        .filter_map(Result::ok)
    {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let Some(json) = fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        else {
            continue;
        };
        let Some(run) = parse_run_file(&name, json) else {
            continue;
        };
        if !args
            .engines
            .iter()
            .any(|p| matches_pattern(&run.experiment, p))
            || !args
                .datasets
                .iter()
                .any(|p| matches_pattern(&run.dataset, p))
        {
            continue;
        }
        if name.contains("-search-") {
            searches.push(run);
        } else if name.contains("-upload-") {
            uploads.push(run);
        }
    }
    Ok((searches, uploads))
}

/// Parse a result file's identity: experiment and dataset from `params`, the
/// timestamp from the file name, and the writing process's pid (from `params`,
/// else the search file name `...-<pid>-<timestamp>.json`).
fn parse_run_file(name: &str, json: Value) -> Option<RunFile> {
    let stem = name.strip_suffix(".json")?;
    if stem.len() < TS_LEN + 1 || !stem.is_char_boundary(stem.len() - TS_LEN) {
        return None;
    }
    let (head, ts) = stem.split_at(stem.len() - TS_LEN);
    let timestamp = NaiveDateTime::parse_from_str(ts, TS_FORMAT).ok()?;
    let params = json.get("params")?;
    let experiment = params.get("experiment")?.as_str()?.to_string();
    let dataset = params.get("dataset")?.as_str()?.to_string();
    let pid = params.get("pid").and_then(|v| v.as_u64()).or_else(|| {
        if !name.contains("-search-") {
            return None;
        }
        head.strip_suffix('-')?.rsplit('-').next()?.parse().ok()
    });
    Some(RunFile {
        file: name.to_string(),
        experiment,
        dataset,
        timestamp,
        pid,
        json,
    })
}

fn workload(run: &RunFile) -> Workload {
    let params = &run.json["params"];
    let results = &run.json["results"];
    let Some(ratio) = results["update_search_ratio"].as_str() else {
        return Workload::Search;
    };
    let ratio = ratio.to_string();
    // Files from before `params.insert` existed: only --insert runs score
    // post-insert recall (when the dataset has `allneighbors`).
    let insert = params["insert"]
        .as_bool()
        .unwrap_or_else(|| !results["post_insert_mean_recall"].is_null());
    if insert {
        Workload::Insert { ratio }
    } else {
        Workload::Mixed {
            ratio,
            duration: params["duration_seconds"]
                .as_f64()
                .map(|d| d.round() as u64),
        }
    }
}

/// The upload of the invocation that produced `search`: the one with the same
/// pid, else (older files without an upload pid) the latest upload before the
/// search with no other invocation's search of this engine in between.
fn matching_upload<'a>(
    search: &RunFile,
    searches: &[RunFile],
    uploads: &'a [RunFile],
) -> Option<&'a RunFile> {
    let same_engine =
        |r: &&RunFile| r.experiment == search.experiment && r.dataset == search.dataset;
    if let Some(pid) = search.pid {
        if let Some(u) = uploads
            .iter()
            .filter(same_engine)
            .find(|u| u.pid == Some(pid))
        {
            return Some(u);
        }
    }
    let upload = uploads
        .iter()
        .filter(same_engine)
        .filter(|u| u.pid.is_none() && u.timestamp <= search.timestamp)
        .max_by_key(|u| u.timestamp)?;
    let interleaved = searches.iter().filter(same_engine).any(|s| {
        s.pid != search.pid && s.timestamp >= upload.timestamp && s.timestamp < search.timestamp
    });
    (!interleaved).then_some(upload)
}

/// `calibrated` maps an engine name to its calibrated knob, for result files
/// that don't record it.
fn render_report(
    searches: &[RunFile],
    uploads: &[RunFile],
    engine_patterns: &[String],
    calibrated: &HashMap<String, String>,
) -> String {
    // dataset -> workload -> engine -> runs (oldest first)
    let mut grouped: BTreeMap<&str, BTreeMap<Workload, BTreeMap<&str, Vec<&RunFile>>>> =
        BTreeMap::new();
    for run in searches {
        grouped
            .entry(run.dataset.as_str())
            .or_default()
            .entry(workload(run))
            .or_default()
            .entry(run.experiment.as_str())
            .or_default()
            .push(run);
    }

    let mut out = String::from("# Benchmark report\n");
    let mut sources: Vec<String> = Vec::new();
    let mut older_runs = 0;
    let mut saturated = false;
    let mut uncalibrated = false;
    for (dataset, workloads) in &grouped {
        out.push_str(&format!("\n## {}\n", dataset));
        let dataset_engines: BTreeSet<&str> = workloads
            .values()
            .flat_map(|engines| engines.keys().copied())
            .collect();
        let mut workloads: Vec<_> = workloads.iter().collect();
        workloads.sort_by(|a, b| a.0.sort_key().partial_cmp(&b.0.sort_key()).unwrap());
        for (workload, engines) in workloads {
            let mut engines: Vec<_> = engines.iter().collect();
            // Order rows as the --engines patterns list them, then by name.
            engines.sort_by_key(|(name, _)| {
                let rank = engine_patterns
                    .iter()
                    .position(|p| matches_pattern(name, p))
                    .unwrap_or(usize::MAX);
                (rank, name.to_string())
            });
            let mut rows = Vec::new();
            for (engine, runs) in &engines {
                let mut runs = runs.to_vec();
                runs.sort_by_key(|r| r.timestamp);
                older_runs += runs.len() - 1;
                let run = runs[runs.len() - 1];
                let upload = matching_upload(run, searches, uploads);
                sources.push(match upload {
                    Some(u) => format!("{} (upload: {})", run.file, u.file),
                    None => run.file.clone(),
                });
                saturated |= run.json["results"]["client_saturated"].as_bool() == Some(true);
                let mut row = row(engine, workload, run, upload);
                if calibration_failed(run, calibrated) {
                    row[1].push_str(" †");
                    uncalibrated = true;
                }
                rows.push(row);
            }
            out.push_str(&format!("\n### {}\n\n", workload.title()));
            out.push_str(&markdown_table(&headers(workload), &rows));
            let missing: Vec<String> = dataset_engines
                .iter()
                .filter(|e| !engines.iter().any(|(name, _)| **name == **e))
                .map(|e| format!("`{}`", e))
                .collect();
            if !missing.is_empty() {
                out.push_str(&format!("\nNo result: {}.\n", missing.join(", ")));
            }
        }
    }

    out.push_str("\n## Notes\n\n");
    out.push_str(
        "- Recall is mean recall@top against the dataset's `neighbors` ground truth; \
         post-insert recall is one pass after all inserts, against `allneighbors`.\n\
         - Latencies are per query (or per update), in milliseconds; p99.9 comes from \
         the run's HDR latency histogram. Upload is the COPY load plus any index \
         build after it, in seconds.\n\
         - Each row is the newest run of that engine and workload.",
    );
    if older_runs > 0 {
        out.push_str(&format!(" {} older run(s) are not shown.", older_runs));
    }
    out.push('\n');
    if saturated {
        out.push_str(
            "- `*` marks runs whose client was saturated: their throughput and \
             latency reflect the client, not the server.\n",
        );
    }
    if uncalibrated {
        out.push_str(
            "- `†` marks runs whose calibration failed: they ran with the engine's \
             default search knob, not one tuned to the target precision.\n",
        );
    }
    out.push_str("\nSources:\n\n");
    for s in sources {
        out.push_str(&format!("- `{}`\n", s));
    }
    out
}

/// The engine calibrates a search knob, but the run's `search_params` don't
/// set it: calibration failed and the run fell back to the default.
fn calibration_failed(run: &RunFile, calibrated: &HashMap<String, String>) -> bool {
    let params = &run.json["params"];
    let Some(param) = params["calibration_param"]
        .as_str()
        .or_else(|| calibrated.get(&run.experiment).map(String::as_str))
    else {
        return false;
    };
    params["search_params"][param].is_null()
}

fn headers(workload: &Workload) -> Vec<&'static str> {
    let mut h = vec![
        "Engine",
        "Search knob",
        "Recall",
        "Recall p10",
        "QPS",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "p99.9 ms",
    ];
    if !matches!(workload, Workload::Search) {
        h.extend([
            "Updates/s",
            "Update p50 ms",
            "Update p99 ms",
            "Update p99.9 ms",
            "Update retries",
        ]);
    }
    if matches!(workload, Workload::Insert { .. }) {
        h.extend(["Post-insert recall", "Post-insert recall p10"]);
    }
    h.push("Upload s");
    h
}

fn row(engine: &str, workload: &Workload, run: &RunFile, upload: Option<&RunFile>) -> Vec<String> {
    let r = &run.json["results"];
    let saturated = r["client_saturated"].as_bool() == Some(true);
    let failed = r["failed_queries"].as_u64().unwrap_or(0);
    let mut qps = fmt_num(&r["rps"], 1);
    if saturated {
        qps.push_str(" *");
    }
    if failed > 0 {
        qps.push_str(&format!(" ({} failed)", failed));
    }
    let mut cells = vec![
        format!("`{}`", engine),
        search_knob(&run.json["params"]["search_params"]),
        fmt_num(&r["mean_recall"], 4),
        fmt_num(&r["recall_p10"], 2),
        qps,
        fmt_ms(&r["p50_time"]),
        fmt_ms(&r["p95_time"]),
        fmt_ms(&r["p99_time"]),
        fmt_num(&r["latency_hdr"]["p999_ms"], 0),
    ];
    if !matches!(workload, Workload::Search) {
        cells.extend([
            fmt_num(&r["update_rps"], 1),
            fmt_ms(&r["update_p50_time"]),
            fmt_ms(&r["update_p99_time"]),
            fmt_num(&r["update_latency_hdr"]["p999_ms"], 0),
            fmt_num(&r["update_retries"], 0),
        ]);
    }
    if matches!(workload, Workload::Insert { .. }) {
        cells.extend([
            fmt_num(&r["post_insert_mean_recall"], 4),
            fmt_num(&r["post_insert_recall_p10"], 2),
        ]);
    }
    cells.push(
        upload
            .map(|u| fmt_num(&u.json["results"]["upload_time"], 0))
            .unwrap_or_else(|| "–".to_string()),
    );
    cells
}

/// The search parameters that were set, e.g. `ef=122`, skipping nulls and the
/// vector data type (a Redis encoding detail).
fn search_knob(search_params: &Value) -> String {
    let Some(map) = search_params.as_object() else {
        return "–".to_string();
    };
    let knobs: Vec<String> = map
        .iter()
        .filter(|(k, v)| !v.is_null() && k.as_str() != "data_type")
        .map(|(k, v)| match v {
            Value::String(s) => format!("{}={}", k, s),
            other => format!("{}={}", k, other),
        })
        .collect();
    if knobs.is_empty() {
        "default".to_string()
    } else {
        knobs.join(", ")
    }
}

fn fmt_num(v: &Value, decimals: usize) -> String {
    v.as_f64()
        .filter(|x| x.is_finite())
        .map(|x| format!("{:.*}", decimals, x))
        .unwrap_or_else(|| "–".to_string())
}

/// Seconds to whole milliseconds (finer digits are run-to-run noise).
fn fmt_ms(v: &Value) -> String {
    v.as_f64()
        .filter(|x| x.is_finite())
        .map(|x| format!("{:.0}", x * 1000.0))
        .unwrap_or_else(|| "–".to_string())
}

fn markdown_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut out = format!("| {} |\n", headers.join(" | "));
    out.push_str(&format!(
        "|{}|\n",
        headers
            .iter()
            .enumerate()
            .map(|(i, _)| if i == 0 { "---" } else { "---:" })
            .collect::<Vec<_>>()
            .join("|")
    ));
    for row in rows {
        out.push_str(&format!("| {} |\n", row.join(" | ")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn search_file(experiment: &str, ts: &str, pid: u64, params: Value, results: Value) -> RunFile {
        let mut p = json!({"experiment": experiment, "dataset": "ds"});
        p.as_object_mut()
            .unwrap()
            .extend(params.as_object().unwrap().clone());
        let name = format!("{}-ds-search-0-{}-{}.json", experiment, pid, ts);
        parse_run_file(&name, json!({"params": p, "results": results})).unwrap()
    }

    fn upload_file(experiment: &str, ts: &str, pid: Option<u64>, upload_time: f64) -> RunFile {
        let mut p = json!({"experiment": experiment, "dataset": "ds"});
        if let Some(pid) = pid {
            p["pid"] = pid.into();
        }
        let name = format!("{}-ds-upload-0-875000-{}.json", experiment, ts);
        parse_run_file(
            &name,
            json!({"params": p, "results": {"upload_time": upload_time}}),
        )
        .unwrap()
    }

    #[test]
    fn parses_timestamp_and_pid_from_file_names() {
        let run = parse_run_file(
            "e-ds-search-0-mixed-100x100-782925-2026-10-20-19-38-43.json",
            json!({"params": {"experiment": "e", "dataset": "ds"}, "results": {}}),
        )
        .unwrap();
        assert_eq!(run.pid, Some(782925));
        assert_eq!(
            run.timestamp,
            NaiveDateTime::parse_from_str("2026-10-20-19-38-43", TS_FORMAT).unwrap()
        );
        // Uploads have no pid in the name; `params.pid` wins when present.
        assert_eq!(upload_file("e", "2026-10-20-19-36-31", None, 1.0).pid, None);
        assert_eq!(
            upload_file("e", "2026-10-20-19-36-31", Some(7), 1.0).pid,
            Some(7)
        );
        assert!(parse_run_file("e-ds-summary.json", json!({})).is_none());
    }

    #[test]
    fn classifies_workloads() {
        let ts = "2026-10-20-19-38-43";
        let search = search_file("e", ts, 1, json!({}), json!({}));
        assert_eq!(workload(&search), Workload::Search);
        let mixed = search_file(
            "e",
            ts,
            1,
            json!({"duration_seconds": 55.0, "insert": false}),
            json!({"update_search_ratio": "100:100"}),
        );
        assert_eq!(
            workload(&mixed),
            Workload::Mixed {
                ratio: "100:100".into(),
                duration: Some(55)
            }
        );
        let insert = search_file(
            "e",
            ts,
            1,
            json!({"insert": true}),
            json!({"update_search_ratio": "100:100"}),
        );
        assert_eq!(
            workload(&insert),
            Workload::Insert {
                ratio: "100:100".into()
            }
        );
        // Older files without `params.insert`: post-insert recall marks inserts.
        let old_insert = search_file(
            "e",
            ts,
            1,
            json!({}),
            json!({"update_search_ratio": "100:100", "post_insert_mean_recall": 0.95}),
        );
        assert!(matches!(workload(&old_insert), Workload::Insert { .. }));
        assert!(Workload::Search.sort_key() < workload(&mixed).sort_key());
        assert!(workload(&mixed).sort_key() < workload(&insert).sort_key());
    }

    #[test]
    fn pairs_search_with_upload_of_same_invocation() {
        let u1 = upload_file("e", "2026-10-20-10-00-00", None, 100.0);
        let s1 = search_file("e", "2026-10-20-10-05-00", 11, json!({}), json!({}));
        // A later --skip-upload invocation must not inherit u1.
        let s2 = search_file("e", "2026-10-20-11-00-00", 22, json!({}), json!({}));
        let u3 = upload_file("e", "2026-10-20-12-00-00", Some(33), 300.0);
        let s3 = search_file(
            "e",
            "2026-10-20-12-10-00",
            33,
            json!({"pid": 33}),
            json!({}),
        );
        let searches = vec![s1.clone(), s2.clone(), s3.clone()];
        let uploads = vec![u1.clone(), u3.clone()];
        assert_eq!(
            matching_upload(&s1, &searches, &uploads).map(|u| &u.file),
            Some(&u1.file)
        );
        assert!(matching_upload(&s2, &searches, &uploads).is_none());
        assert_eq!(
            matching_upload(&s3, &searches, &uploads).map(|u| &u.file),
            Some(&u3.file)
        );
        // Several searches of one invocation share its upload.
        let s1b = search_file("e", "2026-10-20-10-06-00", 11, json!({}), json!({}));
        let searches = vec![s1.clone(), s1b.clone()];
        assert_eq!(
            matching_upload(&s1b, &searches, &uploads).map(|u| &u.file),
            Some(&u1.file)
        );
    }

    #[test]
    fn renders_tables_with_newest_run_per_engine() {
        let results = |rps: f64| {
            json!({
                "rps": rps, "mean_recall": 0.95, "recall_p10": 0.87,
                "p50_time": 0.0276, "p95_time": 0.035, "p99_time": 0.0426,
                "latency_hdr": {"p999_ms": 61.4},
                "client_saturated": false, "failed_queries": 0
            })
        };
        let old = search_file(
            "hnsw",
            "2026-10-20-10-00-00",
            1,
            json!({"search_params": {"ef": 100}}),
            results(1000.0),
        );
        let new = search_file(
            "hnsw",
            "2026-10-20-11-00-00",
            2,
            json!({"search_params": {"ef": 122}}),
            results(3630.9),
        );
        let mut insert_results = results(237.1);
        insert_results["update_search_ratio"] = "100:100".into();
        insert_results["update_rps"] = 215.6.into();
        insert_results["update_p50_time"] = 0.4037.into();
        insert_results["update_p99_time"] = 0.6406.into();
        insert_results["update_latency_hdr"] = json!({"p999_ms": 1977.3});
        insert_results["update_retries"] = 808479.into();
        insert_results["post_insert_mean_recall"] = 0.9497.into();
        insert_results["post_insert_recall_p10"] = 0.87.into();
        insert_results["client_saturated"] = true.into();
        let insert = search_file(
            "svs",
            "2026-10-20-12-00-00",
            3,
            json!({"insert": true, "search_params": {"ef": null, "search_window_size": 122}}),
            insert_results,
        );
        let upload = upload_file("hnsw", "2026-10-20-10-59-00", Some(2), 545.7);
        let report = render_report(
            &[old, new, insert],
            &[upload],
            &["svs".to_string(), "hnsw".to_string()],
            &HashMap::new(),
        );
        assert!(report.contains("## ds\n"));
        assert!(report
            .contains("| `hnsw` | ef=122 | 0.9500 | 0.87 | 3630.9 | 28 | 35 | 43 | 61 | 546 |"));
        assert!(!report.contains("1000.0"), "older run must not be shown");
        assert!(report.contains("1 older run(s) are not shown."));
        assert!(report.contains("### Insert 100:100 (inserts:searches) until all inserted"));
        assert!(report.contains(
            "| `svs` | search_window_size=122 | 0.9500 | 0.87 | 237.1 * | 28 | 35 | 43 | 61 \
             | 215.6 | 404 | 641 | 1977 | 808479 | 0.9497 | 0.87 | – |"
        ));
        assert!(report.contains("`*` marks runs whose client was saturated"));
        // Search-only table comes before the insert table.
        assert!(report.find("### Search only").unwrap() < report.find("### Insert").unwrap());
        // Each table lists the dataset's engines it has no run of.
        let insert_table = &report[report.find("### Insert").unwrap()..];
        assert!(insert_table.contains("No result: `hnsw`."));
        let search_table = &report[..report.find("### Insert").unwrap()];
        assert!(search_table.contains("No result: `svs`."));
    }

    #[test]
    fn marks_runs_whose_calibration_failed() {
        let results = json!({"rps": 1.0, "failed_queries": 0});
        let ts = "2026-10-20-10-00-00";
        // Recorded in the file: calibrated, then fell back.
        let ok = search_file(
            "a",
            ts,
            1,
            json!({"calibration_param": "search_window_size",
                   "search_params": {"ef": null, "search_window_size": 244}}),
            results.clone(),
        );
        let failed = search_file(
            "b",
            ts,
            2,
            json!({"calibration_param": "search_window_size", "search_params": null}),
            results.clone(),
        );
        // Older file: the knob comes from the engine config.
        let old_failed = search_file("c", ts, 3, json!({"search_params": null}), results.clone());
        let old_ef = search_file(
            "d",
            ts,
            4,
            json!({"search_params": {"ef": 122}}),
            results.clone(),
        );
        // Not calibrated at all.
        let fixed = search_file("e", ts, 5, json!({"search_params": null}), results);
        let calibrated = HashMap::from([
            ("c".to_string(), "search_window_size".to_string()),
            ("d".to_string(), "ef".to_string()),
        ]);
        assert!(!calibration_failed(&ok, &calibrated));
        assert!(calibration_failed(&failed, &calibrated));
        assert!(calibration_failed(&old_failed, &calibrated));
        assert!(!calibration_failed(&old_ef, &calibrated));
        assert!(!calibration_failed(&fixed, &calibrated));
        let report = render_report(&[ok, failed], &[], &["*".to_string()], &calibrated);
        assert!(report.contains("| `b` | – † |"));
        assert!(report.contains("`†` marks runs whose calibration failed"));
    }

    #[test]
    fn markdown_table_right_aligns_numbers() {
        let t = markdown_table(&["Engine", "QPS"], &[vec!["a".into(), "1.0".into()]]);
        assert_eq!(t, "| Engine | QPS |\n|---|---:|\n| a | 1.0 |\n");
    }

    #[test]
    fn search_knob_skips_nulls_and_data_type() {
        assert_eq!(
            search_knob(&json!({"ef": null, "search_window_size": 122, "data_type": "FLOAT16"})),
            "search_window_size=122"
        );
        assert_eq!(search_knob(&json!({"ef": null})), "default");
        assert_eq!(search_knob(&Value::Null), "–");
    }
}
