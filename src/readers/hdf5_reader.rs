//! HDF5 file reader for vector datasets.

use hdf5::File as Hdf5File;

/// Read a 2D float dataset by name, returning (vectors, dim). Shared by the
/// `train` and `insert` readers below.
fn read_hdf5_dataset_vectors(
    file: &Hdf5File,
    dataset_name: &str,
    normalize: bool,
) -> Result<(Vec<Vec<f32>>, usize), String> {
    let ds = file
        .dataset(dataset_name)
        .map_err(|e| format!("Failed to open '{}' dataset: {}", dataset_name, e))?;

    let shape = ds.shape();
    let dim = shape[1];

    // Read as flat array
    let flat: Vec<f32> = ds
        .read_raw()
        .map_err(|e| format!("Failed to read '{}' dataset: {}", dataset_name, e))?;

    // Convert to Vec<Vec<f32>> and optionally normalize
    let mut vectors: Vec<Vec<f32>> = flat.chunks(dim).map(|chunk| chunk.to_vec()).collect();

    if normalize {
        for vec in &mut vectors {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vec.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    Ok((vectors, dim))
}

/// Read vectors from HDF5 file, returning (ids, vectors).
/// If normalize is true, each vector is divided by its L2 norm.
pub fn read_hdf5_vectors(path: &str, normalize: bool) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
    let file = Hdf5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;
    let (vectors, _dim) = read_hdf5_dataset_vectors(&file, "train", normalize)?;

    // Generate sequential IDs (matching Python behavior)
    let ids: Vec<i64> = (0..vectors.len() as i64).collect();

    Ok((ids, vectors))
}

/// Read the `insert` dataset from an HDF5 file, returning (ids, vectors) for
/// vectors meant to be inserted as NEW points during a mixed benchmark
/// (rather than re-upserting existing `train` points). IDs continue after the
/// `train` dataset's ids (`train_count..train_count+insert_count`) so they
/// never collide with the initial corpus's keys.
pub fn read_hdf5_insert_vectors(
    path: &str,
    normalize: bool,
) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
    let file = Hdf5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;
    let train_count = file
        .dataset("train")
        .map_err(|e| format!("Failed to open 'train' dataset: {}", e))?
        .shape()[0];
    let (vectors, _dim) = read_hdf5_dataset_vectors(&file, "insert", normalize)?;

    let ids: Vec<i64> = (train_count as i64..(train_count + vectors.len()) as i64).collect();

    Ok((ids, vectors))
}

/// Read the `allneighbors` dataset from an HDF5 file: ground-truth nearest
/// neighbors for each query computed against the FULL post-insert corpus
/// (`train` + `insert`), unlike `neighbors` which is ground truth against
/// `train` alone. Used to measure recall once a mixed `--insert` benchmark has
/// finished ingesting every insert vector. Not every `--insert` dataset carries
/// this (it predates the feature for some), so callers should treat a missing
/// dataset as an optional/skippable condition rather than fatal.
pub fn read_hdf5_all_neighbors(path: &str) -> Result<Vec<Vec<i64>>, String> {
    let file = Hdf5File::open(path).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;
    let ds = file
        .dataset("allneighbors")
        .map_err(|e| format!("Failed to open 'allneighbors' dataset: {}", e))?;

    let shape = ds.shape();
    if shape.len() != 2 {
        return Err("Expected 2D allneighbors dataset".to_string());
    }
    let num_queries = shape[0];
    let k = shape[1];

    let flat: Vec<i64> = ds
        .read_raw()
        .map_err(|e| format!("Failed to read 'allneighbors' dataset: {}", e))?;

    Ok(flat
        .chunks(k)
        .take(num_queries)
        .map(|chunk| chunk.to_vec())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_hdf5(
        dir: &std::path::Path,
        name: &str,
        train: &[[f32; 2]],
        insert: &[[f32; 2]],
    ) -> std::path::PathBuf {
        let path = dir.join(name);
        let file = Hdf5File::create(&path).unwrap();
        let write = |ds_name: &str, data: &[[f32; 2]]| {
            let ds = file
                .new_dataset::<f32>()
                .shape((data.len(), 2))
                .create(ds_name)
                .unwrap();
            let flat: Vec<f32> = data.iter().flatten().copied().collect();
            ds.write_raw(&flat).unwrap();
        };
        write("train", train);
        if !insert.is_empty() {
            write("insert", insert);
        }
        path
    }

    fn write_i64_dataset(file: &Hdf5File, ds_name: &str, rows: &[Vec<i64>]) {
        let cols = rows.first().map(|r| r.len()).unwrap_or(0);
        let ds = file
            .new_dataset::<i64>()
            .shape((rows.len(), cols))
            .create(ds_name)
            .unwrap();
        let flat: Vec<i64> = rows.iter().flatten().copied().collect();
        ds.write_raw(&flat).unwrap();
    }

    #[test]
    fn insert_vectors_ids_continue_after_train() {
        let dir = tempfile::tempdir().unwrap();
        let train = [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let insert = [[3.0, 3.0], [4.0, 4.0]];
        let path = write_hdf5(dir.path(), "d.hdf5", &train, &insert);

        let (ids, vectors) = read_hdf5_insert_vectors(path.to_str().unwrap(), false).unwrap();
        assert_eq!(ids, vec![3, 4]);
        assert_eq!(vectors, vec![vec![3.0f32, 3.0], vec![4.0, 4.0]]);
    }

    #[test]
    fn insert_vectors_missing_dataset_errors() {
        let dir = tempfile::tempdir().unwrap();
        let train = [[1.0, 0.0], [0.0, 1.0]];
        let path = write_hdf5(dir.path(), "d.hdf5", &train, &[]);

        let err = read_hdf5_insert_vectors(path.to_str().unwrap(), false).unwrap_err();
        assert!(err.contains("insert"), "got: {err}");
    }

    #[test]
    fn reads_all_neighbors_dataset() {
        let dir = tempfile::tempdir().unwrap();
        let train = [[1.0, 0.0], [0.0, 1.0]];
        let path = write_hdf5(dir.path(), "d.hdf5", &train, &[]);
        let file = Hdf5File::open_rw(&path).unwrap();
        let rows = vec![vec![1i64, 2, 3], vec![4, 5, 6]];
        write_i64_dataset(&file, "allneighbors", &rows);
        drop(file);

        let got = read_hdf5_all_neighbors(path.to_str().unwrap()).unwrap();
        assert_eq!(got, rows);
    }

    #[test]
    fn all_neighbors_missing_dataset_errors() {
        let dir = tempfile::tempdir().unwrap();
        let train = [[1.0, 0.0], [0.0, 1.0]];
        let path = write_hdf5(dir.path(), "d.hdf5", &train, &[]);

        let err = read_hdf5_all_neighbors(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("allneighbors"), "got: {err}");
    }
}
