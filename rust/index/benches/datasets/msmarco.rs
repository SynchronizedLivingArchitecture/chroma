//! MS MARCO v2 with Cohere embed-multilingual-v3 embeddings: ~138M vectors, 1024 dimensions.

use std::fs::File;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Float64Array, ListArray};
use arrow::datatypes::ArrowNativeType;
use chroma_distance::DistanceFunction;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::{ground_truth, Dataset, Query};

const REPO_ID: &str = "Cohere/msmarco-v2-embed-multilingual-v3";
const NUM_SHARDS: usize = 139;
pub const DIMENSION: usize = 1024;
pub const DATA_LEN: usize = 138_364_198;
const COLUMN: &str = "emb";

const SHARD_FILES: [&str; NUM_SHARDS] = [
    "corpus/0000.parquet",
    "corpus/0001.parquet",
    "corpus/0002.parquet",
    "corpus/0003.parquet",
    "corpus/0004.parquet",
    "corpus/0005.parquet",
    "corpus/0006.parquet",
    "corpus/0007.parquet",
    "corpus/0008.parquet",
    "corpus/0009.parquet",
    "corpus/0010.parquet",
    "corpus/0011.parquet",
    "corpus/0012.parquet",
    "corpus/0013.parquet",
    "corpus/0014.parquet",
    "corpus/0015.parquet",
    "corpus/0016.parquet",
    "corpus/0017.parquet",
    "corpus/0018.parquet",
    "corpus/0019.parquet",
    "corpus/0020.parquet",
    "corpus/0021.parquet",
    "corpus/0022.parquet",
    "corpus/0023.parquet",
    "corpus/0024.parquet",
    "corpus/0025.parquet",
    "corpus/0026.parquet",
    "corpus/0027.parquet",
    "corpus/0028.parquet",
    "corpus/0029.parquet",
    "corpus/0030.parquet",
    "corpus/0031.parquet",
    "corpus/0032.parquet",
    "corpus/0033.parquet",
    "corpus/0034.parquet",
    "corpus/0035.parquet",
    "corpus/0036.parquet",
    "corpus/0037.parquet",
    "corpus/0038.parquet",
    "corpus/0039.parquet",
    "corpus/0040.parquet",
    "corpus/0041.parquet",
    "corpus/0042.parquet",
    "corpus/0043.parquet",
    "corpus/0044.parquet",
    "corpus/0045.parquet",
    "corpus/0046.parquet",
    "corpus/0047.parquet",
    "corpus/0048.parquet",
    "corpus/0049.parquet",
    "corpus/0050.parquet",
    "corpus/0051.parquet",
    "corpus/0052.parquet",
    "corpus/0053.parquet",
    "corpus/0054.parquet",
    "corpus/0055.parquet",
    "corpus/0056.parquet",
    "corpus/0057.parquet",
    "corpus/0058.parquet",
    "corpus/0059.parquet",
    "corpus/0060.parquet",
    "corpus/0061.parquet",
    "corpus/0062.parquet",
    "corpus/0063.parquet",
    "corpus/0064.parquet",
    "corpus/0065.parquet",
    "corpus/0066.parquet",
    "corpus/0067.parquet",
    "corpus/0068.parquet",
    "corpus/0069.parquet",
    "corpus/0070.parquet",
    "corpus/0071.parquet",
    "corpus/0072.parquet",
    "corpus/0073.parquet",
    "corpus/0074.parquet",
    "corpus/0075.parquet",
    "corpus/0076.parquet",
    "corpus/0077.parquet",
    "corpus/0078.parquet",
    "corpus/0079.parquet",
    "corpus/0080.parquet",
    "corpus/0081.parquet",
    "corpus/0082.parquet",
    "corpus/0083.parquet",
    "corpus/0084.parquet",
    "corpus/0085.parquet",
    "corpus/0086.parquet",
    "corpus/0087.parquet",
    "corpus/0088.parquet",
    "corpus/0089.parquet",
    "corpus/0090.parquet",
    "corpus/0091.parquet",
    "corpus/0092.parquet",
    "corpus/0093.parquet",
    "corpus/0094.parquet",
    "corpus/0095.parquet",
    "corpus/0096.parquet",
    "corpus/0097.parquet",
    "corpus/0098.parquet",
    "corpus/0099.parquet",
    "corpus/0100.parquet",
    "corpus/0101.parquet",
    "corpus/0102.parquet",
    "corpus/0103.parquet",
    "corpus/0104.parquet",
    "corpus/0105.parquet",
    "corpus/0106.parquet",
    "corpus/0107.parquet",
    "corpus/0108.parquet",
    "corpus/0109.parquet",
    "corpus/0110.parquet",
    "corpus/0111.parquet",
    "corpus/0112.parquet",
    "corpus/0113.parquet",
    "corpus/0114.parquet",
    "corpus/0115.parquet",
    "corpus/0116.parquet",
    "corpus/0117.parquet",
    "corpus/0118.parquet",
    "corpus/0119.parquet",
    "corpus/0120.parquet",
    "corpus/0121.parquet",
    "corpus/0122.parquet",
    "corpus/0123.parquet",
    "corpus/0124.parquet",
    "corpus/0125.parquet",
    "corpus/0126.parquet",
    "corpus/0127.parquet",
    "corpus/0128.parquet",
    "corpus/0129.parquet",
    "corpus/0130.parquet",
    "corpus/0131.parquet",
    "corpus/0132.parquet",
    "corpus/0133.parquet",
    "corpus/0134.parquet",
    "corpus/0135.parquet",
    "corpus/0136.parquet",
    "corpus/0137.parquet",
    "corpus/0138.parquet",
];

fn cache_dir() -> PathBuf {
    dirs::home_dir()
        .expect("failed to get home directory")
        .join(".cache/msmarco_v2")
}

fn gt_path() -> PathBuf {
    cache_dir().join("ground_truth.parquet")
}

/// MS MARCO v2 dataset handle.
pub struct MsMarco {
    shard_paths: Vec<PathBuf>,
}

impl MsMarco {
    /// Load MS MARCO v2 dataset from HuggingFace Hub.
    /// Requires ground truth to be precomputed at ~/.cache/msmarco_v2/ground_truth.parquet
    pub async fn load() -> io::Result<Self> {
        // Check ground truth exists before downloading shards
        if !ground_truth::exists(&gt_path()) {
            return Err(io::Error::other(format!(
                "Ground truth not found at {}.\n  \
                 Run: python sphroma/scripts/compute_ground_truth.py --dataset msmarco",
                gt_path().display()
            )));
        }

        println!("Loading MS MARCO v2 from HuggingFace Hub...");

        let api = hf_hub::api::tokio::Api::new().map_err(io::Error::other)?;
        let repo = api.dataset(REPO_ID.to_string());

        let mut shard_paths = Vec::with_capacity(NUM_SHARDS);
        for filename in SHARD_FILES.iter() {
            let path = repo.get(filename).await.map_err(io::Error::other)?;
            shard_paths.push(path);
        }

        Ok(Self { shard_paths })
    }

    /// Load vectors in range [offset, offset+limit).
    /// Returns (global_id, embedding) pairs.
    pub fn load_range(&self, offset: usize, limit: usize) -> io::Result<Vec<(u32, Arc<[f32]>)>> {
        let end = (offset + limit).min(DATA_LEN);
        if offset >= end {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(end - offset);
        let mut global_idx = 0usize;
        let mut collected = 0usize;

        for shard_path in &self.shard_paths {
            if collected >= limit || global_idx >= end {
                break;
            }

            let file = File::open(shard_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let num_rows = builder.metadata().file_metadata().num_rows() as usize;

            // Skip shards entirely before our range
            if global_idx + num_rows <= offset {
                global_idx += num_rows;
                continue;
            }

            let reader = builder
                .with_batch_size(10_000)
                .build()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            for batch in reader {
                if collected >= limit {
                    break;
                }

                let batch = batch.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let col_idx = batch
                    .schema()
                    .fields()
                    .iter()
                    .position(|f| f.name() == COLUMN)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "column not found")
                    })?;

                let col = batch.column(col_idx);
                let list_array = col.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "column is not a list")
                })?;

                let offsets = list_array.offsets();
                let inner = list_array.values();

                for i in 0..list_array.len() {
                    if list_array.is_null(i) {
                        global_idx += 1;
                        continue;
                    }

                    // Skip if before offset
                    if global_idx < offset {
                        global_idx += 1;
                        continue;
                    }

                    // Stop if we've collected enough
                    if collected >= limit {
                        break;
                    }

                    let start = offsets[i].as_usize();
                    let end_off = offsets[i + 1].as_usize();

                    let vec: Arc<[f32]> = if let Some(f32_arr) =
                        inner.as_any().downcast_ref::<Float32Array>()
                    {
                        Arc::from(&f32_arr.values()[start..end_off])
                    } else if let Some(f64_arr) = inner.as_any().downcast_ref::<Float64Array>() {
                        let values: Vec<f32> = f64_arr.values()[start..end_off]
                            .iter()
                            .map(|&v| v as f32)
                            .collect();
                        Arc::from(values)
                    } else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "unsupported array type",
                        ));
                    };

                    result.push((global_idx as u32, vec));
                    global_idx += 1;
                    collected += 1;
                }
            }
        }

        Ok(result)
    }
}

impl Dataset for MsMarco {
    fn name(&self) -> &str {
        "msmarco-v2"
    }

    fn dimension(&self) -> usize {
        DIMENSION
    }

    fn data_len(&self) -> usize {
        DATA_LEN
    }

    fn k(&self) -> usize {
        ground_truth::K
    }

    fn load_range(&self, offset: usize, limit: usize) -> io::Result<Vec<(u32, Arc<[f32]>)>> {
        MsMarco::load_range(self, offset, limit)
    }

    fn queries(&self, distance_function: DistanceFunction) -> io::Result<Vec<Query>> {
        ground_truth::load(&gt_path(), distance_function)
    }
}
