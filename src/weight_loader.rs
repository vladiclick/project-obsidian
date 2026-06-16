// weight_loader.rs — load NON-expert tensors (attention, norms, router gate,
// embed, lm_head) from the model's safetensors shards into StorageModeShared
// MTLBuffers. Raw bytes (4-bit packed stays packed; dequant happens in-kernel).
// Experts (switch_mlp.*) are streamed per-token by disk_io, so skipped here.

use crate::disk_io::ObsidianError;
use memmap2::MmapOptions;
use metal::{Buffer, Device, MTLResourceOptions};
use safetensors::SafeTensors;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;

pub struct Weights {
    pub tensors: HashMap<String, Buffer>,
    pub shapes: HashMap<String, Vec<usize>>,
}

impl Weights {
    pub fn load_non_expert(model_dir: &Path, dev: &Device) -> Result<Self, ObsidianError> {
        let idx = model_dir.join("model.safetensors.index.json");
        let shards: Vec<String> = if idx.exists() {
            let v: serde_json::Value = serde_json::from_reader(File::open(&idx)?)
                .map_err(|e| ObsidianError::Parse(e.to_string()))?;
            v["weight_map"].as_object()
                .ok_or_else(|| ObsidianError::Parse("no weight_map".into()))?
                .values().filter_map(|x| x.as_str().map(String::from))
                .collect::<BTreeSet<_>>().into_iter().collect()
        } else {
            vec!["model.safetensors".into()]
        };

        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        let opt = MTLResourceOptions::StorageModeShared;

        for sf in &shards {
            let f = File::open(model_dir.join(sf))?;
            let mm = unsafe { MmapOptions::new().map(&f)? };
            let st = SafeTensors::deserialize(&mm)
                .map_err(|e| ObsidianError::Parse(e.to_string()))?;
            for name in st.names() {
                if name.contains("switch_mlp") { continue; } // experts: streamed
                let view = st.tensor(name)
                    .map_err(|e| ObsidianError::Parse(e.to_string()))?;
                let data = view.data();
                let buf = dev.new_buffer(data.len() as u64, opt);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(), buf.contents() as *mut u8, data.len());
                }
                tensors.insert(name.to_string(), buf);
                shapes.insert(name.to_string(), view.shape().to_vec());
            }
        }
        log::info!("loaded {} non-expert tensors", tensors.len());
        Ok(Self { tensors, shapes })
    }

    pub fn get(&self, name: &str) -> Option<&Buffer> {
        self.tensors.get(name)
    }
}
