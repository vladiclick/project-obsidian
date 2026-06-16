// disk_io.rs — Zero-overhead pread() Expert Streaming
//
// Transcribed from MASTER_BLUEPRINT.md §2.2. Unlike executor.rs this depends
// only on REAL crates (safetensors, memmap2, rayon, metal, libc). The `metal`
// crate's exact MTLBuffer/Device method names (new_buffer_with_bytes_no_copy,
// contents, etc.) are version-dependent and may need adjustment once `cargo`
// resolves the actual `metal` crate version.

use rayon::prelude::*;
use safetensors::SafeTensors;
use memmap2::MmapOptions;
use std::collections::HashMap;
use std::fs::File;
use std::os::fd::AsRawFd;

const PAGE_SIZE: usize = 16384; // Apple Silicon page size

#[derive(Clone, Copy, Debug)]
pub struct TensorLoc {
    pub data_offset: u64,
    pub byte_len: usize,
}

pub struct ExpertIndex {
    layers: Vec<LayerExpertFile>,
}

struct LayerExpertFile {
    #[allow(dead_code)]
    mmap: memmap2::Mmap,
    fd: File,
    tensors: HashMap<u32, HashMap<String, TensorLoc>>,
}

impl ExpertIndex {
    pub fn open(expert_dir: &std::path::Path, n_layers: usize) -> Result<Self, ObsidianError> {
        let mut layers = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let path = expert_dir.join(format!("layer_{}_experts.safetensors", il));
            let file = File::open(&path)
                .map_err(|e| ObsidianError::Io(format!("open layer {il}: {e}")))?;
            let mmap = unsafe { MmapOptions::new().map(&file)
                .map_err(|e| ObsidianError::Io(format!("mmap {il}: {e}")))? };
            let st = SafeTensors::deserialize(&mmap)
                .map_err(|e| ObsidianError::Parse(format!("st {il}: {e}")))?;

            let mut tensors: HashMap<u32, HashMap<String, TensorLoc>> = HashMap::new();
            let mmap_ptr = mmap.as_ptr() as usize;
            for name in st.names() {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() != 3 { continue; }
                let expert_id: u32 = match parts[0][1..].parse() { Ok(v) => v, Err(_) => continue };
                let proj_kind = format!("{}.{}", parts[1], parts[2]);
                let view = st.tensor(name)
                    .map_err(|e| ObsidianError::Parse(format!("tensor {name}: {e}")))?;
                let data_offset = (view.data().as_ptr() as usize - mmap_ptr) as u64;
                tensors.entry(expert_id).or_default().insert(
                    proj_kind, TensorLoc { data_offset, byte_len: view.data().len() });
            }
            layers.push(LayerExpertFile { mmap, fd: file, tensors });
        }
        Ok(Self { layers })
    }

    pub fn layer_fd(&self, layer: usize) -> &File { &self.layers[layer].fd }

    pub fn expert_locs(&self, layer: usize, ids: &[u32])
        -> Vec<(u32, &HashMap<String, TensorLoc>)> {
        let lf = &self.layers[layer];
        ids.iter().filter_map(|&e| lf.tensors.get(&e).map(|m| (e, m))).collect()
    }
}

struct ReadReq { fd: i32, offset: u64, len: usize, eid: u32, key: String }
pub struct RawTensor { pub eid: u32, pub key: String, pub bytes: Vec<u8> }

/// Read K experts' tensors in parallel via pread(). Returns raw byte buffers
/// (page-aligned) keyed by (expert_id, "proj.kind"); caller wraps in MTLBuffer.
pub fn read_experts_parallel(
    index: &ExpertIndex,
    layer: usize,
    expert_ids: &[u32],
) -> Vec<RawTensor> {
    let fd = index.layer_fd(layer).as_raw_fd();
    let locs = index.expert_locs(layer, expert_ids);
    let reqs: Vec<ReadReq> = locs.iter().flat_map(|(eid, t)| {
        t.iter().map(move |(k, loc)| ReadReq {
            fd, offset: loc.data_offset, len: loc.byte_len, eid: *eid, key: k.clone(),
        })
    }).collect();

    reqs.par_iter().filter_map(|r| {
        let aligned = ((r.len + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
        let mut buf: Vec<u8> = Vec::with_capacity(aligned);
        unsafe {
            let n = libc::pread(r.fd, buf.as_mut_ptr() as *mut libc::c_void, r.len, r.offset as i64);
            if n < 0 { return None; }
            buf.set_len(r.len);
        }
        Some(RawTensor { eid: r.eid, key: r.key.clone(), bytes: buf })
    }).collect()
}

#[derive(Debug)]
pub enum ObsidianError { Io(String), Parse(String), Metal(String) }

impl From<std::io::Error> for ObsidianError {
    fn from(e: std::io::Error) -> Self { ObsidianError::Io(e.to_string()) }
}
