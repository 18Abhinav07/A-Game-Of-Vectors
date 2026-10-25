//! sieve: IVF + 1-bit Matryoshka codes + asymmetric LUT scoring + int8 rerank.
//! See DESIGN.md for the decisions behind each module.

pub mod blas;
pub mod data;
pub mod eval;
pub mod ivf;
pub mod lut;
pub mod postings;
pub mod quant;
pub mod rabitq;
pub mod shard;
pub mod store;
pub mod tier2;
pub mod topk;

/// Full embedding width (Arctic Embed M v1.5).
pub const DIM: usize = 768;
/// Matryoshka prefix the codes are built from.
pub const PDIM: usize = 256;
/// Bytes per 1-bit code of the prefix.
pub const CODE_BYTES: usize = PDIM / 8;
