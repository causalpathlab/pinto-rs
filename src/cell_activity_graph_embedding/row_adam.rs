//! A table trained one batch of rows at a time by a lazy Adam.
//!
//! cage-cell learns a row per cell pair, millions of them, and a step
//! touches only the few thousand its draws name. A table in the `VarMap`
//! would be stepped densely by candle's AdamW: every row, its two moments
//! and its weight decay, every step, whether the step drew it or not. So
//! the table and its moments live here instead, outside the map:
//!
//! ```text
//! rows(ids)  -> [U × D] copy on the compute device (the step's leaf)
//! step(ids, z, g)  on those distinct rows only, with each row's own step
//!                  count t_r:
//!   m_r ← β1 m_r + (1-β1) g_r          v_r ← β2 v_r + (1-β2) g_r²
//!   z_r ← z_r (1 - lr λ) - lr · (m_r / (1-β1^t_r)) / (sqrt(v_r / (1-β2^t_r)) + ε)
//! ```
//!
//! the expressions candle's `AdamW::step` evaluates, in the same order, so a
//! row stepped every step ends where the dense optimizer puts it, bit for
//! bit, and a row no step drew keeps its value and its moments exactly.
//! A row a step names twice is the caller's to merge first: its leaf holds
//! each row once, so autograd has summed the gradient already. Each touched
//! row is written back in place ([`set_rows`]), the moments `[m | v]` side
//! by side in one table so they move in one read and one write; nothing
//! table-sized is copied in a step.
//!
//! The table may live on the host while the step computes on the device
//! ([`Residency`]): only a step's rows cross, both ways.

use legume_numeric::candle::candle_core::{
    CpuStorage, DType, Device, InplaceOp3, Layout, Result, Tensor,
};
use legume_numeric::candle::candle_nn::ParamsAdamW;
use legume_numeric::candle::fast_index::gather_rows;

/// Where a row table and its moments are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// On the device the step computes on.
    Device,
    /// In host memory; a step moves its rows to the device and back.
    Host,
}

/// Where a table of `bytes` (with its two moments) is kept: on the device
/// when that fits half of `fraction` (clamped to 0.05–0.95, as
/// `--gpu-mem-fraction` is) of its `free` memory, the rest left to the
/// steps; on the host otherwise. A device that cannot say what is free
/// (the CPU) keeps it.
pub fn residency(bytes: usize, free: Option<usize>, fraction: f32) -> Residency {
    match free {
        None => Residency::Device,
        Some(free) => {
            let room = f64::from(fraction.clamp(0.05, 0.95)) * free as f64 / 2.0;
            if bytes as f64 <= room {
                Residency::Device
            } else {
                Residency::Host
            }
        }
    }
}

/// A `[R × D]` table, its Adam moments `[R × 2D]` (`m` then `v`), and each
/// row's step count.
pub struct RowAdam {
    table: Tensor,
    moments: Tensor,
    steps: Vec<u32>,
    params: ParamsAdamW,
    compute: Device,
}

impl RowAdam {
    /// A table of `rows × d` values `init` (row-major), kept on `store`
    /// (the compute device, or the CPU for a host-resident table), its
    /// moments zero; steps run on `compute` with `params`, the dense
    /// optimizer's settings. The table owns its storage: a step writes it
    /// in place, so it must share it with no other tensor.
    pub fn new(
        init: Vec<f32>,
        (rows, d): (usize, usize),
        store: &Device,
        params: ParamsAdamW,
        compute: &Device,
    ) -> Result<Self> {
        Ok(Self {
            table: Tensor::from_vec(init, (rows, d), store)?,
            moments: Tensor::zeros((rows, 2 * d), DType::F32, store)?,
            steps: vec![0; rows],
            params,
            compute: compute.clone(),
        })
    }

    pub fn n_rows(&self) -> usize {
        self.steps.len()
    }

    /// Where the table is kept.
    pub fn residency(&self) -> Residency {
        if self.table.device().same_device(&self.compute) {
            Residency::Device
        } else {
            Residency::Host
        }
    }

    /// The table as it stands.
    #[cfg(test)]
    pub fn table(&self) -> &Tensor {
        &self.table
    }

    /// The table, the moments freed first.
    pub fn into_table(self) -> Tensor {
        let Self { table, moments, .. } = self;
        drop(moments);
        table
    }

    /// The moments `(m, v)` as they stand.
    #[cfg(test)]
    pub fn moments(&self) -> (Tensor, Tensor) {
        let d = self.table.dim(1).expect("a 2-d table");
        (
            self.moments
                .narrow(1, 0, d)
                .and_then(|t| t.contiguous())
                .expect("m"),
            self.moments
                .narrow(1, d, d)
                .and_then(|t| t.contiguous())
                .expect("v"),
        )
    }

    /// Rows `ids` of the table, `[ids.len() × D]` on the compute device,
    /// detached: the leaf a step differentiates.
    pub fn rows(&self, ids: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_slice(ids, ids.len(), self.table.device())?;
        Ok(gather_rows(&self.table, &idx)?
            .to_device(&self.compute)?
            .detach())
    }

    /// One lazy Adam step of the distinct rows `ids` (ascending, as a
    /// step's leaf lists them), whose current values `current` and
    /// gradient `grad` (`[ids.len() × D]` each) are on the compute device.
    pub fn step(&mut self, ids: &[u32], current: &Tensor, grad: &Tensor) -> Result<()> {
        if !ids.windows(2).all(|w| w[0] < w[1]) {
            return Err(legume_numeric::candle::candle_core::Error::Msg(
                "row Adam: a step's rows must be distinct and ascending".into(),
            ));
        }
        if ids.is_empty() {
            return Ok(());
        }
        let n = ids.len();
        let d = current.dim(1)?;
        let p = &self.params;
        let (beta1, beta2) = (p.beta1, p.beta2);
        let lr_lambda = p.lr * p.weight_decay;
        let (mut scale_m, mut scale_v) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for &r in ids {
            let t = &mut self.steps[r as usize];
            *t += 1;
            scale_m.push((1f64 / (1f64 - beta1.powi(*t as i32))) as f32);
            scale_v.push((1f64 / (1f64 - beta2.powi(*t as i32))) as f32);
        }
        let scale_m = Tensor::from_vec(scale_m, (n, 1), &self.compute)?;
        let scale_v = Tensor::from_vec(scale_v, (n, 1), &self.compute)?;
        let store = self.table.device().clone();
        let idx = Tensor::from_slice(ids, n, &store)?;
        let mv = gather_rows(&self.moments, &idx)?.to_device(&self.compute)?;
        let (m, v) = (mv.narrow(1, 0, d)?, mv.narrow(1, d, d)?);
        let theta = current.detach();
        // candle's `AdamW::step`, row-wise
        let next_m = ((m * beta1)? + (grad * (1.0 - beta1))?)?;
        let next_v = ((v * beta2)? + (grad.sqr()? * (1.0 - beta2))?)?;
        let m_hat = next_m.broadcast_mul(&scale_m)?;
        let v_hat = next_v.broadcast_mul(&scale_v)?;
        let next_theta = (theta * (1f64 - lr_lambda))?;
        let adjusted_grad = (m_hat / (v_hat.sqrt()? + p.eps)?)?;
        let next_theta = (next_theta - (adjusted_grad * p.lr)?)?;
        set_rows(&self.table, &idx, &next_theta.to_device(&store)?)?;
        let next_mv = Tensor::cat(&[&next_m, &next_v], 1)?;
        set_rows(&self.moments, &idx, &next_mv.to_device(&store)?)?;
        Ok(())
    }
}

/// `table[ids[i]] = src[i]`, in place, for a contiguous f32 `table`
/// `[R × D]`, distinct u32 `ids` `[n]` below `R` and `src` `[n × D]`.
pub fn set_rows(table: &Tensor, ids: &Tensor, src: &Tensor) -> Result<()> {
    let src = src.to_dtype(DType::F32)?.contiguous()?;
    let ids = ids.contiguous()?;
    if table.device().is_metal() {
        let (n, d) = src.dims2()?;
        let wide = ids.unsqueeze(1)?.broadcast_as((n, d))?.contiguous()?;
        return table.scatter_set(&wide, &src, 0);
    }
    table.inplace_op3(&ids, &src, &SetRows)
}

struct SetRows;

impl SetRows {
    fn check(table_l: &Layout, ids_l: &Layout, src_l: &Layout) -> Result<(usize, usize)> {
        let (_, d) = table_l.shape().dims2()?;
        let n = ids_l.shape().elem_count();
        let (n_src, d_src) = src_l.shape().dims2()?;
        if n != n_src || d != d_src {
            return Err(legume_numeric::candle::candle_core::Error::Msg(format!(
                "set_rows: {n} ids, {n_src} x {d_src} rows, a table {d} wide"
            )));
        }
        Ok((n, d))
    }
}

fn contiguous(l: &Layout, what: &str) -> Result<(usize, usize)> {
    l.contiguous_offsets().ok_or_else(|| {
        legume_numeric::candle::candle_core::Error::Msg(format!(
            "set_rows: {what} is not contiguous"
        ))
    })
}

impl InplaceOp3 for SetRows {
    fn name(&self) -> &'static str {
        "set-rows"
    }

    fn cpu_fwd(
        &self,
        table: &mut CpuStorage,
        table_l: &Layout,
        ids: &CpuStorage,
        ids_l: &Layout,
        src: &CpuStorage,
        src_l: &Layout,
    ) -> Result<()> {
        let (_, d) = Self::check(table_l, ids_l, src_l)?;
        let (t0, t1) = contiguous(table_l, "the table")?;
        let (i0, i1) = contiguous(ids_l, "ids")?;
        let (s0, s1) = contiguous(src_l, "the rows")?;
        let (CpuStorage::F32(table), CpuStorage::U32(ids), CpuStorage::F32(src)) =
            (table, ids, src)
        else {
            return Err(legume_numeric::candle::candle_core::Error::Msg(
                "set_rows: an f32 table, u32 ids and f32 rows".into(),
            ));
        };
        let table = &mut table[t0..t1];
        let n_rows = table.len() / d.max(1);
        for (i, &row) in ids[i0..i1].iter().enumerate() {
            let row = row as usize;
            if row >= n_rows {
                return Err(legume_numeric::candle::candle_core::Error::Msg(format!(
                    "set_rows: row {row} of a table of {n_rows}"
                )));
            }
            table[row * d..(row + 1) * d].copy_from_slice(&src[s0 + i * d..s0 + (i + 1) * d]);
        }
        debug_assert!(s1 - s0 == (i1 - i0) * d);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        table: &mut legume_numeric::candle::candle_core::CudaStorage,
        table_l: &Layout,
        ids: &legume_numeric::candle::candle_core::CudaStorage,
        ids_l: &Layout,
        src: &legume_numeric::candle::candle_core::CudaStorage,
        src_l: &Layout,
    ) -> Result<()> {
        use legume_numeric::candle::candle_core::backend::BackendStorage;
        use legume_numeric::candle::candle_core::cuda_backend::cudarc::driver::{
            LaunchConfig, PushKernelArg,
        };
        use legume_numeric::candle::candle_core::cuda_backend::WrapErr;
        let (n, d) = Self::check(table_l, ids_l, src_l)?;
        let (t0, t1) = contiguous(table_l, "the table")?;
        let (i0, i1) = contiguous(ids_l, "ids")?;
        let (s0, s1) = contiguous(src_l, "the rows")?;
        let total = n * d;
        if total == 0 {
            return Ok(());
        }
        // one thread per element written: a step's rows, never the table
        let n_rows = (t1 - t0) / d.max(1);
        if total > u32::MAX as usize || n_rows > u32::MAX as usize {
            return Err(legume_numeric::candle::candle_core::Error::Msg(format!(
                "set_rows: {total} elements in one write, or {n_rows} rows"
            )));
        }
        let dev = table.device().clone();
        let ids_slice = ids.as_cuda_slice::<u32>()?.slice(i0..i1);
        let src_slice = src.as_cuda_slice::<f32>()?.slice(s0..s1);
        let mut out = table.as_cuda_slice_mut::<f32>()?.slice_mut(t0..t1);
        let func = dev.get_or_load_custom_func("set_rows_f32", "pinto_row_adam", cuda_ptx()?)?;
        let cfg = LaunchConfig::for_num_elems(total as u32);
        let mut builder = func.builder();
        builder.arg(&ids_slice);
        builder.arg(&src_slice);
        builder.arg(&mut out);
        legume_numeric::candle::candle_core::builder_arg!(builder, total as u32);
        legume_numeric::candle::candle_core::builder_arg!(builder, d as u32);
        legume_numeric::candle::candle_core::builder_arg!(builder, n_rows as u32);
        // SAFETY: the kernel reads `ids` and `src` and writes rows of `out`,
        // all sized by the checks above; a row past the table is dropped.
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
const CUDA_SRC: &str = r#"
extern "C" __global__ void set_rows_f32(
    const unsigned int *ids, const float *src, float *out,
    const unsigned int total, const unsigned int d, const unsigned int n_rows)
{
    // the write is a step's rows (32-bit), the table offset 64-bit, so a
    // table past 2^32 elements is addressed whole
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int r = i / d;
    unsigned int row = ids[r];
    if (row >= n_rows) return;
    out[(unsigned long long)row * d + (i - r * d)] = src[i];
}
"#;

/// The kernel's PTX, compiled once per process through nvrtc.
#[cfg(feature = "cuda")]
fn cuda_ptx() -> Result<&'static str> {
    use std::sync::OnceLock;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    match PTX.get_or_init(|| {
        legume_numeric::candle::candle_core::cuda_backend::cudarc::nvrtc::compile_ptx(CUDA_SRC)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(e) => Err(legume_numeric::candle::candle_core::Error::Msg(format!(
            "set_rows: nvrtc failed: {e}"
        ))),
    }
}

#[cfg(test)]
#[path = "row_adam_tests.rs"]
mod tests;
