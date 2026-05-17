//! Environment probe for the auto-provisioner.
//!
//! Sequential, fail-fast checks: GPU → Python → vLLM importable →
//! HF token → disk space. Each step returns one of:
//!
//! - [`ProbeResult::Ready`] — every check passed; the caller can
//!   safely launch a vLLM engine for the requested model.
//! - [`ProbeResult::Skipped`] — a missing prereq the user can fix
//!   (no GPU, no Python, no token, etc.). The auto-provisioner logs
//!   the `reason` + `hint` and continues without the deployment.
//! - [`ProbeResult::Error`] — something genuinely went wrong while
//!   probing (subprocess crashed, IO failure). Logged at `warn` level.
//!
//! The probe is best-effort: a `ProbeResult::Ready` does **not**
//! guarantee the engine will boot — vLLM's own validation runs at
//! launch time. The probe just rules out the obvious "no GPU on
//! this laptop" / "vllm not installed" / "Gemma ToS not accepted"
//! cases up front so the operator gets a clean message instead of a
//! Python stack trace 30 seconds in.

#![cfg(feature = "gemma-default")]

use std::process::Stdio;

use atomr_infer_core::error::InferenceError;

use crate::hf_cache::HfCache;

/// Outcome of the env probe.
#[derive(Debug)]
pub enum ProbeResult {
    /// All gates passed. The caller may launch a vLLM engine for
    /// `model_id` against the resolved [`HfCache`].
    Ready { vram_free_gb: f32, hf_cache: HfCache },
    /// A user-fixable prereq is missing. The auto-provisioner logs
    /// the message and continues without the deployment.
    Skipped { reason: String, hint: String },
    /// Genuine failure during the probe itself (subprocess crash, IO
    /// error). Logged at `warn` so it isn't silently swallowed.
    Error(InferenceError),
}

/// Probe the environment for compatibility with `model_id`.
///
/// `min_vram_gb` is the per-variant floor — see
/// `atomr_infer::defaults::gemma::min_vram_gb` for the canonical
/// table. `min_disk_gb` is the rough on-disk size; for Gemma 4 E4B
/// that's ~7 GB; E2B ~4 GB. Best-effort on non-unix where `df` may
/// not be available — the disk gate is skipped silently in that case.
pub fn probe(
    model_id: &str,
    min_vram_gb: f32,
    min_disk_gb: f32,
    suggest_smaller_variant: Option<&str>,
) -> ProbeResult {
    // 1. HF cache layout (cheap; just env var reads).
    let hf_cache = match HfCache::resolve() {
        Ok(c) => c,
        Err(e) => return ProbeResult::Error(e),
    };

    // 2. GPU + VRAM probe via cudarc. cudarc handles the "no driver"
    //    case by returning an Err from CudaContext::new(0); that's
    //    the path most CPU-only laptops hit.
    let vram_free_gb = match probe_gpu() {
        Ok(gb) => gb,
        Err(reason) => {
            return ProbeResult::Skipped {
                reason,
                hint: "set ATOMR_INFER_GEMMA_AUTO=skip-quietly to suppress this message; \
                       set ATOMR_INFER_GEMMA_MODEL=<remote-provider> to use a remote backend"
                    .into(),
            }
        }
    };

    if vram_free_gb < min_vram_gb {
        let hint = if let Some(smaller) = suggest_smaller_variant {
            format!(
                "free VRAM {vram_free_gb:.1} GB < {min_vram_gb:.1} GB needed for {model_id}; \
                 try ATOMR_INFER_GEMMA_MODEL={smaller}"
            )
        } else {
            format!(
                "free VRAM {vram_free_gb:.1} GB < {min_vram_gb:.1} GB needed for {model_id}; \
                 no smaller supported variant — consider a remote backend"
            )
        };
        return ProbeResult::Skipped {
            reason: format!("insufficient VRAM for {model_id}"),
            hint,
        };
    }

    // 3. Python 3.10+ on PATH.
    match probe_python() {
        Ok(()) => {}
        Err(reason) => {
            return ProbeResult::Skipped {
                reason,
                hint: "install Python 3.10+ and ensure `python3` is on PATH".into(),
            }
        }
    }

    // 4. `import vllm` succeeds.
    match probe_vllm() {
        Ok(version) => tracing::debug!(version, "vllm import probe ok"),
        Err(reason) => {
            return ProbeResult::Skipped {
                reason,
                hint: "install vLLM in the active venv: `pip install 'vllm>=0.6.4'`".into(),
            }
        }
    }

    // 5. HF token present.
    match hf_cache.discover_token() {
        Ok(Some(_)) => {}
        Ok(None) => {
            return ProbeResult::Skipped {
                reason: "no HuggingFace token found".into(),
                hint: format!(
                    "run `huggingface-cli login`, then accept the Gemma ToS at \
                     https://huggingface.co/{model_id}"
                ),
            }
        }
        Err(e) => return ProbeResult::Error(e),
    }

    // 6. Disk space (best-effort).
    if let Some(free) = hf_cache.free_bytes() {
        let free_gb = free as f32 / 1e9;
        if free_gb < min_disk_gb {
            return ProbeResult::Skipped {
                reason: format!(
                    "insufficient disk for {model_id}: {free_gb:.1} GB free at {} \
                     (need ~{min_disk_gb:.1} GB)",
                    hf_cache.hub_cache.display()
                ),
                hint: "free up space or set HF_HUB_CACHE to a different mountpoint".into(),
            };
        }
    } // None ⇒ unknown, skip gate

    ProbeResult::Ready {
        vram_free_gb,
        hf_cache,
    }
}

/// Returns free VRAM on device 0 in GB, or a human-readable reason
/// the GPU couldn't be probed.
///
/// Uses `nvidia-smi` rather than cudarc so a libcuda /
/// CUDA-bindings version mismatch doesn't false-skip the probe.
/// vLLM's own CUDA bindings (separate from cudarc) are what actually
/// run the workload, and they're tolerant of a wider driver range.
fn probe_gpu() -> Result<f32, String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("nvidia-smi not on PATH (no NVIDIA driver?): {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("nvidia-smi exited non-zero: {}", stderr.trim()));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let free_mib: f32 = stdout
        .trim()
        .lines()
        .next()
        .ok_or_else(|| "nvidia-smi returned empty output".to_string())?
        .trim()
        .parse()
        .map_err(|e| format!("could not parse nvidia-smi output `{}`: {e}", stdout.trim()))?;
    Ok(free_mib / 1024.0)
}

/// Verify Python 3.10+ is on PATH.
fn probe_python() -> Result<(), String> {
    let out = std::process::Command::new("python3")
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("python3 not on PATH: {e}"))?;
    if !out.status.success() {
        return Err("python3 --version returned non-zero exit".into());
    }
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // Output is "Python 3.X.Y"
    let version = combined
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| format!("could not parse `{}`", combined.trim()))?;
    let mut parts = version.split('.');
    let major: u32 = parts
        .next()
        .ok_or("missing major")?
        .parse()
        .map_err(|e| format!("bad major: {e}"))?;
    let minor: u32 = parts
        .next()
        .ok_or("missing minor")?
        .parse()
        .map_err(|e| format!("bad minor: {e}"))?;
    if (major, minor) < (3, 10) {
        return Err(format!("Python {major}.{minor} on PATH; vLLM requires 3.10+"));
    }
    Ok(())
}

/// Verify `import vllm` works in the active python3.
fn probe_vllm() -> Result<String, String> {
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg("import vllm; print(vllm.__version__)")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("python3 spawn failed: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        // Surface the most common case: ModuleNotFoundError.
        if stderr.contains("ModuleNotFoundError") || stderr.contains("No module named 'vllm'") {
            return Err("vLLM not importable in active python3".into());
        }
        return Err(format!("vllm import probe failed: {}", stderr.trim()));
    }
    let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
    Ok(version)
}
