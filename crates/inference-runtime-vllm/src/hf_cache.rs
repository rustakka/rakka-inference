//! HuggingFace cache + token resolution.
//!
//! Mirrors the precedence used by every other HF tool so a workstation
//! that already has `huggingface-cli` configured "just works" without
//! re-downloading models. Multi-instance unification falls out of
//! pointing every atomr-infer process at the same `$HF_HOME`.
//!
//! ## Cache resolution
//!
//! 1. `HF_HUB_CACHE` env var (explicit override)
//! 2. `HF_HOME/hub` if `HF_HOME` is set
//! 3. `XDG_CACHE_HOME/huggingface/hub` if `XDG_CACHE_HOME` is set
//! 4. `~/.cache/huggingface/hub`
//!
//! ## Token resolution
//!
//! 1. `HF_TOKEN` env var
//! 2. `HUGGING_FACE_HUB_TOKEN` env var (legacy alias)
//! 3. `{home}/token` file (the `huggingface-cli login` output)

#![cfg(feature = "gemma-default")]

use std::path::{Path, PathBuf};

use atomr_infer_core::error::{InferenceError, InferenceResult};

/// Resolved HuggingFace cache layout. All fields are absolute paths.
#[derive(Debug, Clone)]
pub struct HfCache {
    /// `$HF_HOME` — root of the HuggingFace cache hierarchy.
    pub home: PathBuf,
    /// `$HF_HUB_CACHE` — where downloaded models live (typically
    /// `{home}/hub`).
    pub hub_cache: PathBuf,
    /// Path to the user's saved HF token file
    /// (typically `{home}/token`). May not exist; that's fine.
    pub token_path: PathBuf,
}

impl HfCache {
    /// Resolve the cache layout from env vars and platform defaults.
    /// Does not create directories — the caller may probe `home`
    /// for free space without committing to a layout.
    pub fn resolve() -> InferenceResult<Self> {
        let home = if let Some(h) = env_path("HF_HOME") {
            h
        } else if let Some(xdg) = env_path("XDG_CACHE_HOME") {
            xdg.join("huggingface")
        } else if let Some(home_dir) = dirs::home_dir() {
            home_dir.join(".cache").join("huggingface")
        } else {
            return Err(InferenceError::Internal(
                "hf-cache: no HF_HOME / XDG_CACHE_HOME / $HOME — cannot resolve cache".into(),
            ));
        };

        let hub_cache = env_path("HF_HUB_CACHE").unwrap_or_else(|| home.join("hub"));

        let token_path = home.join("token");

        Ok(Self {
            home,
            hub_cache,
            token_path,
        })
    }

    /// Load the user's HF token. Returns `Ok(Some(token))` if found
    /// in any standard location, `Ok(None)` if absent (caller decides
    /// whether that's an error), `Err(...)` for IO failures reading
    /// the token file.
    pub fn discover_token(&self) -> InferenceResult<Option<String>> {
        // Env vars win — they're the most explicit.
        if let Some(t) = env_string("HF_TOKEN") {
            return Ok(Some(t));
        }
        if let Some(t) = env_string("HUGGING_FACE_HUB_TOKEN") {
            return Ok(Some(t));
        }

        // Then the saved-token file from `huggingface-cli login`.
        if self.token_path.exists() {
            let raw = std::fs::read_to_string(&self.token_path).map_err(|e| {
                InferenceError::Internal(format!(
                    "hf-cache: failed to read {}: {e}",
                    self.token_path.display()
                ))
            })?;
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                return Ok(Some(trimmed.to_string()));
            }
        }

        Ok(None)
    }

    /// Free disk-space at `hub_cache`'s mountpoint, in bytes.
    /// Returns `None` if the path doesn't yet exist (caller may
    /// then fall back to `home`'s parent).
    pub fn free_bytes(&self) -> Option<u64> {
        free_bytes_at(&self.hub_cache)
            .or_else(|| free_bytes_at(&self.home))
            .or_else(|| free_bytes_at(self.home.parent().unwrap_or(Path::new("/"))))
    }
}

fn env_path(var: &str) -> Option<PathBuf> {
    env_string(var).map(PathBuf::from)
}

fn env_string(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(unix)]
fn free_bytes_at(path: &Path) -> Option<u64> {
    let p = path.ancestors().find(|p| p.exists())?;
    // `df --output=avail -B1 <path>` prints a header + the number of
    // available bytes. Best-effort: any non-zero exit, parse failure,
    // or `df` not on PATH ⇒ `None` and the probe skips the gate.
    let out = std::process::Command::new("df")
        .arg("--output=avail")
        .arg("-B1")
        .arg(p)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout.lines().nth(1)?.trim().parse::<u64>().ok()
}

#[cfg(not(unix))]
fn free_bytes_at(_path: &Path) -> Option<u64> {
    // Best-effort on non-unix; the probe falls back to "unknown" and
    // skips the disk-space gate.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests in this module share the global env, so they must be
    /// serialised against one another. `cargo test` runs tests
    /// concurrently by default.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn resolves_with_explicit_hf_home() {
        let _g = env_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        std::env::set_var("HF_HOME", tmp.path());
        std::env::remove_var("HF_HUB_CACHE");
        let c = HfCache::resolve().expect("resolve");
        assert_eq!(c.home, tmp.path());
        assert_eq!(c.hub_cache, tmp.path().join("hub"));
        assert_eq!(c.token_path, tmp.path().join("token"));
    }

    #[test]
    fn explicit_hub_cache_overrides_default() {
        let _g = env_lock();
        let home = tempfile::tempdir().expect("tempdir");
        let hub = tempfile::tempdir().expect("tempdir");
        std::env::set_var("HF_HOME", home.path());
        std::env::set_var("HF_HUB_CACHE", hub.path());
        let c = HfCache::resolve().expect("resolve");
        assert_eq!(c.hub_cache, hub.path());
        std::env::remove_var("HF_HUB_CACHE");
    }

    #[test]
    fn token_discovery_env_var_wins() {
        let _g = env_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        std::env::set_var("HF_HOME", tmp.path());
        std::env::set_var("HF_TOKEN", "hf_test123");
        let c = HfCache::resolve().expect("resolve");
        assert_eq!(c.discover_token().expect("ok"), Some("hf_test123".into()));
        std::env::remove_var("HF_TOKEN");
    }

    #[test]
    fn token_discovery_falls_back_to_file() {
        let _g = env_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        std::env::set_var("HF_HOME", tmp.path());
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
        let token_path = tmp.path().join("token");
        std::fs::write(&token_path, "hf_from_file\n").expect("write");
        let c = HfCache::resolve().expect("resolve");
        assert_eq!(
            c.discover_token().expect("ok"),
            Some("hf_from_file".into())
        );
    }

    #[test]
    fn token_discovery_returns_none_when_absent() {
        let _g = env_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        std::env::set_var("HF_HOME", tmp.path());
        std::env::remove_var("HF_TOKEN");
        std::env::remove_var("HUGGING_FACE_HUB_TOKEN");
        let c = HfCache::resolve().expect("resolve");
        assert_eq!(c.discover_token().expect("ok"), None);
    }
}
