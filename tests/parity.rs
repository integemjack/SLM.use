//! The tiny fixture comes from the PyTorch reference: a small model with every layer
//! type, random weights and logits saved at each position.

use std::path::Path;

use slm::cpu::Cpu;
use slm::model::Model;
use slm::ops::Ops;
use slm::verify;

const FIXTURE: &str = "tests/fixtures/tiny";
const TOLERANCE: f32 = 1e-4;

fn check<O: Ops>(ops: O) {
    let dir = Path::new(FIXTURE);
    let model = Model::load(ops, dir).unwrap();
    let r = verify::run(&model, dir).unwrap();
    assert_eq!(r.checked, 40);
    assert!(
        r.max_diff < TOLERANCE,
        "{}: max diff {}",
        model.device(),
        r.max_diff
    );
    assert!(r.same_argmax, "{}: argmax differs", model.device());
}

#[test]
fn cpu_matches_reference() {
    check(Cpu);
}

#[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
#[test]
fn cuda_matches_reference() {
    match slm::cuda::Cuda::probe().unwrap() {
        Some(ops) => check(ops),
        None => eprintln!("skipped: no NVIDIA GPU"),
    }
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn mlx_matches_reference() {
    check(slm::mlx::Mlx::new(false).unwrap());
}
