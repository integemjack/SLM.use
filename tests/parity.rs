//! The tiny fixture comes from the PyTorch reference: a small model with every layer
//! type, random weights and logits saved at each of 40 positions. Its window is 8, so
//! the chunks below cross the window, wrap the rings and reach the far global layer.

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
    let plain = verify::run_split(&model, dir, 0, 1).unwrap();
    assert_eq!(plain.checked, 40);
    assert!(plain.max_diff < TOLERANCE, "{}: {plain:?}", model.device());
    assert!(plain.same_argmax, "{}: {plain:?}", model.device());
    for (split, chunk) in [
        (1, 1),
        (5, 2),
        (8, 8),
        (9, 3),
        (20, 7),
        (30, 16),
        (39, 256),
        (40, 5),
        (40, 40),
    ] {
        let r = verify::run_split(&model, dir, split, chunk).unwrap();
        assert_eq!(r.checked, 41 - split, "split {split}");
        assert!(
            r.max_diff < TOLERANCE,
            "{}: split {split}, chunk {chunk}: {r:?}",
            model.device()
        );
        assert!(
            r.same_argmax,
            "{}: split {split}, chunk {chunk}: {r:?}",
            model.device()
        );
    }
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
