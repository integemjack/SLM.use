//! Runs the real binary on the tiny fixture, whose default is 3 loops and limit is 6.

use std::process::{Command, Output};

fn slm(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_slm"))
        .args(["run", "Hello", "-m", "tests/fixtures/tiny", "-d", "cpu"])
        .args(["-n", "8", "-t", "0"])
        .args(args)
        .output()
        .unwrap()
}

fn text(args: &[&str]) -> String {
    let out = slm(args);
    assert!(out.status.success(), "{out:?}");
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn a_level_means_its_loop_count() {
    assert_eq!(text(&["--think", "low"]), text(&["-R", "1"]));
    assert_eq!(text(&["--think", "medium"]), text(&[]));
    assert_eq!(text(&["--think", "high"]), text(&["-R", "6"]));
    assert_eq!(text(&["--think", "max"]), text(&["-R", "6"]));
    assert_ne!(text(&["--think", "low"]), text(&["--think", "max"]));
}

#[test]
fn a_level_and_a_loop_count_conflict() {
    assert!(!slm(&["--think", "high", "-R", "2"]).status.success());
    assert!(!slm(&["--think", "huge"]).status.success());
}
