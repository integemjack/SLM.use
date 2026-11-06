//! Token selection, done on the CPU from the logits of the last step.

#[derive(Clone, Debug)]
pub struct Sampling {
    /// 0 picks the most likely token.
    pub temperature: f32,
    /// 0 keeps every token.
    pub top_k: usize,
    /// 1 disables the penalty.
    pub repeat_penalty: f32,
    /// How many recent tokens the penalty looks at.
    pub repeat_window: usize,
}

/// xorshift64*, enough for sampling and reproducible across platforms.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn pick(mut logits: Vec<f32>, recent: &[u32], s: &Sampling, rng: &mut Rng) -> u32 {
    if s.repeat_penalty != 1.0 {
        let start = recent.len().saturating_sub(s.repeat_window);
        let mut seen: Vec<u32> = recent[start..].to_vec();
        seen.sort_unstable();
        seen.dedup();
        for t in seen {
            let l = &mut logits[t as usize];
            *l = if *l > 0.0 {
                *l / s.repeat_penalty
            } else {
                *l * s.repeat_penalty
            };
        }
    }
    if s.temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .fold(0, |b, (i, v)| if *v > logits[b] { i } else { b }) as u32;
    }
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    if s.top_k > 0 && s.top_k < ids.len() {
        ids.select_nth_unstable_by(s.top_k - 1, |a, b| {
            logits[*b as usize].total_cmp(&logits[*a as usize])
        });
        ids.truncate(s.top_k);
    }
    let top = ids
        .iter()
        .map(|i| logits[*i as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = ids
        .iter()
        .map(|i| ((logits[*i as usize] - top) / s.temperature).exp())
        .collect();
    let mut r = rng.next_f32() * weights.iter().sum::<f32>();
    for (i, w) in ids.iter().zip(&weights) {
        r -= w;
        if r <= 0.0 {
            return *i;
        }
    }
    *ids.last().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_takes_the_largest() {
        let s = Sampling {
            temperature: 0.0,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_window: 0,
        };
        assert_eq!(pick(vec![0.0, 3.0, 1.0], &[], &s, &mut Rng::new(1)), 1);
    }

    #[test]
    fn penalty_can_change_the_choice() {
        let s = Sampling {
            temperature: 0.0,
            top_k: 0,
            repeat_penalty: 2.0,
            repeat_window: 8,
        };
        assert_eq!(pick(vec![0.0, 3.0, 2.0], &[1], &s, &mut Rng::new(1)), 2);
    }

    #[test]
    fn top_k_limits_the_candidates() {
        let s = Sampling {
            temperature: 1.0,
            top_k: 2,
            repeat_penalty: 1.0,
            repeat_window: 0,
        };
        let mut rng = Rng::new(7);
        for _ in 0..200 {
            assert!(pick(vec![5.0, 4.0, -9.0, -9.0], &[], &s, &mut rng) < 2);
        }
    }
}
