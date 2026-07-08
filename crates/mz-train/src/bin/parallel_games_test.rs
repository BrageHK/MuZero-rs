use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use burn::rl::Environment;
use mz_rs::env::cartpole::env::CartPoleWrapper;

fn play_games_for(duration: Duration, games_counter: &AtomicU64) {
    let mut env = CartPoleWrapper::default();
    let mut rng = fastrand::Rng::new();
    let start_time = Instant::now();

    while start_time.elapsed() < duration {
        env.reset();
        loop {
            let action = rng.usize(0..2);
            let result = env.step(action);
            if result.done || result.truncated {
                break;
            }
        }
        games_counter.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    let test_duration = Duration::from_secs(3);
    let max_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    println!("Available parallelism: {max_threads}");

    let thread_counts = vec![1, 2, 4, 8, 9, 10, 11, 12, 13, 16, 32, 64, 128, 256, 512];

    for &num_threads in &thread_counts {
        let games_counter = Arc::new(AtomicU64::new(0));

        let start_time = Instant::now();
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let games_counter = Arc::clone(&games_counter);
                thread::spawn(move || play_games_for(test_duration, &games_counter))
            })
            .collect();

        for handle in handles {
            handle.join().expect("thread panicked");
        }
        let elapsed = start_time.elapsed();

        let total_games = games_counter.load(Ordering::Relaxed);
        let games_per_sec = total_games as f64 / elapsed.as_secs_f64();

        println!(
            "Threads: {num_threads}, Games: {total_games}, Time: {elapsed:?}, Games/s: {games_per_sec:.2}"
        );
    }
}
