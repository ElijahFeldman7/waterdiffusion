use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use rand::Rng;

/// Rust implementation of the Levenshtein function
fn levenshtein(x: &[i64], y: &Vec<&Vec<f32>>, gamma: f32) -> f32 {
    let n = x.len();
    let m = y.len();

    let mut a = vec![vec![0.0; m + 1]; n + 1];

    for i in 0..=n {
        for j in 0..=m {
            if i == 0 {
                a[i][j] = j as f32 * gamma;
            } else if j == 0 {
                a[i][j] = i as f32 * gamma;
            } else {
                let cost = if x[i - 1] as usize >= y[j - 1].len() {
                    f32::INFINITY
                } else {
                    let value = y[j - 1][x[i - 1] as usize];
                    if value >= 1.0 {
                        f32::INFINITY
                    } else {
                        (1.0 - value).ln()
                    }
                };

                a[i][j] = (a[i - 1][j] + gamma)
                    .min(a[i][j - 1] + gamma)
                    .min(a[i - 1][j - 1] + cost);
            }
        }
    }

    a[n][m]
}

/// Rust implementation of the detect function
#[pyfunction]
fn detect(tokens: Vec<i64>, n: usize, k: usize, xi: Vec<Vec<f32>>, gamma: f32) -> f32 {
    let m = tokens.len();
    let mut a = vec![vec![0.0; n]; m - (k - 1)];

    for i in 0..(m - (k - 1)) {
        for j in 0..n {
            let xi_slice: Vec<&Vec<f32>> = (0..k)
                .map(|offset| &xi[(j + offset) % n])
                .collect();
            a[i][j] = levenshtein(&tokens[i..(i + k)], &xi_slice, gamma);
        }
    }

    a.into_iter()
        .flat_map(|row| row.into_iter())
        .reduce(f32::min)
        .unwrap_or(f32::INFINITY)
}

/// Rust implementation of the parallel permutation test
#[pyfunction]
fn permutation_test_parallel(
    tokens: Vec<i64>,
    n: usize,
    k: usize,
    vocab_size: usize,
    test_result: f32,
    n_runs: usize,
) -> f64 {
    let p_val: usize = (0..n_runs)
        .into_par_iter() // Parallel iterator using rayon
        .map(|_| {
            // Generate random xi_alternative
            let mut rng = rand::thread_rng();
            let xi_alternative: Vec<Vec<f32>> = (0..n)
                .map(|_| (0..vocab_size).map(|_| rng.gen::<f32>()).collect())
                .collect();

            // Calculate null_result
            let null_result = detect_internal(&tokens, n, k, &xi_alternative, 0.0);
            (null_result <= test_result) as usize
        })
        .sum();

    (p_val as f64 + 1.0) / (n_runs as f64 + 1.0)
}

/// Internal helper function for detect, used in the parallel test
fn detect_internal(tokens: &[i64], n: usize, k: usize, xi: &[Vec<f32>], gamma: f32) -> f32 {
    let m = tokens.len();
    let mut a = vec![vec![0.0; n]; m - (k - 1)];

    for i in 0..(m - (k - 1)) {
        for j in 0..n {
            let xi_slice: Vec<&Vec<f32>> = (0..k)
                .map(|offset| &xi[(j + offset) % n])
                .collect();
            a[i][j] = levenshtein(&tokens[i..(i + k)], &xi_slice, gamma);
        }
    }

    a.into_iter()
        .flat_map(|row| row.into_iter())
        .reduce(f32::min)
        .unwrap_or(f32::INFINITY)
}

#[pymodule]
fn levenshtein_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_test_parallel, m)?)?;
    Ok(())
}