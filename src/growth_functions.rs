//! A collection of often used growth functions.

#![allow(dead_code)]
#![allow(unused_variables)]

/// Standard unimodal "gaussian bump".
/// 
/// ### Parameters
/// 
/// * `params[0]` - **mu**: The position of the mean / highest point of the growth function.
/// 
/// * `params[1]` - **sigma**: Standard deviation of the gaussian bump. 
/// 
/// ### Returns
/// A `f64` in range `[-1.0..1,0]`. 
pub fn standard_lenia(num: f64, params: &[f64]) -> f64 {
    (2.0 * super::sample_normal(num, params[0], params[1])) - 1.0
}

/// Multimodal "gaussian bumps" growth function.
/// 
/// While the Lenia paper calls for a unimodal growth function, then strictly speaking, there are no rules!
/// Do whatever you want.
/// 
/// ### Parameters
/// 
/// * `params[even index]` - **mu**: The position of the means / the centers of the gaussian bumps.
/// 
/// * `params[odd index]` - **sigma**: Standard deviations of the gaussian bumps. Each sigma corresponds
/// to the mu defined by the previous `params` index.
pub fn multimodal_normal(num: f64, params: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in (0..params.len()).step_by(2) {
        sum += super::sample_normal(num, params[i], params[i + 1]);
    }
    (sum * 2.0) - 1.0
}

/// Standard unimodal "polynomial bump".
/// 
/// ### Parameters
/// 
/// `params[0]` - mu
/// 
/// `params[1]` - sigma
/// 
/// `params[2]` - alpha
pub fn polynomial(num: f64, params: &[f64]) -> f64 {
    let l = (num - params[0]).abs();
    let k = params[1] * 3.0;
    if l > k { -1.0 }
    else {
        let a = 1.0 - ((l * l) / (k * k));
        let mut out = 1.0;
        for _ in 0..(params[2] as usize) {
            out *= a;
        }
        (out * 2.0) - 1.0
    }
}

/// Samples from a precalculated distribution.
/// 
/// The distribution is made of evenly spaced points from
/// `0.0` to `1.0`. In the likely event of the sample falling between 2 points in the distribution, 
/// the result will be interpolated linearly between the two points.
/// 
/// ### Parameters
/// 
/// * `params[0..n]` - Distribution in range `[0.0..1.0]` to sample from
pub fn precalculated_linear(num: f64, params: &[f64]) -> f64 {
    let index = num * params.len() as f64;
    if index as usize >= (params.len() - 1) { return params[params.len() - 1] }
    if index as usize <= 0 { return params[0] }
    let a = params[index.abs().floor() as usize];
    let b = params[index.abs().ceil() as usize];
    let dx = index - index.floor();
    let dy = b - a;
    a + (dx * dy)
}

/// Conway's "Game of life" growth function. `Rulestring: B3/S23`
pub fn conway_game_of_life(num: f64, params: &[f64]) -> f64 {
    let index = (num * 9.0).round() as usize;
    if index == 2 { 0.0 }
    else if index == 3 { 1.0 }
    else {-1.0 }
}

/// Basic Smooth Life growth function.
/// 
/// Not faithful to proper SmoothLife, and is not capable of simulating every SmoothLife.
/// 
/// `params[0]` - Birth range start
/// 
/// `params[1]` - Birth range end
/// 
/// `params[2]` - Survive range start
/// 
/// `params[3]` - Survive range end
pub fn smooth_life(num: f64, params: &[f64]) -> f64 {
    if num >= params[0] && num <= params[1] { return 1.0 }
    if num >= params[2] && num <= params[3] { return 0.0 }
    -1.0
}

/// Smooth Life growth function with smoothed stepping. 
/// 
/// Not faithful to proper SmoothLife and is not capable of simulating every SmoothLife.
/// 
/// Step width defines the range within which `~99%` of the change between states takes place.
/// 
/// ### Parameters
/// 
/// `params[0]` - Birth range start
/// 
/// `params[1]` - Birth range end
/// 
/// `params[2]` - Survive range start
/// 
/// `params[3]` - Survive range end
/// 
/// `params[4]` - Birth step width
/// 
/// `params[5]` - Survive step width
pub fn smooth_life_sigmoid_smoothed(num: f64, params: &[f64]) -> f64 {
    let birth = (sigmoid(num, params[0], params[4], 2.0) + 
        sigmoid(num, params[1], -params[4], 2.0)) - 3.0;
    let survive = (sigmoid(num, params[2], params[5], 1.0) + 
        sigmoid(num, params[3], -params[5], 1.0)) - 2.0;
    if birth > survive { birth } else { survive }
}

/// Sigmoid function.
fn sigmoid(x: f64, center: f64, sigma: f64, peak: f64) -> f64 {
    peak / (1.0 + (-((x - center) * (4.0 / sigma))).exp())
}

/// Pass number on virtually unchanged.
/// 
/// Returns `num` multiplied by `params[0]`. Use this growth function if you would like to not use a growth function, 
/// but merely explore the dynamics of iterative application of kernels.
pub fn pass(num: f64, params: &[f64]) -> f64 {
    num * params[0]
}