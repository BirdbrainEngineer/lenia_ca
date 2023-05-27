#![allow(dead_code)]
#![allow(unused_variables)]

/// Collection of generators for some of the common Lenia kernels. 

use super::*;
use ndarray::IxDyn;

/// Creates the kernel base for a gaussian donut in 2d. 
/// The mean (position of the highest value) is placed at `0.5`
/// in the range `[0.0..1.0]`, where `0.0` is the center of the kernel and `1.0` the outer edge.
/// 
/// 
/// ### Parameters
/// 
/// * `radius` - The radius of the kernel. The kernel is guaranteed to be square in shape,
/// but any values outside the radius are set to `0.0`.
/// 
/// * `stddev` - Standard deviation to use. 
pub fn gaussian_donut_2d(radius: usize, stddev: f64) -> ndarray::ArrayD<f64> {
    let diameter = radius * 2;
    let radius = radius as f64;
    let normalizer = 1.0 / radius;
    let mut out = ndarray::ArrayD::zeros(IxDyn(&[diameter, diameter]));
    let x0 = radius;
    let y0 = radius;
    for i in 0..out.shape()[0] {
        for j in 0..out.shape()[1] {
            let x1 = i as f64;
            let y1 = j as f64;
            let dist = ((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)).sqrt();
            if dist <= radius { 
                out[[i, j]] = super::sample_normal(dist * normalizer, 0.5, stddev);
            }
            else { 
                out[[i, j]] = 0.0
            }
        }
    }
    out
}

/// Create the base for a kernel made from multiple concentric gaussian "donuts" in 2d.
/// Each donut/ring is a single index in the list of parameters. 
/// 
/// ### Parameters
/// 
/// * `radius` - The radius of the kernel. The kernel is guaranteed to be square in shape.
/// but any values outside the radius are set to `0.0`.
/// 
/// * `means` - The placement of the peak values of individual rings. 
/// Should be in range `[0.0..1.0]`, where `0.0` is the center point of the kernel and
/// `1.0` is the outer edge of the circular kernel. 
/// 
/// * `peaks` - The maximum value that each individual ring can create. 
/// Can be any positive real number but will later be normalized compared to other rings.
/// 
/// * `stddevs` - The standard deviations of each individual ring.
pub fn multi_gaussian_donut_2d(radius: usize, means: &[f64], peaks: &[f64], stddevs: &[f64]) -> ndarray::ArrayD<f64> {
    if means.len() != peaks.len() || means.len() != stddevs.len() {
        panic!("Function \"multi_gaussian_donut_2d\" expects each mean parameter to be accompanied by a peak and stddev parameter!");
    }
    let diameter = radius * 2;
    let radius = radius as f64;
    let normalizer = 1.0 / radius;
    let mut out = ndarray::ArrayD::zeros(IxDyn(&[diameter, diameter]));
    let x0 = radius;
    let y0 = radius;
    for i in 0..out.shape()[0] {
        for j in 0..out.shape()[1] {
            let x1 = i as f64;
            let y1 = j as f64;
            let dist = ((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)).sqrt();
            if dist <= radius { 
                let mut sum = 0.0;
                for i in 0..means.len() {
                    sum += super::sample_normal(dist * normalizer, means[i], stddevs[i]) * peaks[i].abs();
                }
                out[[i, j]] = sum;
            }
            else { 
                out[[i, j]] = 0.0
            }
        }
    }
    out
}

/// Creates the kernel base for a gaussian donut in n-dimensions. 
/// The mean (position of the highest value) is placed at `0.5`
/// in the range `[0.0..1.0]`, where `0.0` is the center of the kernel and `1.0` the outer edge.
/// 
/// ### Parameters
/// 
/// * `radius` - The radius of the kernel in every axis. 
/// Any values outside the radius are set to `0.0`.
/// 
/// * `stddev` - Standard deviation to use. 
pub fn gaussian_donut_nd(radius: usize, dimensions: usize, stddev: f64) -> ndarray::ArrayD<f64> {
    let radius = radius as f64;
    let normalizer = 1.0 / radius;
    let center = vec![radius; dimensions];
    let mut shape: Vec<usize> = Vec::new();
    let mut index: Vec<f64> = Vec::new();
    for i in 0..dimensions {
        shape.push((radius * 2.0) as usize);
        index.push(0.0);
    }
    let out = ndarray::ArrayD::from_shape_fn(
        shape, 
        |index_info| {
            for i in 0..index.len() {
                index[i] = index_info[i] as f64;
            }
            let dist = euclidean_dist(&index, &center);
            if dist > radius {
                0.0
            }
            else {
                sample_normal(dist * normalizer, 0.5, stddev)
            }
        }
    );
    out
}

/// Create the base for a kernel made from multiple radial gaussian "hyper-donuts" in n dimensions.
/// Each donut/ring is a single index in the list of parameters. 
/// 
/// ### Parameters
/// 
/// * `radius` - The radius of the kernel in each axis.
/// Any values outside the radius are set to `0.0`.
/// 
/// * `means` - The placement of the peak values of individual donuts
/// Should be in range `[0.0..1.0]`, where `0.0` is the center point of the kernel and
/// `1.0` is the outer surface of the hypersphere. 
/// 
/// * `peaks` - The maximum value that each individual donut can create. 
/// Can be any positive real number but will later be normalized compared to other donuts.
/// 
/// * `stddevs` - The standard deviations of each individual donut.
pub fn multi_gaussian_donut_nd(radius: usize, dimensions: usize, means: &[f64], peaks: &[f64], stddevs: &[f64]) -> ndarray::ArrayD<f64> {
    let radius = radius as f64;
    let normalizer = 1.0 / radius;
    let center = vec![radius; dimensions];
    let mut shape: Vec<usize> = Vec::new();
    let mut index: Vec<f64> = Vec::new();
    for i in 0..dimensions {
        shape.push((radius * 2.0) as usize);
        index.push(0.0);
    }
    let out = ndarray::ArrayD::from_shape_fn(
        shape, 
        |index_info| {
            for i in 0..index.len() {
                index[i] = index_info[i] as f64;
            }
            let dist = euclidean_dist(&index, &center);
            if dist > radius {
                0.0
            }
            else {
                let mut sum = 0.0;
                for i in 0..means.len() {
                    sum += super::sample_normal(dist * normalizer, means[i], stddevs[i]) * peaks[i].abs();
                }
                sum
            }
        }
    );
    out
}

/// Creates a radially symmetric kernel. 
/// 
/// ### Parameters
/// 
/// * `radius` - Radius of the kernel field to generate.
/// 
/// * `dimensions` - dimensionality of the kernel field to generate.
/// 
/// * `params[0..n]` - Value to set based on the distance from the center of the kernel
/// to the outer edge of the kernel, where `params[0]` is the value at the kernel center
/// and `params[1]` is the value at the edge of the kernel. 
pub fn precalculated_linear(radius: usize, dimensions: usize, params: &[f64]) -> ndarray::ArrayD<f64> {
    let radius = radius as f64;
    let normalizer = 1.0 / radius;
    let center = vec![radius; dimensions];
    let mut shape: Vec<usize> = Vec::new();
    let mut index: Vec<f64> = Vec::new();
    for i in 0..dimensions {
        shape.push((radius * 2.0) as usize);
        index.push(0.0);
    }
    let out = ndarray::ArrayD::from_shape_fn(
        shape, 
        |index_info| {
            for i in 0..index.len() {
                index[i] = index_info[i] as f64;
            }
            let dist = euclidean_dist(&index, &center);
            if dist > radius {
                0.0
            }
            else {
                growth_functions::precalculated_linear(dist * normalizer, params)
            }
        }
    );
    out
}

/// Creates the kernel base for "polynomial donuts". The peaks of the individual rings are equally spaced
/// around the center of the kernel. 
/// 
/// ### Parameters
/// 
/// * `params[0]` - Polynomial power, usually set to `4.0`;
/// 
/// * `params[1..n]` - Peak heights of the individual donuts.
pub fn polynomial_nd(radius: usize, dimensions: usize, params: &[f64]) -> ndarray::ArrayD<f64> {
    let radius = radius as f64;
    let normalizer = 1.0 / radius;
    let center = vec![radius; dimensions];
    let mut shape: Vec<usize> = Vec::new();
    let mut index: Vec<f64> = Vec::new();
    for i in 0..dimensions {
        shape.push((radius * 2.0) as usize);
        index.push(0.0);
    }
    let out = ndarray::ArrayD::from_shape_fn(
        shape, 
        |index_info| {
            for i in 0..index.len() {
                index[i] = index_info[i] as f64;
            }
            let dist = euclidean_dist(&index, &center);
            if dist > radius {
                0.0
            }
            else {
                let dist = dist * normalizer;
                if dist == 0.0 { 0.0 }
                else {
                    let peak_index = (dist * (params.len() - 1) as f64).ceil() as usize;
                    params[peak_index] * c((params.len() - 1) as f64 * dist - (peak_index - 1) as f64, params[0])
                }
            }
        }
    );
    out
}

fn c(r: f64, alpha: f64) -> f64 {
    let num = 4.0 * r * (1.0 - r);
    let mut out = 1.0;
    for _ in 0..(alpha as usize) {
        out *= num;
    }
    out
}

/// Moore neighborhood with radius of 1 in 2D. 
/// 
/// This is the kernel to use for Conway's game of life. 
pub fn conway_game_of_life() -> ndarray::ArrayD<f64> {
    let mut out = ndarray::ArrayD::from_elem(vec![3 as usize, 3], 1.0);
    out[[1, 1]] = 0.0;
    out
}

/// Kernel for "SmoothLife" cellular automaton in any dimensionality. Not completely faithful to SmoothLife.
/// 
/// ### Parameters
/// 
/// * `radius` - Radius of the kernel to generate.
/// 
/// * `dimensions` - Dimensionality of the kernel to generate.
/// 
/// * `width_ratio` - Controls the width of the neighborhood ring around the center, where `0.0` is empty kernel
/// and `1.0` is a completely filled in disk. Use `0.5` for default SmoothLife. 
pub fn smoothlife(radius: usize, dimensions: usize, width_ratio: f64) -> ndarray::ArrayD<f64> {
    let width_ratio = width_ratio.clamp(0.0, 1.0);
    let center = vec![radius as f64; dimensions];
    let shape = vec![radius * 2; dimensions];
    let out = ndarray::ArrayD::from_shape_fn(shape, |index_info| {
        let mut index = Vec::with_capacity(dimensions);
        for i in 0..dimensions {
            index.push(index_info[i] as f64);
        }
        let mut dist = euclidean_dist(&index, &center);
        dist /= radius as f64;
        if dist > (1.0 - ((1.0 - width_ratio) * 0.5)) { 0.0 }
        else if dist < ((1.0 - width_ratio) * 0.5) { 0.0 }
        else { 1.0 }
    });
    out
}

/// Gives a kernel the size of 1 unit containing `1.0`, but with as many dimensions as `shape`.
pub fn pass(shape: &[usize]) -> ndarray::ArrayD<f64> {
    let mut unit_shape: Vec<usize> = Vec::new();
    for _ in shape {
        unit_shape.push(1);
    }
    ndarray::ArrayD::<f64>::from_shape_fn(unit_shape, |a| { 1.0 })
}

pub fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut out: f64 = 0.0;
    for i in 0..a.len() {
        out += (a[i] - b[i]) * (a[i] - b[i]);
    }
    out.sqrt()
}