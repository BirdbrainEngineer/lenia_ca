#![allow(dead_code)]
#![allow(unused_variables)]

use std::{fmt, sync::{Arc, Mutex}};
use rustfft::{Fft, FftNum, FftPlanner, FftDirection};
use rustfft::num_complex::Complex;
use rayon::prelude::*;


#[derive(Clone)]
pub struct PlannedFFT {
    fft: Arc<dyn Fft<f64>>,
    scratch_space: Vec<Complex<f64>>,
}

impl fmt::Debug for PlannedFFT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreplannedFFT")
         .field("scratch_space", &format!("Vec<Complex<f64>>, len: {}", self.scratch_space.len()))
         .field("fft", &format!("Arc<dyn rustfft::Fft<f64>> => len: {}, direction: {}", self.fft.len(), self.fft.fft_direction()))
         .finish()
    }
}

/*impl ndarray::IntoNdProducer for PlannedFFT {
    type Item = PlannedFFT;

    type Dim = ndarray::Ix1;

    type Output;

    fn into_producer(self) -> Self::Output {
        todo!()
    }
}*/

impl PlannedFFT {
    pub fn new(length: usize, inverse: bool) -> Self {
        if length == 0 { panic!("PlannedFFT::new() - Provided length was 0. Length must be at least 1!"); }
        let mut planner = FftPlanner::new();
        let direction: FftDirection;
        match inverse {
            true => { direction = FftDirection::Inverse; }
            false => { direction = FftDirection::Forward; }
        }
        let fft = planner.plan_fft(
            length, 
            direction,
        );
        let scratch_space: Vec<Complex<f64>> = Vec::from_iter(std::iter::repeat(Complex::new(0.0, 0.0)).take(fft.get_inplace_scratch_len()));

        PlannedFFT {
            fft: fft,
            scratch_space: scratch_space,
        }
    }

    pub fn inverse(&self) -> bool {
        match self.fft.fft_direction() {
            FftDirection::Forward => { return false; }
            FftDirection::Inverse => { return true; }
        }
    }

    pub fn length(&self) -> usize {
        self.fft.len()
    }

    pub fn transform(&mut self, data: &mut [Complex<f64>]) {
        self.fft.process_with_scratch(data, &mut self.scratch_space);
        if self.inverse() {     // I fekin' forgot this AGAIN...
            let inverse_len = 1.0 / data.len() as f64;
            for v in data.iter_mut() {
                v.re *= inverse_len;
                v.im *= inverse_len;
            }
        }
    }
}

#[derive(Debug)]
pub struct PlannedFFTND {
    shape: Vec<usize>,
    fft_instances: Vec<PlannedFFT>,
    inverse: bool
}

impl PlannedFFTND {
    pub fn new(shape: &[usize], inverse: bool) -> Self {
        if shape.is_empty() { panic!("PlannedFFTND::new() - Provided shape was empty! Needs at least 1 dimension!"); }
        /*let base_dim = shape[0];
        for dim in shape {
            if *dim != base_dim { panic!("PlannedFFTND::new() - Dimensions not the same length. Differing dimensions not implemented."); }
        }*/
        let mut ffts: Vec<PlannedFFT> = Vec::with_capacity(shape.len());
        for dim in shape {
            ffts.push(PlannedFFT::new(*dim, inverse));
        }
        PlannedFFTND {
            shape: shape.to_vec(),
            fft_instances: ffts,
            inverse: inverse,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn inverse(&self) -> bool {
        self.inverse
    }

    pub fn transform(&mut self, data: &mut ndarray::ArrayD<Complex<f64>>) {
        if data.shape() != self.shape { panic!("PlannedFFTND::transform() - shape of the data to be transformed does not agree with the shape that the fft can work on!"); }
        let mut axis_iterator: Vec<usize> = Vec::with_capacity(self.shape.len());
        if self.inverse() {
            for i in (0..self.shape.len()).rev() {
                axis_iterator.push(i);
            }
        }
        else {
            for i in 0..self.shape.len() {
                axis_iterator.push(i);
            }
        }
        for axis in axis_iterator {
            for mut lane in data.lanes_mut(ndarray::Axis(axis)) {
                let mut buf = lane.to_vec();
                self.fft_instances[axis].transform(&mut buf);
                for i in 0..lane.len() {
                    lane[i] = buf[i];
                }
            }
        }
    }
}



#[derive(Debug)]
pub struct ParPlannedFFTND {
    shape: Vec<usize>,
    fft_instances: Vec<PlannedFFT>,
    inverse: bool
}

impl ParPlannedFFTND {
    pub fn new(shape: &[usize], inverse: bool) -> Self {
        if shape.is_empty() { panic!("ParPlannedFFTND::new() - Provided shape was empty! Needs at least 1 dimension!"); }
        /*let base_dim = shape[0];
        for dim in shape {
            if *dim != base_dim { panic!("PlannedFFTND::new() - Dimensions not the same length. Differing dimensions not implemented."); }
        }*/
        let mut ffts: Vec<PlannedFFT> = Vec::with_capacity(shape.len());
        for dim in shape {
            ffts.push(PlannedFFT::new(*dim, inverse));
        }
        ParPlannedFFTND {
            shape: shape.to_vec(),
            fft_instances: ffts,
            inverse: inverse,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn inverse(&self) -> bool {
        self.inverse
    }

    pub fn transform(&mut self, data: &mut ndarray::ArrayD<Complex<f64>>) {
        if data.shape() != self.shape { panic!("ParPlannedFFTND::transform() - shape of the data to be transformed does not agree with the shape that the fft can work on!"); }
        let mut axis_iterator: Vec<usize> = Vec::with_capacity(self.shape.len());
        if self.inverse() {
            for i in (0..self.shape.len()).rev() {
                axis_iterator.push(i);
            }
        }
        else {
            for i in 0..self.shape.len() {
                axis_iterator.push(i);
            }
        }
        for axis in axis_iterator {
            let mut data_lane = data.lanes_mut(ndarray::Axis(axis));
            let mut fft_lane = &mut self.fft_instances[axis];
            ndarray::Zip::from(data_lane)
                .into_par_iter()
                .for_each_with(self.fft_instances[axis].clone(), |mut fft, mut row| {
                    let mut lane = row.0.to_vec();
                    fft.transform(&mut lane);
                    for i in 0..row.0.len() {
                        row.0[i] = lane[i];
                    }
                });
        }
    }
}



/* 
#[derive(Debug)]
pub struct PlannedParFFTND {
    shape: Vec<usize>,
    fft_instances: Vec<Vec<Arc<Mutex<PlannedFFT>>>>,
    inverse: bool,
    threads: usize,
}

impl PlannedParFFTND {
    pub fn new(shape: &[usize], inverse: bool, desired_threads: usize) -> Self {
        if shape.is_empty() { panic!("PlannedParFFTND::new() - Provided shape was empty! Needs at least 1 dimension!"); }
        let mut threads: usize; 
        if desired_threads == 0 { 
            threads = std::thread::available_parallelism().unwrap().get() as usize; 
        }
        else {
            threads = desired_threads;
        }
        for (i, dim) in shape.iter().enumerate() {
            if threads > *dim { 
                threads = *dim;
                println!("Warning: Axis {} too short for desired number of threads!", i);
             }
        }
        let mut ffts: Vec<Vec<Arc<Mutex<PlannedFFT>>>> = Vec::with_capacity(shape.len());
        for i in 0..shape.len() {
            let fft_prototype = PlannedFFT::new(shape[i], inverse);
            ffts.push(Vec::with_capacity(threads));
            for _ in 0..threads {
                ffts[i].push(Arc::new(Mutex::new(fft_prototype.clone())));
            }
        }
        PlannedParFFTND {
            shape: shape.to_vec(),
            fft_instances: ffts,
            inverse: inverse,
            threads: threads,
        }
    }

    pub fn adjust_threads(&mut self, ) {

    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn inverse(&self) -> bool {
        self.inverse
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn transform(&mut self, data: &mut ndarray::ArrayD<Complex<f64>>) {
        if data.shape() != self.shape { panic!("PlannedFFTND::transform() - shape of the data to be transformed does not agree with the shape that the fft can work on!"); }
        let mut axis_iterator: Vec<usize> = Vec::with_capacity(self.shape.len());
        if self.inverse() {
            for i in (0..self.shape.len()).rev() {
                axis_iterator.push(i);
            }
        }
        else {
            for i in 0..self.shape.len() {
                axis_iterator.push(i);
            }
        }
        for axis in axis_iterator {
            let mut fft_handles = VecDeque::with_capacity(self.threads);
            let mut buffers: VecDeque<Vec<Complex<f64>>> = VecDeque::new();
            for lane in data.lanes(ndarray::Axis(axis)) {
                buffers.push_back(lane.to_vec());
            }
            let num_lanes = buffers.len();
            
            for i in 0..self.threads {
                let fft_lock = Arc::clone(&self.fft_instances[axis][i]);
                let buffer = buffers.pop_front().unwrap();
                fft_handles.push_back(std::thread::spawn(move || {
                    let mut fft = fft_lock.lock().unwrap();
                    let mut buf = buffer; //buffer_lock.lock().unwrap();
                    fft.transform(&mut buf);
                    buf
                }));
            }
            let mut lanes = data.lanes_mut(ndarray::Axis(axis)).into_iter();

            for lane_index in 0..num_lanes {
                let mut lane = lanes.next().unwrap();
                let result = fft_handles.pop_front().unwrap().join().unwrap();
                for i in 0..lane.len() {
                    lane[i].re = result[i].re;
                    lane[i].im = result[i].im;
                }
                let thread_index = lane_index + self.threads;
                if thread_index < num_lanes {
                    let fft_lock = Arc::clone(&self.fft_instances[axis][thread_index % self.threads]);
                    let buffer = buffers.pop_front().unwrap();
                    fft_handles.push_back(std::thread::spawn(move || {
                        let mut fft = fft_lock.lock().unwrap();
                        let mut buf = buffer; //buffer_lock.lock().unwrap();
                        fft.transform(&mut buf);
                        buf
                    }));   
                }
            }
        }
    }
}*/