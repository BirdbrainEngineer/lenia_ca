use ndarray::{self, Zip, IntoNdProducer};
use num_complex::Complex;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use super::*;
use super::fft::{PlannedFFTND, ParPlannedFFTND};

/// `SimpleLenia` struct implements the non-extended Lenia system with a 2d field and 
/// pre-set parameters to facilitate the creation of the
/// ***Orbium unicaudatus*** glider - hallmark of the Lenia system.
/// 
/// This version of Lenia does not allow for adding extra channels nor convolution channels. 
/// In addition, channel weights are not available for this version of Lenia.
/// 
/// Changeable parameters include the timestep a.k.a. integration step **dt**, 
/// the **growth function**, and the **kernel** given that the kernel is 2-dimensional. 
pub struct StandardLenia {
    dt: f64,
    channel: Channel,
    shape: Vec<usize>,
    conv_channel: ConvolutionChannel,
    convolved: ndarray::ArrayD<Complex<f64>>,
    forward_fft_instance: fft::ParPlannedFFTND,
    inverse_fft_instance: fft::ParPlannedFFTND,
}

impl StandardLenia {
    const DEFAULT_KERNEL_SIZE: usize = 28;
}

impl Lenia for StandardLenia {
    /// Create and initialize a new instance of "Standard Lenia" in 2D. This version of Lenia
    /// can have only a single channel and a single convolution channel. It also does not
    /// support any weights, as it can be encoded within the `dt` parameter. 
    /// 
    /// The size of either dimension may not be smaller than 28 pixels. 
    /// 
    /// By default the kernel, growth function and dt parameter are set such that when 
    /// simulating, the simulation is capable of producing the ***Orbium unicaudatus*** glider.
    /// 
    /// ### Parameters
    /// 
    /// * `shape` - Reference to the shape that the channels in the `Lenia` instance shall have.
    /// 
    /// ### Panics
    /// 
    /// * If the length of `shape` is not `2`.
    /// 
    /// * If either of the axis lengths in `shape` are `<28`.
    fn new(shape: &[usize]) -> Self {
        if shape.len() < 2 || shape.len() > 2 { 
            panic!("StandardLenia2D(new): Expected 2 dimensions for Standard Lenia! Found {}.", shape.len()); 
        }
        for (i, dim) in shape.iter().enumerate() {
            if *dim < Self::DEFAULT_KERNEL_SIZE {
                panic!("StandardLenia::new() - Axis {} is extremely small ({} pixels). Minimum size of each axis for 2D Standard Lenia is {} pixels.", i, *dim, Self::DEFAULT_KERNEL_SIZE);
            }
        }
        let kernel = Kernel::from(
            kernels::gaussian_donut_2d(
                Self::DEFAULT_KERNEL_SIZE, 
                1.0/6.7
            ), 
            shape
        );

        let conv_channel = ConvolutionChannel {
            input_channel: 0,
            kernel: kernel,
            field: ndarray::ArrayD::from_elem(shape, 0.0),
            growth: growth_functions::standard_lenia,
            growth_params: vec![0.15, 0.017],
        };

        let channel = Channel {
            field: ndarray::ArrayD::from_elem(shape, 0.0),
            weights: vec![1.0],
            weight_sum_reciprocal: 1.0,
        };
        
        StandardLenia{
            forward_fft_instance: fft::ParPlannedFFTND::new(shape, false),
            inverse_fft_instance: fft::ParPlannedFFTND::new(shape, true),
            dt: 0.1,
            channel: channel,
            shape: shape.to_vec(),
            conv_channel: conv_channel,
            convolved: ndarray::ArrayD::from_elem(shape, Complex::new(0.0, 0.0)),
        }
    }

    fn iterate(&mut self) {
        self.convolved.zip_mut_with(
            &self.channel.field, 
            |a, b| {
                a.re = *b;
                a.im = 0.0;
            }
        );
        self.forward_fft_instance.transform(&mut self.convolved);
        
        self.convolved.zip_mut_with(
            &self.conv_channel.kernel.transformed, 
            |a, b| {
                // Complex multiplication without cloning
                let real = (a.re * b.re) - (a.im * b.im);
                a.im = ((a.re + a.im) * (b.re + b.im)) - real;
                a.re = real;
            }
        );

        self.inverse_fft_instance.transform(&mut self.convolved);

        self.conv_channel.field.zip_mut_with(
            &self.convolved, 
            |a, b| {
                *a = (self.conv_channel.growth)(b.re, &self.conv_channel.growth_params);
            }
        );

        self.channel.field.zip_mut_with(&self.conv_channel.field, |a, b| { *a = (*a + (*b * self.dt)).clamp(0.0, 1.0); })
    }

    fn set_channels(&mut self, num_channels: usize) {
        println!("Changing the number of channels is not available for Standard Lenia! Try using a different Lenia instead.");
    }

    fn set_conv_channels(&mut self, num_conv_channels: usize) {
        println!("Changing the number of channels is not available for Standard Lenia! Try using a different Lenia instead.");
    }

    fn set_source_channel(&mut self, conv_channel: usize, src_channel: usize) {
        println!("Adding or changing source channels is not available for Standard Lenia! Try using a different Lenia instead.");
    }

    fn set_weights(&mut self, new_weights: &[f64], conv_channel: usize) {
        println!("Adding or changing convolution output weights is not available for Standard Lenia! Try using a different Lenia instead.");
    }

    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize) {
        self.conv_channel.kernel = Kernel::from(kernel, self.channel.field.shape());
    }

    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize) {
        self.conv_channel.growth = f;
        self.conv_channel.growth_params = growth_params;
    }

    fn set_dt(&mut self, new_dt: f64) {
        self.dt = new_dt;
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn get_channel_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64> {
        &self.channel.field
    }

    fn get_kernel_as_ref(&self, conv_channel: usize) -> &Kernel {
        &self.conv_channel.kernel
    }

    fn get_channel_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<f64> {
        &mut self.channel.field
    }

    fn get_convoluted_as_ref(&self, conv_channel: usize) -> &ndarray::ArrayD<Complex<f64>> {
        &self.convolved
    }

    fn get_grown_as_ref(&self, conv_channel: usize) -> &ndarray::ArrayD<f64> {
        &self.conv_channel.field
    }

    fn get_deltas_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64> {
        &self.conv_channel.field    // Same as growth result because weights are not available for Standard Lenia
    }

    fn dt(&self) -> f64 {
        self.dt
    }

    fn channels(&self) -> usize {
        1 as usize
    }

    fn conv_channels(&self) -> usize {
        1 as usize
    }

    fn weights(&self, channel: usize) -> &[f64] {
        &self.channel.weights
    }
}



/// `ExpandedLenia` struct implements the expanded Lenia system, with support for multiple n-dimensional
/// channels, multiple kernels & associated growth functions (convolution channels) and weights.  
pub struct ExpandedLenia {
    dt: f64,
    channels: Vec<Channel>,
    deltas: Vec<ndarray::ArrayD<f64>>,
    shape: Vec<usize>,
    conv_channels: Vec<ConvolutionChannel>,
    convolutions: Vec<ndarray::ArrayD<Complex<f64>>>,
    forward_fft_instances: Vec<fft::ParPlannedFFTND>,
    inverse_fft_instances: Vec<fft::ParPlannedFFTND>,
}

impl Lenia for ExpandedLenia {
    /// Create and initialize a new instance of "ExpandedLenia`. 
    fn new(shape: &[usize]) -> Self {
        let kernel = Kernel::from(kernels::pass(shape), shape);
        
        let conv_channel = ConvolutionChannel {
            input_channel: 0,
            kernel: kernel,
            field: ndarray::ArrayD::from_elem(shape, 0.0),
            growth: growth_functions::pass,
            growth_params: vec![1.0],
        };

        let channel = Channel {
            field: ndarray::ArrayD::from_elem(shape, 0.0),
            weights: vec![1.0],
            weight_sum_reciprocal: 1.0,
        };

        let mut channel_shape = Vec::new();
        for dim in shape {
            channel_shape.push(*dim);
        }
        
        ExpandedLenia{
            forward_fft_instances: vec![fft::ParPlannedFFTND::new(&channel_shape, false)],
            inverse_fft_instances: vec![fft::ParPlannedFFTND::new(&channel_shape, true)],
            dt: 0.1,
            channels: vec![channel],
            deltas: vec![ndarray::ArrayD::from_elem(shape, 0.0)],
            conv_channels: vec![conv_channel],
            convolutions: vec![ndarray::ArrayD::from_elem(shape, Complex::new(0.0, 0.0))],
            shape: shape.to_vec(),
        }
    }

    // This is a very long and complex function, sorry.
    // It uses concurrency to calculate multiple convolutions at the same time, as well as
    // apply weights and sum the results. 
    fn iterate(&mut self) {
        let mut axes: Vec<usize> = Vec::with_capacity(self.shape.len());
        let mut inverse_axes: Vec<usize> = Vec::with_capacity(self.shape.len());
        for i in 0..self.shape.len() {
            axes.push(i);
        }
        for i in (0..self.shape.len()).rev() {
            inverse_axes.push(i);
        }

        //Create mutexes and rwlocks
        let mut sources: Vec<usize> = Vec::with_capacity(self.conv_channels.len());
        let mut channel_rwlocks: Vec<Arc<RwLock<Channel>>> = Vec::with_capacity(self.channels.len());
        let mut delta_rwlocks: Vec<Arc<RwLock<ndarray::ArrayD<f64>>>> = Vec::with_capacity(self.deltas.len());
        let mut conv_channel_mutexes: Vec<Arc<Mutex<ConvolutionChannel>>> = Vec::with_capacity(self.conv_channels.len());
        let mut convolution_mutexes: Vec<Arc<Mutex<ndarray::ArrayD<Complex<f64>>>>> = Vec::with_capacity(self.convolutions.len());
        let mut forward_fft_mutexes: Vec<Arc<Mutex<ParPlannedFFTND>>> = Vec::with_capacity(self.forward_fft_instances.len());
        let mut inverse_fft_mutexes: Vec<Arc<Mutex<ParPlannedFFTND>>> = Vec::with_capacity(self.inverse_fft_instances.len());

        for _ in 0..self.channels.len() {
            channel_rwlocks.push(Arc::new(RwLock::new(self.channels.remove(0))));
            delta_rwlocks.push(Arc::new(RwLock::new(self.deltas.remove(0))));
        }
        for _ in 0..self.conv_channels.len() {
            sources.push(self.conv_channels[0].input_channel);
            conv_channel_mutexes.push(Arc::new(Mutex::new(self.conv_channels.remove(0))));
            convolution_mutexes.push(Arc::new(Mutex::new(self.convolutions.remove(0))));
            forward_fft_mutexes.push(Arc::new(Mutex::new(self.forward_fft_instances.remove(0))));
            inverse_fft_mutexes.push(Arc::new(Mutex::new(self.inverse_fft_instances.remove(0))));
        }

        // Concurrent convolutions
        let mut convolution_handles = Vec::with_capacity(conv_channel_mutexes.len());

        for i in 0..conv_channel_mutexes.len() {
            // Set up and aquire locks on data
            let axes_clone = axes.clone();
            let inverse_axes_clone = inverse_axes.clone();
            let source_lock = Arc::clone(&channel_rwlocks[sources[i]]);
            let delta_lock = Arc::clone(&delta_rwlocks[sources[i]]);
            let convolution_lock = Arc::clone(&convolution_mutexes[i]);
            let convolution_channel_lock = Arc::clone(&conv_channel_mutexes[i]);
            let forward_fft_lock = Arc::clone(&forward_fft_mutexes[i]);
            let inverse_fft_lock = Arc::clone(&inverse_fft_mutexes[i]);

            convolution_handles.push(thread::spawn(move || {
                let mut convolution_channel = convolution_channel_lock.lock().unwrap();
                let input = source_lock.read().unwrap();
                let delta = delta_lock.read().unwrap();
                let mut convolution = convolution_lock.lock().unwrap();
                let mut forward_fft = forward_fft_lock.lock().unwrap();
                let mut inverse_fft = inverse_fft_lock.lock().unwrap();
                // Get data from source channel
                convolution.zip_mut_with(
                    &input.field, 
                    |a, b| {
                        a.re = *b;
                        a.im = 0.0;
                    }
                );
                // Fourier-transform convolute
                // Forward fft the input data
                forward_fft.transform(&mut convolution);
                // Complex multiplication without cloning
                convolution.zip_mut_with(
                    &convolution_channel.kernel.transformed, 
                    |a, b| {
                        let real = (a.re * b.re) - (a.im * b.im);
                        a.im = ((a.re + a.im) * (b.re + b.im)) - real;
                        a.re = real;
                    }
                );
                // Inverse fft to get convolution result
                inverse_fft.transform(&mut convolution);
                // Apply growth function
                let growth_info = (convolution_channel.growth, convolution_channel.growth_params.clone());
                convolution_channel.field.zip_mut_with(
                    &convolution, 
                    |a, b| {
                        *a = (growth_info.0)(b.re, &growth_info.1);
                    }
                );
            }));
        }

        let mut summing_handles = Vec::with_capacity(channel_rwlocks.len());

        for handle in convolution_handles {
            handle.join().unwrap();
        }

        // Collapse convolution channel mutexes back into a single owned vector
        let mut convolution_channels: Vec<ConvolutionChannel> = Vec::with_capacity(conv_channel_mutexes.len());
        for i in 0..conv_channel_mutexes.len() {
            let data = conv_channel_mutexes.remove(0);
            convolution_channels.push(Arc::try_unwrap(data).unwrap().into_inner().unwrap());
        }

        // Concurrent summing of results
        // Make and aquire locks
        let convoluted_results_rwlock = Arc::new(RwLock::new(convolution_channels));

        for i in 0..channel_rwlocks.len() {
            let dt = self.dt.clone();
            let channel_lock = Arc::clone(&channel_rwlocks[i]);
            let delta_lock = Arc::clone(&delta_rwlocks[i]);
            let convoluted_results_lock = Arc::clone(&convoluted_results_rwlock);
            
            // Thread code
            summing_handles.push(thread::spawn(move || {
                let mut channel = channel_lock.write().unwrap();
                let mut deltas = delta_lock.write().unwrap();
                let convoluted_results = convoluted_results_lock.read().unwrap();

                let previous_deltas = deltas.clone();
                // Apply weighted sums and dt to get the delta to be added to channel
                for i in 0..channel.weights.len(){
                    deltas.zip_mut_with(&convoluted_results[i].field, 
                        |a, b| {
                            if i == 0 { *a = 0.0; }
                            *a += *b * channel.weights[i];
                            if i == channel.weights.len() { *a *= channel.weight_sum_reciprocal; }
                        }
                    );
                }
                // Add update channel and clamp
                let dt_reciprocal = 1.0 / dt;
                ndarray::Zip::from(&mut channel.field).and(&mut deltas.view_mut()).and(&previous_deltas).par_for_each(|a, b, c| {
                    let previous = *a;
                    *a = (previous + (*b * dt)).clamp(0.0, 1.0);
                });
            }));
        }

        for _ in 0..convolution_mutexes.len() {
            self.forward_fft_instances.push(Arc::try_unwrap(forward_fft_mutexes.remove(0)).unwrap().into_inner().unwrap());
            self.inverse_fft_instances.push(Arc::try_unwrap(inverse_fft_mutexes.remove(0)).unwrap().into_inner().unwrap());
            self.convolutions.push(Arc::try_unwrap(convolution_mutexes.remove(0)).unwrap().into_inner().unwrap());
        }

        for handle in summing_handles {
            handle.join().unwrap();
        }

        // Return ownership of all data back to Lenia instance
        self.conv_channels = Arc::try_unwrap(convoluted_results_rwlock).unwrap().into_inner().unwrap();

        for _ in 0..channel_rwlocks.len() {
            self.channels.push(Arc::try_unwrap(channel_rwlocks.remove(0)).unwrap().into_inner().unwrap());
            self.deltas.push(Arc::try_unwrap(delta_rwlocks.remove(0)).unwrap().into_inner().unwrap());
        }
        
    }

    fn set_channels(&mut self, num_channels: usize) {
        if num_channels <= self.channels.len() {
            for i in (num_channels..self.channels.len()).rev() {
                self.channels.remove(i);
                self.deltas.remove(i);
            }
        }
        else {
            let weights_prototype: Vec<f64> = vec![0.0; self.conv_channels.len()];
            for _ in self.channels.len()..num_channels {
                self.channels.push(
                    Channel { 
                        field: ndarray::ArrayD::from_elem(self.shape.clone(), 0.0),
                        weights: weights_prototype.clone(),
                        weight_sum_reciprocal: 0.0,
                    }
                );
                self.deltas.push(ndarray::ArrayD::from_elem(self.shape.clone(), 0.0));
            }
        }
    }

    fn set_conv_channels(&mut self, num_conv_channels: usize) {
        if num_conv_channels <= self.conv_channels.len() {
            for i in (num_conv_channels..self.conv_channels.len()).rev() {
                self.conv_channels.remove(i);
                self.forward_fft_instances.remove(i);
                self.inverse_fft_instances.remove(i);
                self.convolutions.remove(i);
            }
            for channel in &mut self.channels {
                for i in (num_conv_channels..channel.weights.len()).rev() {
                    channel.weights.remove(i);
                }
                let sum: f64 = channel.weights.iter().sum();
                channel.weight_sum_reciprocal = 1.0 / sum;
            }
        }
        else {
            for i in self.conv_channels.len()..num_conv_channels {
                self.conv_channels.push(
                    ConvolutionChannel { 
                        input_channel: 0, 
                        field: self.conv_channels[0].field.clone(), 
                        kernel: Kernel::from(kernels::pass(&self.shape), &self.shape), 
                        growth: growth_functions::pass, 
                        growth_params: vec![0.0],
                    }
                );
                self.forward_fft_instances.push(fft::ParPlannedFFTND::new(&self.shape, false));
                self.inverse_fft_instances.push(fft::ParPlannedFFTND::new(&self.shape, true));
                //self.forward_fft_instances.push(fft::PlannedParFFTND::new(&self.shape, false, 0));
                //self.inverse_fft_instances.push(fft::PlannedParFFTND::new(&self.shape, true, 0));
                self.convolutions.push(ndarray::ArrayD::from_elem(self.shape.clone(), Complex::new(0.0, 0.0)));
            }
            for channel in &mut self.channels {
                for _ in channel.weights.len()..num_conv_channels {
                    channel.weights.push(0.0);
                }
            }
        }
    }

    fn set_weights(&mut self, new_weights: &[f64], channel: usize) {
        let mut weights: Vec<f64>;
        if new_weights.len() < self.conv_channels.len() {
            weights = new_weights.clone().to_vec();
            for _ in new_weights.len()..self.conv_channels.len() {
                weights.push(0.0);
            }
        }
        else {
            weights = Vec::with_capacity(new_weights.len());
            for i in 0..self.conv_channels.len() {
                weights.push(new_weights[i]);
            }
        }
        let mut sum: f64 = 0.0;
        for weight in &weights {
            sum += weight.abs();
        }
        self.channels[channel].weights = weights;
        self.channels[channel].weight_sum_reciprocal = 1.0 / sum;
    }

    fn set_source_channel(&mut self, conv_channel: usize, src_channel: usize) {
        self.conv_channels[conv_channel].input_channel = src_channel;
    } 

    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize) {
        self.conv_channels[conv_channel].kernel = Kernel::from(kernel, &self.shape);
    } 

    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize) {
        self.conv_channels[conv_channel].growth = f;
        self.conv_channels[conv_channel].growth_params = growth_params;
    } 

    fn set_dt(&mut self, new_dt: f64) {
        self.dt = new_dt;
    } 

    fn shape(&self) -> &[usize] {
        &self.shape
    } 

    fn get_channel_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64> {
        &self.channels[channel].field
    }

    fn get_kernel_as_ref(&self, conv_channel: usize) -> &Kernel {
        &self.conv_channels[conv_channel].kernel
    }

    fn get_channel_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<f64> {
        &mut self.channels[channel].field
    }

    fn get_convoluted_as_ref(&self, conv_channel: usize) -> &ndarray::ArrayD<Complex<f64>> {
        &self.convolutions[conv_channel]
    }

    fn get_grown_as_ref(&self, conv_channel: usize) -> &ndarray::ArrayD<f64> {
        &self.conv_channels[conv_channel].field
    }

    fn get_deltas_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64> {
        &self.deltas[channel]
    }

    fn dt(&self) -> f64 {
        self.dt
    } 

    fn channels(&self) -> usize {
        self.channels.len()
    } 

    fn conv_channels(&self) -> usize {
        self.conv_channels.len()
    } 

    fn weights(&self, channel: usize) -> &[f64] {
        &self.channels[channel].weights
    } 
}


