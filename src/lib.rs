//! `Lenia_ca` is a crate that provides core functionality for simulating the Lenia system of cellular automata. The crate was made
//! as a programming excersize in making a large-ish Rust project. Since this was the first proper Rust project for the author, then
//! the crate has some weird quirks and inefficient... perhaps even illogical ways of structuring it. 
//! 
//! For now, the general way to use the crate is to import it like you would any other Rust crate, and then use the `Simulator` struct
//! essentially exclusively. You may also want to look into the `kernels` module and `growth_functions` module, as they contain a number
//! of useful generators and functions for Lenia systems. 

#![allow(dead_code)]
#![allow(unused_variables)]
#[cfg(target_has_atomic = "ptr")]

use std::fmt;
use std::{thread::JoinHandle};
use ndarray::{self, Axis, Slice, Order, Ix2};
use num_complex::Complex;
use png;
mod fft;
pub mod lenias;
pub mod kernels;
pub mod growth_functions;


trait SetBytes {
    fn set_low(&mut self, value: u8);
    fn set_high(&mut self, value: u8);
}

impl SetBytes for u16 {
    fn set_low(&mut self, value: u8) {
        *self &= !0xff;
        *self |= value as u16;
    }
    fn set_high(&mut self, value: u8) {
        *self &= !0xff00;
        *self |= (value as u16) << 8;
    }
}

trait GetBytes {
    fn get_low(&self) -> u8;
    fn get_high(&self) -> u8;
}

impl GetBytes for u16 {
    fn get_low(&self) -> u8 {
        (*self & 0xff) as u8
    }
    fn get_high(&self) -> u8 {
        ((*self & 0xff00) >> 8) as u8
    }
}

/// Samples the normal distribution where the peak (at `x = mu`) is 1.
/// This is not suitable for use as a gaussian probability density function!
/// 
/// ### Parameters
/// 
/// * `x` - Point of the normal distribution to sample.
/// 
/// * `mu` - The mean (point of the highest value/peak) of the normal distribution.
/// 
/// * `stddev` - Standard deviation of the normal distribution. 
fn sample_normal(x: f64, mu: f64, stddev: f64) -> f64 {
    (-(((x - mu) * (x - mu))/(2.0 * (stddev * stddev)))).exp()
}


fn sample_exponential(x: f64, exponent: f64, peak: f64) -> f64 {
    peak * (-(x * exponent)).exp()
}

/// Euclidean distance between points `a` and `b`. 
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut out: f64 = 0.0;
    for i in 0..a.len() {
        out += (a[i] - b[i]) * (a[i] - b[i]);
    }
    out.sqrt()
}

/// Extract data from n-dimensional array into a 2-dimensional array.
/// 
/// Extract a 2d array (`ndarray::Array2`) of `f64` values of a 2d slice of a channel's data. 
/// Use this to simply get a 2d frame for rendering. 
/// 
/// ### Parameters
/// 
/// * `input` - Channel data to extract the 2d frame from.
/// 
/// * `output` - 2D array into which to place the extracted frame.
/// 
/// * `display_axes` - Indexes of the axes to extract
/// 
/// * `dimensions` - Which indexes in any other axes the 2d slice is extracted from. 
/// The entries for axes selected in `display_axes` can be any number, and will be disregarded. 
pub fn get_frame(input: &ndarray::ArrayD<f64>, output: &mut ndarray::Array2<f64>, display_axes: &[usize; 2], dimensions: &[usize]) {
    if input.shape().len() == 2 {
        ndarray::Zip::from(output).and(input.view().into_dimensionality::<ndarray::Ix2>().unwrap()).par_for_each(|a, b| { *a = *b; });
        return;
    }
    let data = input.slice_each_axis(
        |a|{
            if a.axis.index() == display_axes[0] || a.axis.index() == display_axes[1] {
                return Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }
            }
            else {
                return Slice {
                    start: dimensions[a.axis.index()] as isize,
                    end: Some((dimensions[a.axis.index()] + 1) as isize),
                    step: 1,
                }
            }
        }
    );
    let data = data.to_shape(
        ((
            input.shape()[display_axes[0]],
            input.shape()[display_axes[1]]
        ), Order::RowMajor)
    ).unwrap();
    ndarray::Zip::from(output).and(&data).par_for_each(|a, b| { *a = *b; });
}

/// Loads a png into an `ndarray`. 
/// 
/// ### Parameters
/// 
/// * `file_path` - Path to the 2d slice of a frame to load. 
/// 
/// ### Panics
/// 
/// * If the bit-depth of the png is less than 8. 
/// 
/// * If the png has a color type different from Grayscale, Grayscale with alpha, RGB or RGBA. 
pub fn load_from_png(file_path: &str) -> ndarray::Array2<f64> {
    let decoder = png::Decoder::new(std::fs::File::open(file_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    if info.bit_depth != png::BitDepth::Eight && info.bit_depth != png::BitDepth::Sixteen {
        panic!("lenia_ca::load_from_png() - Unable to load from .png, as it has a bit depth of less than 8!");
    }
    let output: ndarray::Array2::<f64>;
    let offset: usize;
    match info.color_type {
        png::ColorType::Grayscale => {
            if info.bit_depth == png::BitDepth::Eight { offset = 1; }
            else { offset = 2; }
        }
        png::ColorType::GrayscaleAlpha => {
            if info.bit_depth == png::BitDepth::Eight { offset = 2; }
            else { offset = 4; }
        }
        png::ColorType::Rgb => { 
            if info.bit_depth == png::BitDepth::Eight { offset = 3; }
            else { offset = 6; }
        }
        png::ColorType::Rgba => { 
            if info.bit_depth == png::BitDepth::Eight { offset = 4; }
            else { offset = 8; }
        }
        _ => { panic!("lenia_ca::load_from_png() - Unsupported color type!"); }
    }
    let shape = [info.width as usize, info.height as usize];
    if info.bit_depth == png::BitDepth::Sixteen {
        output = ndarray::Array2::from_shape_fn(Ix2(shape[0], shape[1]), |a| {
            let mut num: u16 = 0;
            num.set_high(*buf.get((a.1 * info.width as usize * offset) + (a.0 * offset)).unwrap());
            num.set_low(*buf.get((a.1 * info.width as usize * offset) + (a.0 * offset + 1)).unwrap());
            num as f64 * (1.0 / 65535.0)
        });
    }
    else {
        output = ndarray::Array2::from_shape_fn(Ix2(shape[0], shape[1]), |a| {
            *buf.get((a.1 * info.width as usize * offset) + (a.0 * offset)).unwrap() as f64 * (1.0 / 255.0)
        });
    }
    output
}

/// Export a frame as a png or a bunch of png-s if multidimensional. 
/// 
/// ### Parameters
/// 
/// * `bit_depth` - Controls whether to output as 8-bit grayscale or 16-bit grayscale png.
/// 
/// * `frame` - Reference to the frame to be stored.
/// 
/// * `prefix` - Output file name. Numbers will be added after this string based on the 2d slice
/// of the frame (if exporting a 3d or higher dimensionality frame). **This prefix should also 
/// contain the frame number, if saving multiple successive frames.**
/// 
/// * `folder_path` - Folder path to where to save the frame at.
/// 
/// ### Panics
/// 
/// Under various circumstances, most commonly if the folder given by `folder_path` does not exist.
pub fn export_frame_as_png(bit_depth: png::BitDepth, frame: &ndarray::ArrayD<f64>, prefix: &str, folder_path: &str) -> JoinHandle<()>{
    if frame.shape().is_empty() { panic!("lenia_ca::export_frame_as_png() - Can not export an empty frame!") }

    let path_base = format!("{}{}{}",
        if folder_path.is_empty() { &"./" } else { folder_path.clone() }, 
        if folder_path.chars().last().unwrap() != '/' && folder_path.chars().last().unwrap() != '\\' { &"/" } else { &"" },
        prefix
    );
    let data;
    if frame.shape().len() == 1 {
        data = frame.to_shape((ndarray::IxDyn(&[frame.shape()[0], 1]), Order::RowMajor)).unwrap().mapv(|el| { el.clone() } );
    }
    else {
        data = frame.clone();
    }

    std::thread::spawn(move || {
        let mut indexes: Vec<usize> = vec![0; data.shape().len()];
        nested_png_export(bit_depth, path_base, &data, &mut indexes, 0);
    })
}

fn nested_png_export(bit_depth: png::BitDepth, path: String, data: &ndarray::ArrayD<f64>, indexes: &mut Vec<usize>, current_axis: usize) {
    if current_axis == (indexes.len() - 2) {
        let file_path = format!("{}.png", &path);
        println!("{}", &file_path);
        let file = std::fs::File::create(file_path).unwrap();
        let buf_writer = std::io::BufWriter::new(file);
        let width = data.shape()[data.shape().len()-2];
        let height = data.shape()[data.shape().len()-1];
        let mut encoder = png::Encoder::new(
            buf_writer, 
            width as u32, 
            height as u32
        );
        let mut image_data: Vec<u8> = Vec::with_capacity(width * height * if bit_depth == png::BitDepth::Eight {1} else {2});
        let image_data_buffer = data.slice_each_axis(
            |a|{
                if a.axis.index() == (indexes.len() - 2) || a.axis.index() == (indexes.len() - 1) {
                    return Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
                else {
                    return Slice {
                        start: indexes[a.axis.index()] as isize,
                        end: Some((indexes[a.axis.index()] + 1) as isize),
                        step: 1,
                    }
                }
            }
        )
        .to_shape(((width * height), Order::ColumnMajor))
        .unwrap()
        .mapv(|el| { el });

        for i in 0..(width * height) {
            if bit_depth == png::BitDepth::Eight {
                image_data.push((image_data_buffer[[i]] * 255.0) as u8);
            }
            else if bit_depth == png::BitDepth::Sixteen {
                let num = (image_data_buffer[[i]] * 65535.0) as u16;
                image_data.push(num.get_high());
                image_data.push(num.get_low());
            }
            else {
                panic!("lenia_ca::nested_png_export() - Unsupported bit depth!");
            }
        }
        encoder.set_depth(bit_depth);
        encoder.set_color(png::ColorType::Grayscale);
        let mut writer = encoder.write_header().unwrap();
        let write_result = writer.write_image_data(&image_data);
        match write_result {
            Ok(_) => {}
            Err(_) => { panic!("lenia_ca::nested_png_export() - Failed to write png!"); }
        }
    }
    else {
        for i in 0..data.shape()[current_axis] {
            indexes[current_axis] = i;
            nested_png_export(
                bit_depth,
                format!("{}_{}", &path, i), 
                data,
                indexes,
                current_axis + 1
            );
        }
    }
}

/// A Lenia simulation.
/// 
/// Container type for a `Lenia` implementation. It is not recommended to control the Lenia instance directly on your own. 
/// The Simulator has all the needed methods to control a Lenia instance in normal operation. 
pub struct Simulator<L: Lenia> {
    sim: L,
}

impl<L: Lenia> Simulator<L> {
    /// Initialize a Lenia simulator. 
    /// 
    /// Barring wanting to change the type of the `Lenia` instance used by the `Simulator`, 
    /// this should ever need to be called only once during the lifetime of your 
    /// Lenia simulation program.
    /// 
    /// ### Parameters
    /// 
    /// * `channel_shape` - The shape (number of dimensions and their lengths) of the
    /// channels for the `Lenia` instance. 
    /// 
    /// ### Panics
    /// 
    /// If any axis length in `channel_shape`is `0`.
    pub fn new(channel_shape: &[usize]) -> Self {
        for (i, dim) in channel_shape.iter().enumerate() {
            if *dim == 0 {
                panic!("Simulator::new() - Axis {} of the provided shape has a length of 0! Each axis must have a length of at least 1.", i);
            }
        }
        Simulator{
            sim: L::new(channel_shape),
        }
    }

    /// Re-initializes the `Lenia` instance, losing **all** of the previous changes, such as
    /// kernel changes, channel additions or any other parameter changes from the defaults
    /// of the specific `Lenia` instance implementation. 
    /// 
    /// Call this if the shape of the channels needs to be changed, or a major restructuring of
    /// channels and/or convolution channels is wanted.
    /// 
    /// ### Parameters
    /// 
    /// * `channel_shape` - The shape (number of dimensions and their lengths) of the
    /// channels for the `Lenia` instance.
    /// 
    /// ### Panics
    /// 
    /// If any axis length in `channel_shape`is `0`.
    pub fn remake(&mut self, channel_shape: &[usize]) {
        for (i, dim) in channel_shape.iter().enumerate() {
            if *dim == 0 {
                panic!("Simulator::new() - Axis {} of the provided shape has a length of 0! Each axis must have a length of at least 1.", i);
            }
        }
        self.sim = L::new(channel_shape);
    }

    /// Set the number of channels in the `Lenia` instance. 
    /// 
    /// **In case the number of channels
    /// is less than the current number of channels, it is up to the user to make sure that
    /// no convolution channel tries to use a dropped channel as its source!**
    /// 
    /// All values in newly created channels will be set to `0.0`.
    /// 
    /// The weights from all convolution channels into any newly created channels will start
    /// off at `0.0`.
    /// 
    /// ### Parameters
    /// 
    /// * `channels` - The number of channels the `Lenia` instance should have.
    /// 
    /// ### Panics
    /// 
    /// If `channels` is `0`.
    pub fn set_channels(&mut self, channels: usize) {
        if channels == 0 {
            panic!("Simulator::set_channels: Attempting to set the number of channels to 0. This is not allowed.");
        }
        if channels == self.sim.channels() { return; }
        self.sim.set_channels(channels);
    }

    /// Set the number of convolution channels in the `Lenia` instance. 
    /// 
    /// If the new number of 
    /// convolution channels is less than currently, then any convolution channels with an index
    /// higher than the new number of channels will be dropped, and their corresponding contribution
    /// to weighted averages for summing purged. 
    /// 
    /// If the new number of convolution channels is greater than currently then any new 
    /// convolution channels will need to have their kernels and growth functions set. In addition,
    /// channel weights for the new convolution channels will default to `0.0`. 
    /// 
    /// ### Parameters
    /// 
    /// * `convolution_channels` - The number of convolution channels the `Lenia` instance should have.
    /// 
    /// ### Panics
    /// 
    /// If `convolution_channels` is `0`.
    pub fn set_convolution_channels(&mut self, convolution_channels: usize) {
        if convolution_channels == 0 {
            panic!("Simulator::set_convolution_channels: Attempting to set the number of convolution channels to 0. This is not allowed.");
        }
        if convolution_channels == self.convolution_channels() { return; }
        self.sim.set_conv_channels(convolution_channels);
    }

    /// Set the source channel a given convolution channel should act on. 
    /// 
    /// ### Parameters
    /// 
    /// * `convolution_channel` - The convolution channel which will have its source changed.
    /// 
    /// * `source_channel` - The channel that the convolution channel should use as its source
    /// for convoluting.
    /// 
    /// ### Panics
    /// 
    /// * If the specified `convolution_channel` does not exist.
    /// 
    /// * If the specified `source_channel` does not exist.
    pub fn set_convolution_channel_source(&mut self, convolution_channel: usize, source_channel: usize) {
        if convolution_channel >= self.sim.conv_channels() {
            panic!("Simulator::set_convolution_channel_source: Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        if source_channel >= self.sim.channels() {
            panic!("Simulator::set_convolution_channel_source: Specified channel (index {}) does not exist. Current number of channels: {}.", source_channel, self.sim.channels());
        }
        self.sim.set_source_channel(convolution_channel, source_channel);
    }

    /// Set the kernel of the specified convolution channel. 
    /// 
    /// ### Parameters
    /// 
    /// * `kernel` - n-dimensional array (`ndarray::ArrayD<f64>`), where the number of 
    /// dimensions / axes must match the number of dimensions / axes of the channels of the
    /// `Lenia` instance. 
    /// 
    /// * `convolution_channel` - The convolution channel to which the new kernel is to be assigned.
    /// 
    /// ### Panics
    /// 
    /// If the specified `convolution_channel` does not exist. 
    /// 
    /// If the dimensionality of the kernel is not the same as the channels'
    pub fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, convolution_channel: usize) {
        if convolution_channel >= self.sim.conv_channels() {
            panic!("Simulator::set_kernel: Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        if kernel.shape().len() != self.sim.shape().len() { 
            panic!("Simulator::set_kernel: Number of kernel dimensionality ({}) does not agree with channels' dimensionality ({}).", kernel.shape().len(), self.sim.shape().len());
        }
        self.sim.set_kernel(kernel, convolution_channel);
    }

    /// Set the growth function and its parameters of the specified convolution channel.
    /// 
    /// ### Parameters
    /// 
    /// * `f` - Growth function to use.
    /// 
    /// * `growth_parameters` - The parameters passed to the growth function. 
    /// 
    /// * `convolution_channel` - The convoltution channel to which the new growth function and
    /// parameters are to be assigned.
    /// 
    /// ### Panics
    /// 
    /// If the specified `convolution_channel` does not exist. 
    pub fn set_growth_function(&mut self, f: fn(f64, &[f64]) -> f64, growth_parameters: Vec<f64>, convolution_channel: usize) {
        if convolution_channel >= self.sim.conv_channels() {
            panic!("Simulator::set_growth_function: Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        self.sim.set_growth(f, growth_parameters, convolution_channel);
    }

    /// Set the convolution channel weights for a specific channel.
    /// 
    /// * If the length of weights is greater than the number of convolution channels, 
    /// then the spare weights will be ignored.
    /// 
    /// * If the length of weights is less than the number of convolution channels, 
    /// then the missing weights will default to `0.0`.
    /// 
    /// ### Parameters
    /// 
    /// * `channel` - The channel, which the new weights will be assigned to.
    /// 
    /// * `weights` - The weights to assign. Index in the array corresponds to
    /// the index of the convoution channel. 
    pub fn set_weights(&mut self, channel: usize, weights: &[f64]) {
        self.sim.set_weights(weights, channel);
    }

    /// Set the integration step (a.k.a. timestep) parameter `dt` of the `Lenia` instance.
    /// 
    /// ### Parameters
    /// 
    /// * `dt` - The new dt value for the `Lenia` instance to use.
    pub fn set_dt(&mut self, dt: f64) {
        self.sim.set_dt(dt);
    }

    /// Performs a single iteration of the `Lenia` instance. Channels are updated with
    /// the resulting new state of the simulation. 
    pub fn iterate(&mut self) {
        self.sim.iterate();
    }

    /// Fills a channel with user data. The shapes of the `data` and the channel(s) in the
    /// `Lenia` instance must be the same. 
    /// 
    /// ### Parameters
    /// 
    /// * `data` - Reference to the n-dimensional array (`ndarray::ArrayD`) of `f64` values
    /// from which to fill the channel's data.
    /// 
    /// * `channel` - Index of the channel to fill. 
    ///
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist. 
    pub fn fill_channel(&mut self, data: &ndarray::ArrayD<f64>, channel: usize) {
        if channel >= self.sim.channels() {
            panic!("Simulator::fill_channel: Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        let channel_data = self.sim.get_channel_as_mut_ref(channel);
        channel_data.zip_mut_with(data, 
            |a, b| {
                *a = *b;
            }
        );
    }

    /// Retrieve a referenced to the specified channel's data. 
    /// 
    /// ### Parameters
    /// 
    /// * `channel` - Index of the channel to get a reference from.
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist. 
    pub fn get_channel_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64> {
        if channel >= self.sim.channels() {
            panic!("Simulator::get_channel_data_as_ref: Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        self.sim.get_channel_as_ref(channel)
    }

    /// Mutable version of `get_channel_as_ref()`.
    pub fn get_channel_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<f64> {
        if channel >= self.sim.channels() {
            panic!("Simulator::get_channel_data_as_ref() - Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        self.sim.get_channel_as_mut_ref(channel)
    }
    
    /// Retrieve a reference to the specified channel's "deltas". Deltas are the amounts added onto the 
    /// previous iteration's result to get the current iteration's result. 
    /// 
    /// Note that `dt` parameter has not been applied for this field, and no clamp / clip operation has
    /// been performed, thus the numbers will be in range `[-1.0..1.0]`.
    /// 
    /// ### Parameters
    /// 
    /// * `channel` - Index of the channel from which the reference to data is to be taken.
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist. 
    pub fn get_deltas_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64> {
        if channel >= self.sim.channels() {
            panic!("Simulator::get_deltas_as_ref() - Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        self.sim.get_deltas_as_ref(channel)
    }

    /// Retrieve a reference to the specified convolution channel's convolution result, 
    /// also called the "potential distribution". 
    /// 
    /// ### Parameters
    /// 
    /// * `convolution_channel` - Index of the convolution channel from which to
    /// produce the `f64` `ndarray`. 
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist.
    pub fn get_convoluted(&self, convolution_channel: usize) -> ndarray::ArrayD<f64> {
        if convolution_channel >= self.sim.channels() {
            panic!("Simulator::get_convoluted() - Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        self.sim.get_convoluted_as_ref(convolution_channel).map(|a| { a.re })
    }

    /// Retrieve a reference to the specified convolution channel's convolution results with
    /// the growth function applied, also called the "activation".
    /// 
    /// ### Parameters
    /// 
    /// * `convolution_channel` - Index of the convolution channel from which the
    /// reference to data is to be taken.
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist.
    pub fn get_activated_as_ref(&self, convolution_channel: usize) -> &ndarray::ArrayD<f64> {
        if convolution_channel >= self.sim.channels() {
            panic!("Simulator::get_grown_as_ref() - Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        self.sim.get_grown_as_ref(convolution_channel)
    }

    /// Retrieve the kernel being used for the specified convolution channels' convolution.
    /// 
    /// ### Parameters
    /// 
    /// * `convolution_channel` - Index of the convolution channel from which the kernel will be supplied.
    pub fn get_kernel_as_ref(&self, convolution_channel: usize) -> &Kernel {
        self.sim.get_kernel_as_ref(convolution_channel)
    }

    /// Get the current integration step (a.k.a. timestep) parameter `dt` of the `Lenia` instance.
    pub fn dt(&self) -> f64 {
        self.sim.dt()
    }

    /// Get the shape of the channels and convolution channels of the `Lenia` instance.
    pub fn shape(&self) -> &[usize] {
        self.sim.shape()
    }

    /// Get the number of channels initialized in the `Lenia` instance.
    pub fn channels(&self) -> usize {
        self.sim.channels()
    }

    /// Get the number of convolution channels initialized in the `Lenia` instance.
    pub fn convolution_channels(&self) -> usize {
        self.sim.conv_channels()
    }
}

/// Lenia functionality trait.
/// 
/// Lenia trait organizes together all the functionality to interact with a Lenia simulation. 
pub trait Lenia {
    /// Creates a new `Lenia` instance. 
    fn new(shape: &[usize]) -> Self;
    /// Sets the number of channels in the `Lenia` instance. 
    /// 
    /// **If the new number of channels is fewer than currently then the user is responsible for re-making
    /// the convolution channels or deleting invalidated convolution channels and
    /// make sure that no convolution channel tries to convolute a non-existent channel!**
    fn set_channels(&mut self, num_channels: usize);
    /// Sets the number of convolution channels in the `Lenia` instance. 
    /// 
    /// * Any convolution channels
    /// that have an index larger than the new number of channels **will be dropped**. Conversely,
    /// no convolution channels get invalidated if the new number of convolution channels is
    /// greater than the previous number of convolution channels. 
    /// 
    /// * Any newly initialized convolution channels will have to have their kernels and
    /// growth functions added. By default all channels will use a weight of `0.0` for the new
    /// channels. 
    fn set_conv_channels(&mut self, num_conv_channels: usize);
    /// Sets the source channel for a convolution channel.
    fn set_source_channel(&mut self, conv_channel: usize, src_channel: usize);
    /// Sets the convolution kernel for a convolution channel.
    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize);
    /// Sets the growth function for a convolution channel.
    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize);
    /// Sets the weights for input into a channel from convolution channels for summing. 
    /// 
    /// * If the length of `new weights` is less than the number of convolution channels then
    /// the weight for the uncovered convolution channels defaults to `0.0`. 
    /// 
    /// * If the length of `new weights` is greater than the number of convolution channels then
    /// the excess weights will be disregarded, and their effect for the weighted average on the 
    /// channel is not taken into account.
    fn set_weights(&mut self, new_weights: &[f64], channel: usize);
    /// Sets the dt parameter of the `Lenia` instance. 
    fn set_dt(&mut self, new_dt: f64);
    /// Returns a reference to a convolution channel's kernel.
    fn get_kernel_as_ref(&self, conv_channel: usize) -> &Kernel;
    /// Returns a reference to a channel's current data. 
    fn get_channel_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64>;
    /// Returns a mutable reference to a channel's current data.
    fn get_channel_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<f64>;
    /// Returns a reference to the convolution result.
    fn get_convoluted_as_ref(&self, conv_channel: usize) -> &ndarray::ArrayD<Complex<f64>>;
    /// Returns a reference to the field with growth function applied.
    fn get_grown_as_ref(&self, conv_channel: usize) -> &ndarray::ArrayD<f64>;
    /// Returns a reference to the results to be added to previous channel state. Lacks `dt` scaling.
    fn get_deltas_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64>;
    /// Returns the shape of the channels and convolution channels (as reference). 
    fn shape(&self) -> &[usize];
    /// Returns the current `dt` parameter of the `Lenia` instance.
    fn dt(&self) -> f64;
    /// Returns the number of channels in the `Lenia` instance.
    fn channels(&self) -> usize;
    /// Returns the number of convolution channels in the `Lenia` instance.
    fn conv_channels(&self) -> usize;
    /// Returns the weights of the specified channel.
    fn weights(&self, channel: usize) -> &[f64];
    /// Calculates the next state of the `Lenia` instance, and updates the data in channels accordingly.
    fn iterate(&mut self);
}

#[derive(Clone, Debug)]
/// A single channel in a Lenia simulation.
/// 
/// The `Channel` struct is a wrapper for holding the data of a single channel in a 
/// `Lenia` simulation. 
pub struct Channel {
    pub field: ndarray::ArrayD<f64>,
    pub weights: Vec<f64>,
    pub weight_sum_reciprocal: f64,
}

#[derive(Clone)]
/// A single kernel-growth function pair in a Lenia simulation.
/// 
/// The `ConvolutionChannel` struct holds relevant data for the convolution step of the
/// Lenia simulation. This includes the kernel, the intermittent convolution step, and the 
/// growth function.
pub struct ConvolutionChannel {
    pub input_channel: usize,
    pub field: ndarray::ArrayD<f64>,
    pub kernel: Kernel,
    pub growth: fn(f64, &[f64]) -> f64,
    pub growth_params: Vec<f64>,
}

impl fmt::Debug for ConvolutionChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConvolutionChannel")
         .field("input_channel", &self.input_channel)
         .field("field", &self.field)
         .field("kernel", &self.kernel)
         .field("growth", &"fn(f64, &[f64]) -> f64")
         .field("growth_params", &self.growth_params)
         .finish()
    }
}


#[derive(Clone, Debug)]
/// N-dimensional kernel.
/// 
/// The `Kernel` struct holds the data of a specific kernel to be used for convolution in
/// the Lenia simulation. It also implements the necessary conversions to normalize a
/// kernel and prepare it for convolution using fast-fourier-transform. 
/// 
/// ### Parameters
/// 
/// * `base` - The original `ndarray::ArrayD<f64>` fom which the `Kernel` got made from.
/// 
/// * `normalized` - The scaled down version of the base Kernel such that the sum of 
/// all the values of the kernel is `1.0`.
/// 
/// * `shifted` - Normalized kernel with its center shifted to the "top-right" corner and
/// the re-sized to match the size of the `Lenia` instance channels. 
/// This is necessary for fourier-transforming the kernel.
/// 
/// * `transformed` - Fourier-transformed kernel.
pub struct Kernel {
    pub base: ndarray::ArrayD<f64>,
    pub normalized: ndarray::ArrayD<f64>,
    pub shifted: ndarray::ArrayD<f64>,
    pub transformed: ndarray::ArrayD<Complex<f64>>,
}

impl Kernel {
    /// Creates a new Kernel struct from an n-dimensional array (`ndarray::ArrayD<f64>`).
    /// 
    /// Creates the normalized version of the kernel.
    /// 
    /// Creates a version of the kernel that has been transformed using discrete-fourier-transform, 
    /// and shifted, for future use in fast-fourier-transform based convolution. 
    /// 
    /// ### Parameters
    /// 
    /// * `kernel` - Base data for the kernel.
    /// 
    /// * `channel_shape` - Shape of the channel the kernel is supposed to act on.
    /// 
    /// ### Panics
    /// 
    /// * If the number of axes of the `kernel` and `channel_shape` are not equal.
    /// 
    /// * If any of the corresponding axis lengths in `kernel` are greater than in `channel_shape`.
    pub fn from(kernel: ndarray::ArrayD<f64>, channel_shape: &[usize]) -> Self {
        let mut normalized_kernel = kernel.clone();
        let shifted_and_fft = ndarray::ArrayD::from_elem(channel_shape, Complex::new(0.0, 0.0));
        
        // Check for coherence in dimensionality and that the kernel is not
        // larger than the channel it is used to convolve with.
        if normalized_kernel.shape().len() != shifted_and_fft.shape().len() { 
            panic!("Supplied kernel dimensionality does not match the supplied channel dimensionality!
                \nkernel: {} dimensional vs. channel: {} dimensional",
                normalized_kernel.shape().len(), shifted_and_fft.shape().len()
            ); 
        }
        for (i, dim) in normalized_kernel.shape().iter().enumerate() {
            if *dim > shifted_and_fft.shape()[i] { 
                panic!("Supplied kernel is larger than the channel it acts on in axis {}!", i);
            }
        }

        // Normalize the kernel 
        let scaler = 1.0 / normalized_kernel.sum();
        for elem in &mut normalized_kernel {
            *elem *= scaler;
        }

        // Expand the kernel to match the size of the channel shape
        let mut shifted = ndarray::ArrayD::from_elem(channel_shape, 0.0);

        normalized_kernel.assign_to(shifted.slice_each_axis_mut(
            |a| Slice {
                start: (a.len/2 - normalized_kernel.shape()[a.axis.index()]/2) as isize,
                end: Some((
                    a.len/2
                    + normalized_kernel.shape()[a.axis.index()]/2 
                    + normalized_kernel.shape()[a.axis.index()]%2
                ) as isize),
                step: 1
            }
        ));
        
        // Shift the kernel into the corner
        for (i, axis) in channel_shape.iter().enumerate() {
            let mut shifted_buffer = shifted.clone();
            shifted.slice_axis(
                    Axis(i), 
                    Slice{
                        start: -(*axis as isize / 2), 
                        end: None, 
                        step: 1,
                    }
                )
                .assign_to(shifted_buffer.slice_axis_mut(
                    Axis(i),
                    Slice { 
                        start: 0, 
                        end: Some(*axis as isize / 2), 
                        step: 1,
                    }
                )
            );
            shifted.slice_axis(
                    Axis(i), 
                    Slice{
                        start: 0, 
                        end: Some(*axis as isize / 2), 
                        step: 1,
                    }
                )
                .assign_to(shifted_buffer.slice_axis_mut(
                    Axis(i),
                    Slice { 
                        start: -(*axis as isize / 2), 
                        end: None, 
                        step: 1,
                    }
                )
            );
            shifted = shifted_buffer;
        }
        let shifted_stored = shifted.clone();

        // Create the discrete-fourier-transformed representation of the kernel for fft-convolving. 
        let mut shifted_and_fft = shifted.mapv(|elem| {Complex::new(elem, 0.0)});

        let mut fft_instance = fft::PlannedFFTND::new(&channel_shape, false);
        fft_instance.transform(&mut shifted_and_fft);

        // Create the kernel
        Kernel{
            base: kernel,
            normalized: normalized_kernel,
            shifted: shifted_stored,
            transformed: shifted_and_fft,
        }
    }
}



