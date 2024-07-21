//! pYIN[^paper] is a pitch (fundamental frequency) detection algorithm.
//!
//! This crate provides a pitch estimate for each frame of the audio signal and a probability the frame is voiced region.
//!
//! This crate provides a C language FFI and an executable binary. For the details of the executable binary, refer to the [repo](https://github.com/Sytronik/pyin-rs).
//!
//! The implementation is based on [librosa v0.9.1](https://librosa.org/doc/0.9.1/_modules/librosa/core/pitch.html#pyin).
//! For easy translation from Python + Numpy to Rust, the implementation is written on top of [ndarray](https://crates.io/crates/ndarray) crate.
//!
//! ## Usage
//! ```
//! use ndarray::prelude::*;
//! use pyin::{PYINExecutor, Framing, PadMode};
//!
//! let fmin = 60f64;  // minimum frequency in Hz
//! let fmax = 600f64; // maximum frequency in Hz
//! let sr = 24000;    // sampling rate of audio data in Hz
//! let frame_length = 2048; // frame length in samples
//! let (win_length, hop_length, resolution) = (None, None, None);  // None to use default values
//! let mut pyin_exec = PYINExecutor::new(fmin, fmax, sr, frame_length, win_length, hop_length, resolution);
//!
//! let wav: Vec<f64> = (0..24000).map(|i| (2. * std::f64::consts::PI * (i as f64) / 200.).sin()).collect();
//! let fill_unvoiced = f64::NAN;
//! let framing = Framing::Center(PadMode::Constant(0.));  // Zero-padding is applied on both sides of the signal. (only if cetner is true)
//!
//! // timestamp (Array1<f64>) - contains the timestamp (in seconds) of each frames
//! // f0 (Array1<f64>) contains the pitch estimate in Hz. (NAN if unvoiced)
//! // voiced_flag (Array1<bool>) contains whether the frame is voiced or not.
//! // voiced_prob (Array1<f64>) contains the probability of the frame is voiced.
//! let (timestamp, f0, voiced_flag, voiced_prob) = pyin_exec.pyin(&wav, fill_unvoiced, framing);
//! ```
//!
//! [^paper]: Mauch, Matthias, and Simon Dixon. “pYIN: A fundamental frequency estimator using probabilistic threshold distributions.” 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.

mod pad;
mod pyin;
mod roll;
mod util;
mod viterbi;
mod windows;

use std::mem;
use std::slice;

use libc::{self, c_double, c_uint, c_void};

pub use pad::{Framing, PadMode};
pub use pyin::PYINExecutor;

/// C lang FFI for pYIN
///
/// detailed documentation is in [`PYINExecutor`](pyin::PYINExecutor)
///
/// # Note
///
/// - If `win_length` is `0`, use default value (frame_length / 2)
/// - If `hop_length` is `0`, use default value (frame_length / 4)
/// - If `resolution` <= 0, use default value (0.1)
/// - If `pad_mode` is 0, zero padding. If 1, reflection padding
///
///
/// # Safety
///
/// The caller must call free on f0, voiced_flag, voiced_prob to
/// prevent a memory leak.
#[no_mangle]
pub unsafe extern "C" fn pyin(
    // outputs
    timestamp: *mut *mut c_double,
    f0: *mut *mut c_double,
    voiced_flag: *mut *mut bool,
    voiced_prob: *mut *mut c_double,
    n_frames: *mut c_uint,

    // inputs
    input: *const c_double,
    length: c_uint,
    sr: c_uint,
    fmin: c_double,
    fmax: c_double,
    frame_length: c_uint,
    win_length: c_uint,   // If 0, use default value (frame_length / 2)
    hop_length: c_uint,   // If 0, use default value (frame_length / 4)
    resolution: c_double, // 0 < resolution < 1. If <= 0, use default value (0.1)
    fill_unvoiced: c_double,
    center: bool,
    pad_mode: c_uint, // 0: zero padding, 1: reflect padding
) -> isize {
    if f0.is_null()
        || !(*f0).is_null()
        || voiced_flag.is_null()
        || !(*voiced_flag).is_null()
        || voiced_prob.is_null()
        || !(*voiced_prob).is_null()
        || input.is_null()
    {
        return libc::EINVAL as isize;
    }
    let length = if length > 0 {
        length as usize
    } else {
        return 1;
    };
    if sr == 0 {
        return 1;
    }
    if fmin < 0. {
        return 1;
    }
    if fmax > sr as c_double / 2. {
        return 1;
    }
    let frame_length = if frame_length > 0 {
        frame_length as usize
    } else {
        return 1;
    };
    let win_length = (win_length > 0).then_some(win_length as usize);
    let hop_length = (hop_length > 0).then_some(hop_length as usize);
    let resolution = match resolution {
        x if 0. < x && x < 1. => Some(x),
        x if x <= 0. => None,
        _ => return 1,
    };
    let pad_mode = match pad_mode {
        0 => PadMode::Constant(0.),
        1 => PadMode::Reflect,
        _ => return 1,
    };
    let wav = slice::from_raw_parts(input, length);

    let mut pyin_executor = PYINExecutor::<c_double>::new(
        fmin,
        fmax,
        sr,
        frame_length,
        win_length,
        hop_length,
        resolution,
    );
    let framing = if center {
        Framing::Center(pad_mode)
    } else {
        Framing::Valid
    };
    let (_timestamp, _f0, _voiced_flag, _voiced_prob) =
        pyin_executor.pyin(wav, fill_unvoiced, framing);

    *n_frames = _f0.len() as c_uint;
    let double_memsize = _f0.len() * mem::size_of::<c_double>();
    let bool_memsize = _voiced_flag.len() * mem::size_of::<bool>();
    *timestamp = libc::malloc(double_memsize) as *mut c_double;
    *f0 = libc::malloc(double_memsize) as *mut c_double;
    *voiced_flag = libc::malloc(bool_memsize) as *mut bool;
    *voiced_prob = libc::malloc(double_memsize) as *mut c_double;
    if (*timestamp).is_null()
        || (*f0).is_null()
        || (*voiced_flag).is_null()
        || (*voiced_prob).is_null()
    {
        return libc::EINVAL as isize;
    }
    libc::memcpy(
        *timestamp as *mut c_void,
        _timestamp.as_ptr() as *const c_void,
        double_memsize,
    );
    libc::memcpy(
        *f0 as *mut c_void,
        _f0.as_ptr() as *const c_void,
        double_memsize,
    );
    libc::memcpy(
        *voiced_flag as *mut c_void,
        _voiced_flag.as_ptr() as *const c_void,
        bool_memsize,
    );
    libc::memcpy(
        *voiced_prob as *mut c_void,
        _voiced_prob.as_ptr() as *const c_void,
        double_memsize,
    );
    0
}
