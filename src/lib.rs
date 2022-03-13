pub mod pad;
mod pyin;
mod roll;
mod util;
mod viterbi;
mod windows;

use std::mem;
use std::slice;

use libc::{self, c_double, c_uint, c_void};
use ndarray::prelude::*;

use pad::PadMode;
pub use pyin::PYinExecutor;

#[no_mangle]
pub unsafe extern "C" fn pyin(
    f0: *mut *mut c_double,
    voiced_flag: *mut *mut bool,
    voiced_prob: *mut *mut c_double,
    n_frames: *mut c_uint,
    input: *const c_double,
    length: c_uint,
    sr: c_uint,
    fmin: c_double,
    fmax: c_double,
    frame_length: c_uint,
    win_length: c_uint,
    hop_length: c_uint,
    resolution: c_double,
    fill_unvoiced: c_double,
    center: bool,
    pad_mode: c_uint,
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
    let sr = if sr > 0 {
        sr as u32
    } else {
        return 1;
    };
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
    let win_length = (win_length > 0).then(|| win_length as usize);
    let hop_length = (hop_length > 0).then(|| hop_length as usize);
    let resolution = match resolution {
        x if 0. < x && x < 1. => Some(x as f64),
        x if x <= 0. => None,
        _ => return 1,
    };
    let pad_mode = match pad_mode {
        0 => PadMode::Constant(0.),
        1 => PadMode::Reflect,
        _ => return 1,
    };
    let wav = CowArray::from(slice::from_raw_parts(input, length));

    let mut pyin_executor = PYinExecutor::<c_double>::new(
        fmin as f64,
        fmax as f64,
        sr,
        frame_length,
        win_length,
        hop_length,
        resolution,
    );
    let (_f0, _voiced_flag, _voiced_prob) =
        pyin_executor.pyin(wav, fill_unvoiced, center, pad_mode);

    *n_frames = _f0.len() as c_uint;
    let double_memsize = _f0.len() * mem::size_of::<c_double>();
    let bool_memsize = _voiced_flag.len() * mem::size_of::<bool>();
    *f0 = libc::malloc(double_memsize) as *mut c_double;
    *voiced_flag = libc::malloc(bool_memsize) as *mut bool;
    *voiced_prob = libc::malloc(double_memsize) as *mut c_double;
    if (*f0).is_null() || (*voiced_flag).is_null() || (*voiced_prob).is_null() {
        return libc::EINVAL as isize;
    }
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
