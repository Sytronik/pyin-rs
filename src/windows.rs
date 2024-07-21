use std::ops::Div;

use ndarray::{concatenate, prelude::*, ScalarOperand};
use realfft::num_traits::{Float, FloatConst, FromPrimitive, ToPrimitive};

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub(crate) enum WindowType {
    Hann,
    Blackman,
    Triangle,
    BoxCar,
}

#[inline]
pub(crate) fn calc_normalized_win<T>(
    win_type: WindowType,
    size: usize,
    norm_factor: impl ToPrimitive,
    symmetric: bool,
) -> Array1<T>
where
    T: Float + FloatConst + FromPrimitive + Div + ScalarOperand,
{
    let norm_factor = T::from(norm_factor).unwrap();
    match win_type {
        WindowType::Hann => hann(size, symmetric) / norm_factor,
        WindowType::Blackman => blackman(size, symmetric) / norm_factor,
        WindowType::Triangle => triangle(size, symmetric) / norm_factor,
        WindowType::BoxCar => Array1::from_elem(size, T::one() / norm_factor),
    }
}

pub(crate) fn triangle<T>(size: usize, symmetric: bool) -> Array1<T>
where
    T: Float + FromPrimitive + ScalarOperand,
{
    let is_odd = size % 2 == 1;
    let size2 = if !symmetric && !is_odd {
        size + 1
    } else {
        size
    };
    let mut half_win: Array1<_> = (1..((size2 + 1) / 2 + 1))
        .map(|x| T::from(x).unwrap())
        .collect();
    let win = if size2 % 2 == 0 {
        half_win = (half_win * T::from(2.0).unwrap() - T::one()) / T::from(size2).unwrap();
        concatenate!(Axis(0), half_win, half_win.slice(s![..;-1]))
    } else {
        half_win = half_win * T::from(2.0).unwrap() / T::from(size2 + 1).unwrap();

        concatenate!(Axis(0), half_win, half_win.slice(s![..-1;-1]))
    };

    if !symmetric && !is_odd {
        win.slice_move(s![..;-1])
    } else {
        win
    }
}

#[inline]
pub(crate) fn hann<T>(size: usize, symmetric: bool) -> Array1<T>
where
    T: Float + FloatConst + FromPrimitive,
{
    cosine_window(
        T::from(0.5).unwrap(),
        T::from(0.5).unwrap(),
        T::zero(),
        T::zero(),
        size,
        symmetric,
    )
}

// from rubato crate
pub(crate) fn blackman<T>(size: usize, symmetric: bool) -> Array1<T>
where
    T: Float + FloatConst + FromPrimitive,
{
    assert!(size > 1);
    let size2 = if symmetric { size + 1 } else { size };
    let pi2 = T::from_u8(2).unwrap() * T::PI();
    let pi4 = T::from_u8(4).unwrap() * T::PI();
    let np_f = T::from(size2).unwrap();
    let a = T::from(0.42).unwrap();
    let b = T::from(0.5).unwrap();
    let c = T::from(0.08).unwrap();
    (0..size2)
        .map(|x| {
            let x_float = T::from_usize(x).unwrap();
            a - b * (pi2 * x_float / np_f).cos() + c * (pi4 * x_float / np_f).cos()
        })
        .skip(if symmetric { 1 } else { 0 })
        .collect()
}

#[allow(clippy::many_single_char_names)]
fn cosine_window<T>(a: T, b: T, c: T, d: T, size: usize, symmetric: bool) -> Array1<T>
where
    T: Float + FloatConst + FromPrimitive,
{
    assert!(size > 1);
    let size2 = if symmetric { size } else { size + 1 };
    let cos_fn = |i| {
        let x = T::PI() * T::from_usize(i).unwrap() / T::from_usize(size2 - 1).unwrap();
        let b_ = b * (T::from_u8(2).unwrap() * x).cos();
        let c_ = c * (T::from_u8(4).unwrap() * x).cos();
        let d_ = d * (T::from_u8(6).unwrap() * x).cos();
        (a - b_) + (c_ - d_)
    };
    (0..size2).map(cos_fn).take(size).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_win() {
        assert_eq!(hann::<f32>(4, false), arr1(&[0f32, 0.5, 1., 0.5]));
    }

    #[test]
    fn test_triangle_win() {
        assert_eq!(triangle::<f32>(3, true), arr1(&[0.5, 1., 0.5]));
    }
}
