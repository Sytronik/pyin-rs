use std::cmp::Ord;
use std::fmt::Debug;
use std::ops::{Add, DivAssign, Mul, Neg};

use ndarray::{prelude::*, ScalarOperand};
use realfft::num_traits::{Float, FloatConst, FromPrimitive};

use crate::pad::{Pad, PadMode};
use crate::roll::Roll;
use crate::windows::{calc_normalized_win, WindowType};

pub(crate) fn parabolic_interpolation<A: Float + Add + Mul + ScalarOperand>(
    frames: ArrayView2<A>,
) -> Array2<A> {
    assert!(frames.shape()[0] > 2);
    let parabola_a = (&frames.slice(s![..-2, ..]) + &frames.slice(s![2.., ..])
        - &frames.slice(s![1..-1, ..]) * A::from(2.0).unwrap())
        / A::from(2.0).unwrap();
    let parabola_b =
        (&frames.slice(s![2.., ..]) - &frames.slice(s![..-2, ..])) / A::from(2.0).unwrap();
    let mut parabolic_shifts =
        &parabola_b.neg() / (&parabola_a * A::from(2.0).unwrap() + A::min_positive_value());
    for x in parabolic_shifts.iter_mut() {
        if x.abs() > A::one() {
            *x = A::zero();
        }
    }
    parabolic_shifts.pad((1, 1), Axis(0), PadMode::Constant(A::zero()))
}

pub(crate) fn boltzmann_pmf(k: isize, lambda: f64, n: usize) -> f64 {
    let fact = (1.0 - (-lambda).exp()) / (1.0 - (-lambda * n as f64).exp());
    fact * (-lambda * k as f64).exp()
}

pub(crate) fn transition_local<A>(
    n_states: usize,
    width: usize,
    win_type: WindowType,
    wrap: bool,
) -> Array2<A>
where
    A: Float + FloatConst + FromPrimitive + DivAssign + ScalarOperand + Debug,
{
    assert!(n_states > 1);
    assert!(width > 0);
    assert!(width < n_states);
    let width_arr = Array1::from_elem(n_states, width);
    let mut transition = Array2::<A>::zeros((n_states, n_states));

    for (i, width_i) in width_arr.into_iter().enumerate() {
        let trans_row = calc_normalized_win(win_type, width_i, A::one(), true);
        let n_pad_left = (n_states - width_i) / 2;
        let n_pad_right = n_states - width_i - n_pad_left;
        let mut trans_row = trans_row
            .pad(
                (n_pad_left, n_pad_right),
                Axis(0),
                PadMode::Constant(A::zero()),
            )
            .roll(((n_states / 2 + i + 1) % n_states) as isize, Axis(0));
        if !wrap {
            trans_row
                .slice_mut(s![(i + width_i / 2 + 1).min(n_states)..])
                .fill(A::zero());
            trans_row
                .slice_mut(s![..(i as isize - width_i as isize / 2).max(0)])
                .fill(A::zero());
        }
        transition.index_axis_mut(Axis(0), i).assign(&trans_row);
    }

    // Row-normalize
    transition /= &transition.sum_axis(Axis(1)).insert_axis(Axis(1));
    transition
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    #[test]
    fn test_boltzmann_pmf() {
        assert_abs_diff_eq!(
            Array::from_shape_fn(19, |i| {
                let k = i as isize;
                let lambda = 1.4;
                let n = 19;
                boltzmann_pmf(k, lambda, n)
            }),
            array![
                7.53403036e-01f64,
                1.85786901e-01,
                4.58144858e-02,
                1.12977131e-02,
                2.78598175e-03,
                6.87014641e-04,
                1.69415725e-04,
                4.17774034e-05,
                1.03021808e-05,
                2.54048652e-06,
                6.26476262e-07,
                1.54487144e-07,
                3.80960607e-08,
                9.39437291e-09,
                2.31662384e-09,
                5.71272405e-10,
                1.40874041e-10,
                3.47391107e-11,
                8.56655923e-12
            ],
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_transition_local() {
        assert_eq!(
            transition_local::<f32>(5, 3, WindowType::Triangle, false),
            array![
                [2f32 / 3., 1. / 3., 0., 0., 0.],
                [0.25, 0.5, 0.25, 0., 0.],
                [0., 0.25, 0.5, 0.25, 0.],
                [0., 0., 0.25, 0.5, 0.25],
                [0., 0., 0., 1. / 3., 2. / 3.]
            ]
        );
        assert_eq!(
            transition_local::<f32>(5, 3, WindowType::Triangle, true),
            array![
                [0.5f32, 0.25, 0., 0., 0.25],
                [0.25, 0.5, 0.25, 0., 0.],
                [0., 0.25, 0.5, 0.25, 0.],
                [0., 0., 0.25, 0.5, 0.25],
                [0.25, 0., 0., 0.25, 0.5]
            ]
        );
    }
}
