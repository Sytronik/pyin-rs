use std::mem::MaybeUninit;

use ndarray::OwnedRepr;
use ndarray::{prelude::*, Data, RemoveAxis, Slice, Zip};

/// Represents where the first frame starts.
pub enum Framing<T> {
    /// The first sample of audio signal becomes the center of the first frame. So, padding is added to the left.
    Center(PadMode<T>),
    /// The first sample of audio signal becomes the first sample of the first frame. So, padding is not added.
    Valid,
}

/// Padding mode
pub enum PadMode<T> {
    /// Constant padding
    Constant(T),

    /// Reflection padding
    Reflect,
}

pub(crate) trait Pad<A> {
    type WithOwnedA;
    fn pad(&self, n_pads: (usize, usize), axis: Axis, mode: PadMode<A>) -> Self::WithOwnedA;
}

impl<A, S, D> Pad<A> for ArrayBase<S, D>
where
    A: Copy,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    type WithOwnedA = ArrayBase<OwnedRepr<A>, D>;
    fn pad(
        &self,
        (n_pad_left, n_pad_right): (usize, usize),
        axis: Axis,
        mode: PadMode<A>,
    ) -> Self::WithOwnedA {
        let mut shape = self.raw_dim();
        shape[axis.index()] += n_pad_left + n_pad_right;
        let mut result = Self::WithOwnedA::uninit(shape);

        let s_result_main = if n_pad_right > 0 {
            Slice::from(n_pad_left as isize..-(n_pad_right as isize))
        } else {
            Slice::from(n_pad_left as isize..)
        };
        Zip::from(self).map_assign_into(result.slice_axis_mut(axis, s_result_main), |x| *x);
        match mode {
            PadMode::Constant(constant) => {
                result
                    .slice_axis_mut(axis, Slice::from(..n_pad_left))
                    .mapv_inplace(|_| MaybeUninit::new(constant));
                if n_pad_right > 0 {
                    result
                        .slice_axis_mut(axis, Slice::from(-(n_pad_right as isize)..))
                        .mapv_inplace(|_| MaybeUninit::new(constant));
                }
            }
            PadMode::Reflect => {
                let pad_left = self
                    .axis_iter(axis)
                    .skip(1)
                    .chain(self.axis_iter(axis).rev().skip(1))
                    .cycle()
                    .take(n_pad_left);
                result
                    .axis_iter_mut(axis)
                    .take(n_pad_left)
                    .rev()
                    .zip(pad_left)
                    .for_each(|(y, x)| Zip::from(x).map_assign_into(y, |x| *x));

                if n_pad_right > 0 {
                    let pad_right = self
                        .axis_iter(axis)
                        .rev()
                        .skip(1)
                        .chain(self.axis_iter(axis).skip(1))
                        .cycle()
                        .take(n_pad_right);
                    result
                        .axis_iter_mut(axis)
                        .rev()
                        .take(n_pad_right)
                        .rev()
                        .zip(pad_right)
                        .for_each(|(y, x)| Zip::from(x).map_assign_into(y, |x| *x));
                }
            }
        }
        unsafe { result.assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::arr2;

    #[test]
    fn test_pad() {
        assert_eq!(
            arr2(&[[1, 2, 3]]).pad((1, 2), Axis(0), PadMode::Constant(10)),
            arr2(&[[10, 10, 10], [1, 2, 3], [10, 10, 10], [10, 10, 10]])
        );
        assert_eq!(
            arr2(&[[1, 2, 3]]).pad((3, 4), Axis(1), PadMode::Reflect),
            arr2(&[[2, 3, 2, 1, 2, 3, 2, 1, 2, 3]])
        );
    }
}
