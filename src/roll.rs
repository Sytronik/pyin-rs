use ndarray::{concatenate, prelude::*, Data, OwnedRepr, RemoveAxis, Slice};

pub(crate) trait Roll<A> {
    type WithOwnedA;
    fn roll(&self, shift: isize, axis: Axis) -> Self::WithOwnedA;
}

impl<A, S, D> Roll<A> for ArrayBase<S, D>
where
    A: Copy,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    type WithOwnedA = ArrayBase<OwnedRepr<A>, D>;
    fn roll(&self, shift: isize, axis: Axis) -> Self::WithOwnedA {
        let length = self.shape()[axis.index()];
        let abs_shift = shift.unsigned_abs();
        if shift == 0 {
            return self.to_owned();
        }
        let (left, right) = if shift > 0 {
            (
                self.slice_axis(axis, Slice::from((length - abs_shift)..)),
                self.slice_axis(axis, Slice::from(0..(length - abs_shift))),
            )
        } else {
            (
                self.slice_axis(axis, Slice::from(abs_shift..)),
                self.slice_axis(axis, Slice::from(..abs_shift)),
            )
        };
        concatenate![axis, left, right]
    }
}
