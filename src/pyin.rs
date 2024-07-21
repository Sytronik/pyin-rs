use std::cmp::Ord;
use std::iter;
use std::mem::MaybeUninit;
use std::ops::{Add, DivAssign, Mul, MulAssign, Sub};
use std::sync::Arc;

use approx::AbsDiffEq;
use getset::{CopyGetters, Getters, Setters};
use ndarray::{prelude::*, ScalarOperand, Zip};
use ndarray_stats::{MaybeNan, QuantileExt};
use realfft::{
    num_complex::Complex,
    num_traits::{Float, FloatConst, NumAssign},
    ComplexToReal, FftNum, RealFftPlanner, RealToComplex,
};
use statrs::distribution::{Beta, ContinuousCDF};

use crate::pad::{Framing, Pad, PadMode};
use crate::util::*;
use crate::viterbi::viterbi;
use crate::windows::WindowType;

/// pYIN algorithm executor.
///
/// It's generic over the input type, which can be`f32` or `f64`.
/// It contains memory for the FFTs and some reusable constants.
/// The [`new()`](Self::new) method creates a new instance, and the [`pyin()`](Self::pyin) method executes the algorithm.
#[derive(Getters, Setters, CopyGetters, Clone)]
pub struct PYINExecutor<A>
where
    A: Float
        + FloatConst
        + FftNum
        + Add
        + Sub
        + Mul
        + MulAssign
        + DivAssign
        + NumAssign
        + ScalarOperand
        + MaybeNan
        + AbsDiffEq<Epsilon = A>,
    <A as MaybeNan>::NotNan: Ord,
{
    fmin: A,
    #[allow(dead_code)]
    fmax: A,
    sr: u32,
    frame_length: usize,
    win_length: usize,
    hop_length: usize,
    min_period: usize,
    max_period: usize,
    fft_module: Arc<dyn RealToComplex<A>>,
    ifft_module: Arc<dyn ComplexToReal<A>>,
    frame_fft: Array1<Complex<A>>,
    frame_rev_fft: Array1<Complex<A>>,
    acf_frame: Array1<A>,
    fft_scratch: Vec<Complex<A>>,
    ifft_scratch: Vec<Complex<A>>,
    n_bins_per_semitone: usize,
    n_pitch_bins: usize,

    /// number of thresholds for peak estimation. (always returns `100`)
    #[getset(get_copy = "pub")]
    n_thresholds: usize,

    /// shape parameters for the beta distribution prior over thresholds. (always returns `(2.0, 18.0)`)
    #[getset(get_copy = "pub")]
    beta_parameters: (f64, f64),
    beta_probs: Array1<A>,

    /// shape parameter for the Boltzmann distribution prior over troughs. Larger values will assign more mass to smaller periods. (default: `2.0`)
    #[getset(get_copy = "pub", set = "pub")]
    boltzmann_parameter: f64,

    /// maximum pitch transition rate in octaves per second. (always returns `35.92`)
    #[getset(get_copy = "pub")]
    max_transition_rate: f64,

    /// probability of switching from voiced to unvoiced or vice versa. (always returns `0.01`)
    #[getset(get_copy = "pub")]
    switch_prob: A,
    transition: Array2<A>,

    /// maximum probability to add to global minimum if no trough is below threshold. (default: `0.01`)
    #[getset(get_copy = "pub", set = "pub")]
    no_trough_prob: A,
}

impl<A> PYINExecutor<A>
where
    A: Float
        + FloatConst
        + FftNum
        + Add
        + Sub
        + Mul
        + MulAssign
        + DivAssign
        + NumAssign
        + ScalarOperand
        + MaybeNan
        + AbsDiffEq<Epsilon = A>,
    <A as MaybeNan>::NotNan: Ord,
{
    /// Create a new PYIN executor instance.
    ///
    /// # Arguments
    /// * `fmin` - minimum frequency in Hz
    /// * `fmax` - maximum frequency in Hz
    /// * `sr` - sampling rate in Hz
    /// * `frame_length` - frame length in samples
    /// * `win_length` - length of the window for calculating autocorrelation in samples. (If `None`, defaults to `frame_length/2`)
    /// * `hop_length` - number of audio samples between adjacent pYIN predictions. (If `None`, defaults to `frame_length/4`)
    /// * `resolution` - Resolution of the pitch bins. should be `0` < `resolution` < `1`. `0.01` corresponds to cents.(If `None`, defaults to `0.1`)
    pub fn new(
        fmin: f64,
        fmax: f64,
        sr: u32,
        frame_length: usize,
        win_length: Option<usize>,
        hop_length: Option<usize>,
        resolution: Option<f64>,
    ) -> Self {
        assert!(0. < fmin && fmin < fmax && fmax <= (sr as f64) / 2.);
        assert!(sr > 0);
        assert!(frame_length > 0);
        let win_length = win_length.unwrap_or(frame_length / 2);
        assert!(0 < win_length && win_length <= frame_length);
        let hop_length = hop_length.unwrap_or(frame_length / 4);
        assert!(hop_length > 0);
        let resolution = resolution.unwrap_or(0.1);
        assert!(0. < resolution && resolution < 1.);
        let n_bins_per_semitone = (1.0 / resolution).ceil() as usize;

        let min_period = ((sr as f64 / fmax).floor() as usize).max(1);
        let max_period = ((sr as f64 / fmin).ceil() as usize).min(frame_length - win_length - 1);
        if max_period - min_period < 2 {
            panic!("min(ceil(sr / fmin), (frame_length - win_length - 1)) + 2 < floor(sr / fmax) should be satisfied!");
        }

        let n_pitch_bins =
            (12.0 * n_bins_per_semitone as f64 * (fmax / fmin).log2()).floor() as usize + 1;
        if n_pitch_bins < 2 {
            panic!(
                "The number of possible pitch value is 1!\n
                  Possible reasons:\n
                  1. fmin and fmax is too close.\n
                  2. resolution is too high."
            );
        }

        let n_thresholds = 100;
        let beta_parameters = (2.0, 18.0);
        let beta_dist = Beta::new(beta_parameters.0, beta_parameters.1).unwrap();
        let beta_cdf: Array1<_> = (0..(n_thresholds + 1))
            .map(|i| beta_dist.cdf(i as f64 / n_thresholds as f64))
            .collect();
        let beta_probs: Array1<_> = beta_cdf
            .windows(2)
            .into_iter()
            .map(|x| A::from(x[1] - x[0]).unwrap())
            .collect();

        // Construct transition matrix.
        // Construct the within voicing transition probabilities
        let max_transition_rate = 35.92;
        let switch_prob = A::from(0.01).unwrap();
        let max_semitones_per_frame =
            (max_transition_rate * 12.0 * hop_length as f64 / sr as f64).round() as usize;
        let transition_width = max_semitones_per_frame * n_bins_per_semitone + 1;
        if n_pitch_bins < transition_width {
            panic!(
                "Cannot calculate transition matrix!\n
                  Possible reasons:\n
                  1. fmin and fmax is too close.\n
                  2. resolution is too high.\n
                  3. hop_length is too short."
            );
        }
        let transition =
            transition_local::<A>(n_pitch_bins, transition_width, WindowType::Triangle, false);
        // Include across voicing transition probabilities
        /* transition = np.block(
            [
                [(1 - switch_prob) * self.transition, switch_prob * self.transition],
                [switch_prob * self.transition, (1 - switch_prob) * self.transition],
            ]
        ) */
        let transition = Array2::build_uninit((n_pitch_bins * 2, n_pitch_bins * 2), |mut arr| {
            (&transition * (A::one() - switch_prob))
                .move_into_uninit(arr.slice_mut(s![..n_pitch_bins, ..n_pitch_bins]));
            (&transition * switch_prob).move_into_uninit(
                arr.slice_mut(s![..n_pitch_bins, n_pitch_bins..2 * n_pitch_bins]),
            );
            let (block00, block01, mut block10, mut block11) = arr.multi_slice_mut((
                s![..n_pitch_bins, ..n_pitch_bins],
                s![..n_pitch_bins, n_pitch_bins..2 * n_pitch_bins],
                s![n_pitch_bins..2 * n_pitch_bins, ..n_pitch_bins],
                s![
                    n_pitch_bins..2 * n_pitch_bins,
                    n_pitch_bins..2 * n_pitch_bins
                ],
            ));
            block11.assign(&block00);
            block10.assign(&block01);
        });
        let transition = unsafe { transition.assume_init() };

        let mut fft_planner = RealFftPlanner::new();
        let fft_module = fft_planner.plan_fft_forward(frame_length);
        let ifft_module = fft_planner.plan_fft_inverse(frame_length);

        let frame_fft = Array::from(fft_module.make_output_vec());
        let frame_rev_fft = Array::from(fft_module.make_output_vec());
        let acf_frame = Array::from(ifft_module.make_output_vec());
        let fft_scratch = fft_module.make_scratch_vec();
        let ifft_scratch = ifft_module.make_scratch_vec();

        PYINExecutor {
            fmin: A::from(fmin).unwrap(),
            fmax: A::from(fmax).unwrap(),
            sr,
            frame_length,
            win_length,
            hop_length,
            min_period,
            max_period,
            fft_module,
            ifft_module,
            frame_fft,
            frame_rev_fft,
            acf_frame,
            fft_scratch,
            ifft_scratch,
            n_bins_per_semitone,
            n_pitch_bins,
            n_thresholds,
            beta_parameters,
            beta_probs,
            boltzmann_parameter: 2.0,
            max_transition_rate,
            switch_prob,
            transition,
            no_trough_prob: A::from(0.01).unwrap(),
        }
    }

    /// Execute pYIN algorithm.
    ///
    /// # Arguments
    /// * `wav` - audio signal (Vec<A>)
    /// * `fill_unvoiced` - value to fill unvoiced frames. Typically, it is `0.0` or `<A as Float>::nan()`.
    /// * `framing` - where the first frame starts. Refer to [Framing](enum@Framing)
    ///
    /// # Returns
    /// `(timestamp: Vec<f64>, f0: Vec<A>, voiced_flag: Vec<bool>, voiced_prob: Vec<A>)`
    /// * `timestamp` - contains the timestamp (in seconds) of each frames
    /// * `f0` - contains estimated pitch in Hz. (If unvoiced, it is `fill_unvoiced`.)
    /// * `voiced_flag` - contains whether each frame is voiced or unvoiced.
    /// * `voiced_prob` - contains probability of each frame being voiced.
    pub fn pyin(
        &mut self,
        wav: &[A],
        fill_unvoiced: A,
        framing: Framing<A>,
    ) -> (Vec<f64>, Vec<A>, Vec<bool>, Vec<A>) {
        let wav = CowArray::from(wav);
        let wav = if let Framing::Center(pad_mode) = framing {
            wav.pad(
                (self.frame_length / 2, self.frame_length / 2),
                Axis(0),
                pad_mode,
            )
            .into()
        } else {
            if wav.len() < self.frame_length {
                panic!(
                    "Input wav is too short!\n
                     wav length: {}\n
                     frame_length: {}",
                    wav.len(),
                    self.frame_length
                );
            }
            wav
        };

        let yin_frames = self.frame_cum_mean_norm_diff(wav.view()).t().to_owned();
        let parabolic_shifts = parabolic_interpolation(yin_frames.view());

        // Find Yin candidates and probabilities.
        // The implementation here follows the official pYIN software which
        // differs from the method described in the paper.
        // 1. Define the prior over the thresholds.
        let thresholds = Array1::linspace(A::zero(), A::one(), self.n_thresholds + 1);

        let mut yin_probs = Array2::zeros(yin_frames.raw_dim());
        Zip::from(yin_frames.axis_iter(Axis(1)))
            .and(yin_probs.axis_iter_mut(Axis(1)))
            .for_each(|yin_frame, mut yin_prob| {
                // 2. For each frame find the troughs.
                let idxs_trough: Array1<_> = iter::once(yin_frame[0] < yin_frame[1])
                    .chain(
                        yin_frame
                            .windows(3)
                            .into_iter()
                            .map(|x| x[1] < x[0] && x[1] <= x[2]),
                    )
                    .chain(iter::once(
                        yin_frame[yin_frame.len() - 1] < yin_frame[yin_frame.len() - 2],
                    ))
                    .enumerate()
                    .filter_map(|(i, x)| if x { Some(i) } else { None })
                    .collect();

                if idxs_trough.is_empty() {
                    return;
                }

                // 3. Find the troughs below each threshold.
                let trough_thresholds =
                    Array::from_shape_fn((idxs_trough.len(), thresholds.len() - 1), |(i, j)| {
                        yin_frame[idxs_trough[i]] < thresholds[j + 1]
                    });

                // 4. Define the prior over the troughs.
                // Smaller periods are weighted more.
                let mut trough_positions =
                    Array::from_shape_fn(trough_thresholds.raw_dim(), |(i, j)| {
                        trough_thresholds[[i, j]] as isize
                    });
                trough_positions.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
                trough_positions -= 1;
                let n_troughs: Array1<_> = trough_thresholds
                    .axis_iter(Axis(1))
                    .map(|x| x.iter().filter(|v| **v).count())
                    .collect();
                let mut trough_prior =
                    Array::from_shape_fn(trough_positions.raw_dim(), |(i, j)| {
                        if trough_thresholds[[i, j]] {
                            A::from(boltzmann_pmf(
                                trough_positions[[i, j]],
                                self.boltzmann_parameter,
                                n_troughs[j],
                            ))
                            .unwrap()
                        } else {
                            A::zero()
                        }
                    });

                // 5. For each threshold add probability to global minimum if no trough is below threshold,
                // else add probability to each trough below threshold biased by prior.
                trough_prior *= &self.beta_probs;
                let mut probs = trough_prior.sum_axis(Axis(1));
                let global_min = idxs_trough.mapv(|i| yin_frame[i]).argmin_skipnan().unwrap();
                let n_thresholds_below_min = trough_thresholds
                    .slice_move(s![global_min, ..])
                    .into_iter()
                    .filter(|x| !x)
                    .count();
                probs[global_min] +=
                    self.no_trough_prob * self.beta_probs.slice(s![..n_thresholds_below_min]).sum();

                for (i_probs, i_trough) in idxs_trough.into_iter().enumerate() {
                    yin_prob[i_trough] = probs[i_probs];
                }
            });
        let (yin_period, frame_index): (Vec<_>, Vec<_>) = yin_probs
            .indexed_iter()
            .filter_map(|((i, j), v)| if v.is_zero() { None } else { Some((i, j)) })
            .unzip();
        let yin_period = Array::from(yin_period);
        let frame_index = Array::from(frame_index);

        // Refine peak by parabolic interpolation.
        let f0_candidates: Array1<_> =
            Zip::from(&yin_period)
                .and(&frame_index)
                .map_collect(|&i, &j| {
                    A::from(self.sr).unwrap()
                        / (A::from(i + self.min_period).unwrap() + parabolic_shifts[[i, j]])
                });

        // Find pitch bin corresponding to each f0 candidate.
        let bin_index = f0_candidates.mapv(|x| (x / self.fmin).log2())
            * A::from(12 * self.n_bins_per_semitone).unwrap();
        let bin_index =
            bin_index.mapv(|x| x.round().to_usize().unwrap().clamp(0, self.n_pitch_bins));

        // Observation probabilities.
        let mut observation_probs = Array2::zeros((2 * self.n_pitch_bins, yin_frames.shape()[1]));
        for i in 0..bin_index.shape()[0] {
            observation_probs[[bin_index[i], frame_index[i]]] =
                yin_probs[[yin_period[i], frame_index[i]]];
        }
        let mut voiced_prob = observation_probs
            .slice(s![..self.n_pitch_bins, ..])
            .sum_axis(Axis(0));
        voiced_prob.mapv_inplace(|x| x.max(A::zero()).min(A::one()));
        observation_probs
            .slice_mut(s![self.n_pitch_bins.., ..])
            .assign(
                &((-voiced_prob.slice(s![NewAxis, ..]).to_owned() + A::one())
                    / A::from(self.n_pitch_bins).unwrap()),
            );

        let p_init = Array1::from_shape_fn(2 * self.n_pitch_bins, |i| {
            if i < self.n_pitch_bins {
                A::zero()
            } else {
                A::one() / A::from(self.n_pitch_bins).unwrap()
            }
        });

        // Viterbi decoding.
        let (states, _) = viterbi(
            observation_probs.view(),
            self.transition.view(),
            Some(p_init.into()),
        );

        // Find f0 corresponding to each decoded pitch bin.
        let mut freqs = Array1::range(A::zero(), A::from(self.n_pitch_bins).unwrap(), A::one());

        freqs.mapv_inplace(|x| {
            self.fmin * (x / A::from(12 * self.n_bins_per_semitone).unwrap()).exp2()
        });

        let mut f0 = (&states % self.n_pitch_bins).mapv(|x| freqs[x]);
        let voiced_flag = states.mapv(|x| x < self.n_pitch_bins);
        azip!((x in &mut f0, &flag in &voiced_flag) {
            if !flag {
                *x = fill_unvoiced;
            }
        });

        let timestamp_sec = (0..f0.shape()[0]).map(|i| self.frame_to_sec(i)).collect();
        (
            timestamp_sec,
            f0.into_raw_vec(),
            voiced_flag.into_raw_vec(),
            voiced_prob.into_raw_vec(),
        )
    }

    pub fn frame_to_sec(&self, i_frame: usize) -> f64 {
        i_frame as f64 * self.hop_length as f64 / self.sr as f64
    }

    /// Cumulative mean normalized difference function (equation 8 in [#]_)
    fn frame_cum_mean_norm_diff(&mut self, wav: ArrayView1<A>) -> Array2<A> {
        let n_frames = (wav.len() - self.frame_length) / self.hop_length + 1;
        let mut wav_frames = Array2::uninit((n_frames, self.frame_length));
        wav.windows(self.frame_length)
            .into_iter()
            .step_by(self.hop_length)
            .zip(wav_frames.axis_iter_mut(Axis(0)))
            .for_each(|(x, y)| {
                x.assign_to(y);
            });
        let wav_frames = unsafe { wav_frames.assume_init() };

        // Autocorrelation
        let mut acf_frames = Array2::uninit((n_frames, self.frame_length - self.win_length));
        Zip::from(wav_frames.axis_iter(Axis(0)))
            .and(acf_frames.axis_iter_mut(Axis(0)))
            .for_each(|wav_frame, mut frame| {
                let mut wav_frame = wav_frame.to_owned();

                let mut wav_frame_rev = wav_frame.slice(s![1..self.win_length+1;-1]).pad(
                    (0, self.frame_length - self.win_length),
                    Axis(0),
                    PadMode::Constant(A::zero()),
                );

                self.fft_module
                    .process_with_scratch(
                        wav_frame.as_slice_mut().unwrap(),
                        self.frame_fft.as_slice_mut().unwrap(),
                        &mut self.fft_scratch,
                    )
                    .unwrap();
                self.fft_module
                    .process_with_scratch(
                        wav_frame_rev.as_slice_mut().unwrap(),
                        self.frame_rev_fft.as_slice_mut().unwrap(),
                        &mut self.fft_scratch,
                    )
                    .unwrap();
                self.frame_fft *= &self.frame_rev_fft;

                self.ifft_module
                    .process_with_scratch(
                        self.frame_fft.as_slice_mut().unwrap(),
                        self.acf_frame.as_slice_mut().unwrap(),
                        &mut self.ifft_scratch,
                    )
                    .unwrap();
                self.acf_frame /= A::from(self.frame_length).unwrap(); // to match Python numpy.fft.irfft
                Zip::from(self.acf_frame.slice(s![self.win_length..]))
                    .and(&mut frame)
                    .for_each(|&x, y| {
                        if x.abs() >= A::from(1e-6).unwrap() {
                            *y = MaybeUninit::new(x);
                        } else {
                            *y = MaybeUninit::new(A::zero());
                        }
                    });
            });
        let acf_frames = unsafe { acf_frames.assume_init() };

        // Energy terms
        let mut energy_frames = wav_frames.mapv(|x| x.powi(2));
        energy_frames.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
        let mut energy_frames = &energy_frames.slice(s![.., self.win_length..])
            - &energy_frames.slice(s![.., ..-(self.win_length as isize)]);
        energy_frames.mapv_inplace(|x| {
            if x.abs() >= A::from(1e-6).unwrap() {
                x
            } else {
                A::zero()
            }
        });

        // Difference function
        let mut yin_frames = &energy_frames.slice(s![.., 0..1]) + &energy_frames
            - &acf_frames * A::from(2.0).unwrap();

        // Cumulative mean normalized difference function.
        let yin_numerator = yin_frames.slice(s![.., self.min_period..self.max_period + 1]);
        let tau_range = Array::range(A::one(), A::from(self.max_period + 1).unwrap(), A::one())
            .into_shape((1, self.max_period))
            .unwrap();
        let mut cumulative_mean = yin_frames.slice(s![.., 1..self.max_period + 1]).to_owned();
        cumulative_mean.accumulate_axis_inplace(Axis(1), |&prev, curr| *curr += prev);
        cumulative_mean /= &tau_range;
        let yin_denominator = cumulative_mean.slice(s![.., self.min_period - 1..self.max_period]);
        yin_frames = &yin_numerator / (&yin_denominator + A::min_positive_value());
        yin_frames
    }
}
