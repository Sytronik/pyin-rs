use std::fs::File;
use std::io;

use clap::Parser;
use ndarray::prelude::*;
use ndarray_npy::WriteNpyExt;
use rayon::prelude::*;
use rodio::decoder::DecoderError;
use rodio::{Decoder, Source};

use pyin::{pad::PadMode, PYINExecutor};

#[derive(Parser)]
#[clap(author, version, about)]
struct Cli {
    /// input file path.
    input: String,

    /// output file path. if "-", write to stdout.
    output: String,

    /// minimum frequency in Hz.
    fmin: f64,

    /// maximum frequency in Hz.
    fmax: f64,

    /// frame length in milliseconds.
    #[clap(short, long, default_value_t = 80f64)]
    frame_ms: f64,

    /// length of the window for calculating autocorrelation in milliseconds. [default: frame_ms / 2]
    #[clap(long)]
    win_ms: Option<f64>,

    /// gap between adjacent pYIN predictions in milliseconds. [default: frame_ms / 4]
    #[clap(long)]
    hop_ms: Option<f64>,

    /// Resolution of the pitch bins. 0.01 corresponds to cents. [default: 0.1]
    #[clap(long)]
    resolution: Option<f64>,
    /*  #[clap(short, long)]
    sr: Option<u32>,*/
    /// Whether print results to stdout or not. if output is "-", this argument is ignored.
    #[clap(short, long)]
    verbose: bool,
}
fn decode_audio_file(file: File) -> Result<(Array2<f32>, u32), DecoderError> {
    let source = Decoder::new(io::BufReader::new(file))?;
    let sr = source.sample_rate();
    let channels = source.channels() as usize;
    let mut vec: Vec<_> = source.collect();
    if vec.len() < channels {
        (vec.len()..channels).into_iter().for_each(|_| vec.push(0.));
    }

    let shape = (channels, vec.len() / channels);
    vec.truncate(shape.0 * shape.1); // defensive code
    let wav = Array2::from_shape_vec(shape.strides((1, shape.0)), vec).unwrap();
    Ok((wav, sr))
}

fn main() {
    let cli = Cli::parse();
    let (wav, sr) = if &cli.input == "-" {
        // let mut input = std::io::stdin();
        unimplemented!()
    } else {
        let file = File::open(&cli.input)
            .unwrap_or_else(|_| panic!("Inpuf file \"{}\" not found!", &cli.input));

        decode_audio_file(file).expect("Failed to decode audio file!")
    };
    let output_writer = io::BufWriter::new(if &cli.output == "-" {
        Box::new(std::io::stdout()) as Box<dyn io::Write>
    } else {
        Box::new(
            std::fs::File::create(&cli.output)
                .unwrap_or_else(|_| panic!("Could not create output file \"{}\"!", &cli.output)),
        ) as Box<dyn io::Write>
    });

    let wav = wav.mapv(|x| x as f64);
    let ms_to_samples = |ms: f64| (sr as f64 * ms / 1000.).round() as usize;
    let frame_length = ms_to_samples(cli.frame_ms);
    let win_length = cli.win_ms.map(ms_to_samples);
    let hop_length = cli.hop_ms.map(ms_to_samples);
    let pyin_exec = PYINExecutor::new(
        cli.fmin,
        cli.fmax,
        sr,
        frame_length,
        win_length,
        hop_length,
        cli.resolution,
    );
    let results: Vec<_> = wav
        .axis_iter(Axis(0))
        .par_bridge()
        .map(|mono| {
            pyin_exec
                .clone()
                .pyin(mono.into(), f64::NAN, true, PadMode::Constant(0.))
        })
        .collect();
    let pyin_result =
        Array3::from_shape_fn(
            (3, results.len(), results[0].0.len()),
            |(i, j, k)| match i {
                0 => results[j].0[k],
                1 => results[j].1[k] as usize as f64,
                2 => results[j].2[k],
                _ => unreachable!(),
            },
        );

    if cli.verbose && &cli.output != "-" {
        println!("f0 = {}", pyin_result.index_axis(Axis(0), 0));
        println!("voiced_flag = {}", pyin_result.index_axis(Axis(0), 1));
        println!("voiced_prob = {}", pyin_result.index_axis(Axis(0), 2));
    }
    pyin_result
        .write_npy(output_writer)
        .expect("Failed to write pyin result to file!");
}
