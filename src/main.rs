use std::fs::File;
use std::io;

use clap::Parser;
use ndarray::prelude::*;
use ndarray::{stack, CowArray};
use ndarray_npy::WriteNpyExt;
use rodio::decoder::DecoderError;
use rodio::{Decoder, Source};

use pyin::{pad::PadMode, PYinExecutor};

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
    let mut vec: Vec<f32> = source.collect();
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
        let file =
            File::open(&cli.input).expect(&format!("Inpuf file \"{}\" not found!", &cli.input));

        decode_audio_file(file).expect("Failed to decode audio file!")
    };
    if wav.shape()[0] != 1 {
        unimplemented!("Only mono files are supported!");
    }
    let output_writer = io::BufWriter::new(if &cli.output == "-" {
        Box::new(std::io::stdout()) as Box<dyn io::Write>
    } else {
        Box::new(std::fs::File::create(&cli.output).expect(&format!(
            "Could not create output file \"{}\"!",
            &cli.output
        ))) as Box<dyn io::Write>
    });

    let wav: Array1<f64> = wav.slice(s![0, ..]).mapv(|x| x as f64);
    let wav = CowArray::from(wav);
    let ms_to_samples = |ms: f64| (sr as f64 * ms / 1000.).round() as usize;
    let frame_length = ms_to_samples(cli.frame_ms);
    let win_length = cli.win_ms.map(ms_to_samples);
    let hop_length = cli.hop_ms.map(ms_to_samples);
    let mut pyin_exec = PYinExecutor::new(
        cli.fmin,
        cli.fmax,
        sr,
        frame_length,
        win_length,
        hop_length,
        cli.resolution,
    );
    let (f0, voiced_flag, voiced_prob) = pyin_exec.pyin(wav, f64::NAN, true, PadMode::Reflect);

    if cli.verbose && &cli.output != "-" {
        println!("f0 = {}", f0);
        println!("voiced_flag = {}", voiced_flag);
        println!("voiced_prob = {}", voiced_prob);
    }
    let voiced_flag_f64 =
        Array1::<f64>::from_shape_fn(voiced_flag.raw_dim(), |i| voiced_flag[i] as usize as f64);
    let pyin_result = stack![Axis(0), f0, voiced_flag_f64, voiced_prob];
    pyin_result
        .write_npy(output_writer)
        .expect("Failed to write pyin result to file!");
}
