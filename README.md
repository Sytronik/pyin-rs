# pYIN algorithm written in Rust

[![Crates.io Version](https://img.shields.io/crates/v/pyin.svg)](https://crates.io/crates/pyin)
[![Docs.rs](https://docs.rs/pyin/badge.svg)](https://docs.rs/pyin)
[![Crates.io Downloads](https://img.shields.io/crates/d/pyin.svg)](https://crates.io/crates/pyin)

This crate provides a pitch estimate for each frame of the audio signal and a probability that the frame is a voiced region.

The implementation is based on [librosa](https://librosa.org/doc/0.9.1/_modules/librosa/core/pitch.html#pyin).
For easy translation from Python + Numpy to Rust, the implementation is written on top of [ndarray](https://crates.io/crates/ndarray) crate.


## Download & Run

You can download the executable binary from the [Releases](https://github.com/Sytronik/pyin-rs/releases) page.

```
pyin <input_file> <output_npy_file> <fmin> <fmax> --frame_ms <frame length in miliseconds>
```

#### Note

- Supported audio files: the same as [Creak](https://crates.io/crates/creak) crate.
  - Multi-channel audio files are supported.
- output file: npy file contains the output ndarray with
  - shape: (4, no. of channels in input audio, no. of frames)
  - [0, :, :]: timestamp [sec]
  - [1, :, :]: f0 array [Hz]
  - [2, :, :]: voiced flag(1.0 for voiced, 0.0 for unvoiced) array
  - [3, :, :]: voiced probability array
- If "-" is used as the output filename, the app will send output data to stdout.

## Build & Run

You can use this both as an executable binary and as a library (C shared library and Rust library).
When you build, you can use BLAS by turning on the `blas` feature flag.

### As an executable binary

```
cargo run -F build-binary[,blas] --release <input_file> <output_npy_file> <fmin> <fmax> --frame_ms <frame length in miliseconds>
```

or

```
cargo build -F build-binary[,blas] --release
./target/release/pyin <input_file> <output_npy_file> <fmin> <fmax> --frame_ms <frame length in miliseconds>
```

### Example using pYIN as a C shared library

The example is in `test/test.c`. To build and run it with GCC,

```
./compile_test.sh [blas]
LD_LIBRARY_PATH=target/release ./test_pyin
```

### Using pYIN as a Rust library

Add the following to your `Cargo.toml`:

```
[dependencies]
pyin = "1.2.0"
# or, pyin = {version = "1.2.0", features = ["blas"]}
```

## TODO

- Input from stdio
- More options supported by command-line arguments
