# pYIN algorithm written in Rust

[![Crates.io Version](https://img.shields.io/crates/v/pyin.svg)](https://crates.io/crates/pyin)
[![Docs.rs](https://docs.rs/pyin/badge.svg)](https://docs.rs/pyin)
[![Crates.io Downloads](https://img.shields.io/crates/d/pyin.svg)](https://crates.io/crates/pyin)

This crate provides a pitch estimate for each frame of the audio signal and a probability the frame is voiced region.

The implementation is based on [librosa](https://librosa.org/doc/0.9.1/_modules/librosa/core/pitch.html#pyin).
For easy translation from Python + Numpy to Rust, the implementation is written on top of [ndarray](https://crates.io/crates/ndarray) crate.

## Build & Run

You can use this both as a executable binary and as a library (C shared library and Rust library).

### As an executable binary

```
cargo run --release <input_file> <output_npy_file> <fmin> <fmax> --frame_ms <frame length in miliseconds>
```

or

```
cargo build --release
./target/release/pyin <input_file> <output_npy_file> <fmin> <fmax> --frame_ms <frame length in miliseconds>
```

#### Note

- Supported audio files: the same as [Creak](https://crates.io/crates/creak) crate.
  - Multi-channel audio files are supported.
- output file: npy file contains an ndarray (shape=(3, no. of channels in input audio, no. of frames), data=[f0_array, voiced_flag_array, voiced_probs_array])
- If "-" is used as the output filename, the app will send output data to stdout.

### Example using pYIN as a C shared library

The example is in `test/test.c`. To build and run it with GCC,

```
./compile_test.sh
LD_LIBRARY_PATH=target/release ./test_pyin
```

### Using pYIN as a Rust library

Add the following to your `Cargo.toml`:

```
[dependencies]
pyin = "1.0"
```

## TODO

- Input from stdio
- More options supported by command-line arguments
