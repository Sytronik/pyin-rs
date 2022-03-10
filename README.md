# pYin algorithm written in Rust

The implementation is based on [librosa](https://librosa.org/doc/0.8.1/_modules/librosa/core/pitch.html#pyin).

## Run

```
cargo run --release <input_file> <output_npy_file> fmin fmax --frame_ms <frame length in miliseconds>
```

## Note

- Supported audio files: the same as [Rodio](https://github.com/RustAudio/rodio)
- output file: npy file contains an ndarray (shape=(3, no. of frames), data=[f0_array, voiced_flag_array, voiced_probs_array])
- If "-" is used as the output filename, the app will send output data to stdout.

## TODO

- Multi-channel audio support
- Input from stdio
- More options supported by command-line arguments
