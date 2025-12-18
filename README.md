# WGPU audio spectrogram demo

This is a simple demo with Rust and WGPU that uses WGSL to perform a GPU-accelerated FFT on incoming audio.

<img width="1552" height="917" alt="spectrogram_demo" src="https://github.com/user-attachments/assets/77c9fc37-27b6-42f6-8479-8fa4dddd0a31" />

## Running
Just do `cargo run --release`. Requires a GUI.

## Notes
This applicaton uses [your default audio input device returned by cpal](https://docs.rs/cpal/latest/cpal/traits/trait.HostTrait.html#tymethod.default_input_device).
