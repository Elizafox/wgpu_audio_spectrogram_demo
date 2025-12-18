use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{HeapCons, HeapRb, traits::*};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::*,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftParams {
    n: u32,
    m: u32,
    mh: u32,
    direction: i32,    // -1 forward, +1 inverse
    batch_stride: u32, // n (batched = 1)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WindowParams {
    n: u32,
    batch_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Mag2TexParams {
    n: u32,
    bins: u32,
    col: u32,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct View {
    width: u32,
    offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Range {
    k0: u32,
    k1: u32,
}

fn hann(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * (i as f32) / denom).cos())
        .collect()
}

fn make_twiddles(n: usize) -> Vec<[f32; 2]> {
    (0..n)
        .map(|k| {
            let th = -(2.0 * std::f32::consts::PI) * (k as f32) / (n as f32);
            [th.cos(), th.sin()]
        })
        .collect()
}

fn build_input_stream() -> anyhow::Result<AudioIn> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device found"))?;
    let cfg = dev.default_input_config()?;
    let sample_rate = cfg.sample_rate().0;
    let channels = cfg.channels() as usize;

    // 1–2 sec of ring buffer
    let rb = HeapRb::<f32>::new((sample_rate as usize) * 2);
    let (mut prod, cons) = rb.split();

    // Build the stream
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream(
            &cfg.clone().into(),
            move |data: &[f32], _| {
                let mut mono_buf = Vec::with_capacity(data.len() / channels);
                for frame in data.chunks(channels) {
                    mono_buf.push(frame.iter().copied().sum::<f32>() / (channels as f32));
                }
                let _ = prod.push_slice(&mono_buf);
            },
            move |err| eprintln!("cpal error: {err}"),
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream(
            &cfg.clone().into(),
            move |data: &[i16], _| {
                let mut mono_buf = Vec::with_capacity(data.len() / channels);
                for frame in data.chunks(channels) {
                    let sum = frame
                        .iter()
                        .map(|&x| x as f32 / i16::MAX as f32)
                        .sum::<f32>();
                    mono_buf.push(sum / (channels as f32));
                }
                let _ = prod.push_slice(&mono_buf);
            },
            move |err| eprintln!("cpal error: {err}"),
            None,
        )?,
        cpal::SampleFormat::U16 => dev.build_input_stream(
            &cfg.clone().into(),
            move |data: &[u16], _| {
                let mut mono_buf = Vec::with_capacity(data.len() / channels);
                for frame in data.chunks(channels) {
                    let sum = frame
                        .iter()
                        .map(|&x| (x as f32 / u16::MAX as f32) * 2.0 - 1.0)
                        .sum::<f32>();
                    mono_buf.push(sum / (channels as f32));
                }
                let _ = prod.push_slice(&mono_buf);
            },
            move |err| eprintln!("cpal error: {err}"),
            None,
        )?,
        other => return Err(anyhow::anyhow!("Unsupported format {other:?}")),
    };

    stream.play()?;

    Ok(AudioIn {
        _stream: stream,
        rb_cons: cons,
    })
}

struct Gpu {
    // We leak the window (it's open the entire life of the app), so the surface can be 'static
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    _surface_format: wgpu::TextureFormat,

    // pipelines/layouts
    win_pipe: wgpu::ComputePipeline,
    win_bg: wgpu::BindGroup,

    fft_pipe: wgpu::ComputePipeline,
    fft_bgl: wgpu::BindGroupLayout,
    fft_params_buf: wgpu::Buffer,
    tw_buf: wgpu::Buffer,

    mag_pipe: wgpu::ComputePipeline,
    mag_bgl: wgpu::BindGroupLayout,
    mag_params_buf: wgpu::Buffer,

    quad_pipe: wgpu::RenderPipeline,
    quad_bg: wgpu::BindGroup,
    view_buf: wgpu::Buffer,

    // data
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    n: usize,
    bins: usize,

    // Spectrogram texture
    spec_view: wgpu::TextureView,
    hist_width: u32,

    // Keep for lifetime completeness (not used after BG creation)
    _win_params_buf: wgpu::Buffer,
}

impl Gpu {
    async fn new(window: &'static Window, n: usize, hist_width: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window).expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            })
            .await
            .expect("request_device failed");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let max_dim = device.limits().max_texture_dimension_2d; // Maximum texture size

        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.min(max_dim),
            height: size.height.min(max_dim),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);

        // buffers
        let total_bytes = (n * std::mem::size_of::<[f32; 2]>()) as u64;
        let stor = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf_a"),
            size: total_bytes,
            usage: stor,
            mapped_at_creation: false,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf_b"),
            size: total_bytes,
            usage: stor,
            mapped_at_creation: false,
        });

        // window kernel
        let win = hann(n);
        let win_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("window"),
            contents: bytemuck::cast_slice(&win),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let win_params = WindowParams {
            n: n as u32,
            batch_stride: n as u32,
        };
        let _win_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("win_params"),
            contents: bytemuck::bytes_of(&win_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let win_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("window"),
            source: wgpu::ShaderSource::Wgsl(include_str!("window.wgsl").into()),
        });
        let win_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("win_bgl"),
            entries: &[
                // X
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // WIN
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // U
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let win_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("win_pl"),
            bind_group_layouts: &[&win_bgl],
            immediate_size: 0,
        });
        let win_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("win_pipe"),
            layout: Some(&win_pl),
            module: &win_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let win_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("win_bg"),
            layout: &win_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: win_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: _win_params_buf.as_entire_binding(),
                },
            ],
        });

        // Stockham
        let tw = make_twiddles(n);
        let tw_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twiddles"),
            contents: bytemuck::cast_slice(&tw),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fft_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_params"),
            size: std::mem::size_of::<FftParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stockham_stage"),
            source: wgpu::ShaderSource::Wgsl(include_str!("stockham_stage.wgsl").into()),
        });
        let fft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fft_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // in
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // out
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // params
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // twiddles
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let fft_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fft_pl"),
            bind_group_layouts: &[&fft_bgl],
            immediate_size: 0,
        });
        let fft_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft_pipe"),
            layout: Some(&fft_pl),
            module: &fft_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Spectrogram texture + mag2tex
        let bins = n / 2 + 1;
        let tex_width = hist_width.min(max_dim);
        let spec_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("spec_tex"),
            size: wgpu::Extent3d {
                width: tex_width,
                height: bins as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let spec_view = spec_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mag_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mag2tex"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mag2tex.wgsl").into()),
        });
        let mag_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mag_bgl"),
            entries: &[
                // FREQ (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Storage texture (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let mag_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mag_pl"),
            bind_group_layouts: &[&mag_bgl],
            immediate_size: 0,
        });
        let mag_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mag_pipe"),
            layout: Some(&mag_pl),
            module: &mag_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let mag_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mag_params"),
            size: std::mem::size_of::<Mag2TexParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Fullscreen quad render
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view"),
            size: std::mem::size_of::<View>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let quad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quad"),
            source: wgpu::ShaderSource::Wgsl(include_str!("quad.wgsl").into()),
        });
        let quad_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quad_bgl"),
            entries: &[
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Sampled texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // View uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let quad_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quad_pl"),
            bind_group_layouts: &[&quad_bgl],
            immediate_size: 0,
        });
        let quad_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("quad_pipe"),
            layout: Some(&quad_pl),
            vertex: wgpu::VertexState {
                module: &quad_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &quad_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });
        let quad_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quad_bg"),
            layout: &quad_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&spec_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: view_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            surface,
            device,
            queue,
            config,
            _surface_format: surface_format,
            win_pipe,
            win_bg,
            fft_pipe,
            fft_bgl,
            fft_params_buf,
            tw_buf,
            mag_pipe,
            mag_bgl,
            mag_params_buf,
            quad_pipe,
            quad_bg,
            view_buf,
            buf_a,
            buf_b,
            n,
            bins,
            spec_view,
            hist_width: tex_width,
            _win_params_buf,
        }
    }

    fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        let max_dim = self.device.limits().max_texture_dimension_2d;
        self.config.width = size.width.min(max_dim);
        self.config.height = size.height.min(max_dim);
        self.surface.configure(&self.device, &self.config);
    }
}

struct AudioIn {
    _stream: cpal::Stream, // Keep it alive
    rb_cons: HeapCons<f32>,
}

struct App {
    n: usize,
    hist_width: u32,

    window: Option<&'static Window>,
    gpu: Option<Gpu>,

    // Synth state
    col: u32,

    last_frame: Instant,
    target_dt: Duration,

    audio: Option<AudioIn>,

    hop: usize,

    // Audio circular buffer
    circ: Vec<f32>,
    widx: usize, // Write index into `circ`
    have: usize, // How many new samples since we last emitted a frame
}

impl App {
    fn new(n: usize, hist_width: u32) -> Self {
        Self {
            n,
            hist_width,
            window: None,
            gpu: None,
            col: 0,
            last_frame: Instant::now(),
            target_dt: Duration::from_millis(16),
            audio: None,
            hop: n / 2, // 50% overlap
            circ: vec![0.0; n * 4],
            widx: 0,
            have: 0,
        }
    }

    fn ingest_audio(&mut self) {
        let cap = self.circ.len();
        let n = self.n;
        let hop = self.hop;

        let mut tmp = vec![0.0f32; 4096];
        loop {
            let read = match self.audio.as_mut() {
                Some(audio) => audio.rb_cons.pop_slice(&mut tmp),
                None => return,
            };
            if read == 0 {
                break;
            }

            // Write chunk into circular buffer
            for &s in &tmp[..read] {
                self.circ[self.widx] = s;
                self.widx = (self.widx + 1) % cap;
            }
            self.have += read;

            while self.have >= hop {
                self.have -= hop;

                // Snapshot one window
                let start = (self.widx + cap - n) % cap;
                let mut frame = vec![0.0f32; n];
                let first = cap - start;
                if first >= n {
                    frame.copy_from_slice(&self.circ[start..start + n]);
                } else {
                    frame[..first].copy_from_slice(&self.circ[start..]);
                    frame[first..].copy_from_slice(&self.circ[..(n - first)]);
                }

                self.process_frame(&frame);
            }
        }
    }

    fn process_frame(&mut self, frame: &[f32]) {
        let gpu = match self.gpu {
            Some(ref g) => g,
            None => return,
        };

        // Pack to complex
        let cx: Vec<[f32; 2]> = frame.iter().map(|&x| [x, 0.0]).collect();
        gpu.queue
            .write_buffer(&gpu.buf_a, 0, bytemuck::cast_slice(&cx));

        // Window pass
        {
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&gpu.win_pipe);
            pass.set_bind_group(0, &gpu.win_bg, &[]);
            let wg = 256u32;
            let groups_x = (gpu.n as u32).div_ceil(wg);
            pass.dispatch_workgroups(groups_x, 1, 1);
            drop(pass);
            gpu.queue.submit(Some(enc.finish()));
            let _ = gpu.device.poll(wgpu::PollType::Poll);
        }

        // FFT passes
        let stages = (gpu.n as f32).log2() as u32;
        let mut in_is_a = true;
        for s in 1..=stages {
            let m = 1u32 << s;
            let mh = m >> 1;
            let p = FftParams {
                n: gpu.n as u32,
                m,
                mh,
                direction: -1,
                batch_stride: gpu.n as u32,
            };
            gpu.queue
                .write_buffer(&gpu.fft_params_buf, 0, bytemuck::bytes_of(&p));

            let (inbuf, outbuf) = if in_is_a {
                (&gpu.buf_a, &gpu.buf_b)
            } else {
                (&gpu.buf_b, &gpu.buf_a)
            };
            let fft_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft_bg"),
                layout: &gpu.fft_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: outbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gpu.fft_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gpu.tw_buf.as_entire_binding(),
                    },
                ],
            });

            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&gpu.fft_pipe);
            pass.set_bind_group(0, &fft_bg, &[]);

            let threads = (gpu.n as u32) / 2;
            let wg = 256u32;
            let groups_x = threads.div_ceil(wg);
            pass.dispatch_workgroups(groups_x, 1, 1);
            drop(pass);

            gpu.queue.submit(Some(enc.finish()));
            let _ = gpu.device.poll(wgpu::PollType::Poll);

            in_is_a = !in_is_a;
        }

        let final_buf = if in_is_a { &gpu.buf_a } else { &gpu.buf_b };

        // Magnitude -> texture
        let mag_params = Mag2TexParams {
            n: gpu.n as u32,
            bins: gpu.bins as u32,
            col: self.col,
            width: gpu.hist_width,
            height: gpu.bins as u32,
        };
        gpu.queue
            .write_buffer(&gpu.mag_params_buf, 0, bytemuck::bytes_of(&mag_params));

        let mag_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mag_bg"),
            layout: &gpu.mag_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: final_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gpu.mag_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gpu.spec_view),
                },
            ],
        });

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&gpu.mag_pipe);
        pass.set_bind_group(0, &mag_bg, &[]);
        let gx = (gpu.bins as u32).div_ceil(64);
        pass.dispatch_workgroups(gx, 1, 1);
        drop(pass);
        gpu.queue.submit(Some(enc.finish()));
        let _ = gpu.device.poll(wgpu::PollType::Poll);

        self.col = (self.col + 1) % gpu.hist_width;
    }

    fn present(&mut self) {
        let gpu = match self.gpu {
            Some(ref g) => g,
            None => return,
        };

        let offset = (self.col + 1) % gpu.hist_width;
        let view = View {
            width: gpu.hist_width,
            offset,
        };
        gpu.queue
            .write_buffer(&gpu.view_buf, 0, bytemuck::bytes_of(&view));

        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                gpu.surface.configure(&gpu.device, &gpu.config);
                gpu.surface.get_current_texture().expect("acquire surface")
            }
        };
        let view_tex = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_tex,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rp.set_pipeline(&gpu.quad_pipe);
            rp.set_bind_group(0, &gpu.quad_bg, &[]);
            rp.draw(0..6, 0..1);
        }
        gpu.queue.submit(Some(enc.finish()));
        frame.present();
    }
}

impl ApplicationHandler for App {
    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(w) = self.window {
            w.request_redraw();
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs: WindowAttributes = Window::default_attributes()
                .with_title("GPU-accelerated Spectrogram")
                .with_inner_size(PhysicalSize::new(1024, 768));
            let win = event_loop.create_window(attrs).expect("create window");
            let win_static: &'static Window = Box::leak(Box::new(win));
            self.window = Some(win_static);

            // init Audio
            let audio = build_input_stream().expect("audio init");

            // init GPU
            let gpu = pollster::block_on(Gpu::new(win_static, self.n, self.hist_width));

            self.audio = Some(audio);
            self.gpu = Some(gpu);

            event_loop.set_control_flow(ControlFlow::Poll);
            win_static.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(win) = self.window else {
            return;
        };

        if win.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => std::process::exit(0),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                if self.last_frame.elapsed() < self.target_dt {
                    return;
                }

                self.last_frame = Instant::now();
                self.ingest_audio();
                self.present();

                if let Some(win) = self.window {
                    win.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    // Bin size
    const N: usize = 1024;

    // Histogram width
    const HIST_WIDTH: u32 = 1024;

    let event_loop = EventLoop::new()?;
    let mut app = App::new(N, HIST_WIDTH);
    event_loop.run_app(&mut app)?;

    Ok(())
}
