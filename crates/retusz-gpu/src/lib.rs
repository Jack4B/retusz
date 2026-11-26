//! Retusz GPU acceleration via `wgpu` (DirectX 12 on Windows, Vulkan/GL on Linux)
//! with automatic CPU `rayon` fallback when running headless without a GPU adapter.
//! Author: Jack4B.

use bytemuck::{Pod, Zeroable};
use retusz_core::color::Lut3D;
use wgpu::util::DeviceExt;

pub const SHADER_MAIN_WGSL: &str = include_str!("shaders/render_main.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    width: u32,
    height: u32,
    lut_n: u32,
    show_clip: u32,
    vignette: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    pub adapter_name: String,
}

impl GpuRenderer {
    /// Try to initialize a `wgpu` compute device (DX12 / Vulkan / GL / software adapter).
    pub fn try_new() -> Option<Self> {
        pollster::block_on(Self::try_new_async())
    }

    async fn try_new_async() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("retusz_render_main"),
            source: wgpu::ShaderSource::Wgsl(SHADER_MAIN_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("retusz_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("retusz_pipe_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("retusz_compute_pipe"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self { device, queue, pipeline, bgl, adapter_name })
    }

    /// Execute the WGSL tetrahedral LUT + spatial gain compute shader on the GPU.
    pub fn render_lut_gain(
        &self,
        rgb8: &[u8],
        w: usize,
        h: usize,
        lut: &Lut3D,
        gain: &[f32],
        vignette: f32,
        show_clip: bool,
    ) -> Vec<u8> {
        let n = w * h;
        if n == 0 {
            return vec![];
        }
        let mut src_packed = vec![0u32; n];
        for i in 0..n {
            let r = rgb8[i * 3] as u32;
            let g = rgb8[i * 3 + 1] as u32;
            let b = rgb8[i * 3 + 2] as u32;
            src_packed[i] = r | (g << 8) | (b << 16) | (255 << 24);
        }
        let lut_vec4: Vec<[f32; 4]> = lut.data.iter().map(|c| [c[0], c[1], c[2], 0.0]).collect();
        let u = Uniforms {
            width: w as u32,
            height: h as u32,
            lut_n: lut.n as u32,
            show_clip: u32::from(show_clip),
            vignette,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        let u_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("u_buf"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let src_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("src_buf"),
            contents: bytemuck::cast_slice(&src_packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lut_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lut_buf"),
            contents: bytemuck::cast_slice(&lut_vec4),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gain_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gain_buf"),
            contents: bytemuck::cast_slice(gain),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let byte_size = (n * std::mem::size_of::<u32>()) as u64;
        let dst_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dst_buf"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read_buf"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("retusz_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lut_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gain_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dst_buf.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            let gx = ((w as u32) + 15) / 16;
            let gy = ((h as u32) + 15) / 16;
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&dst_buf, 0, &read_buf, 0, byte_size);
        self.queue.submit(Some(enc.finish()));

        let slice = read_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let out_packed: &[u32] = bytemuck::cast_slice(&data);
        let mut out_rgb8 = vec![0u8; n * 3];
        for i in 0..n {
            let p = out_packed[i];
            out_rgb8[i * 3] = (p & 0xFF) as u8;
            out_rgb8[i * 3 + 1] = ((p >> 8) & 0xFF) as u8;
            out_rgb8[i * 3 + 2] = ((p >> 16) & 0xFF) as u8;
        }
        drop(data);
        read_buf.unmap();
        out_rgb8
    }
}

/// CPU fallback matching `SHADER_MAIN_WGSL` to <= 1/255 when running on systems without GPU.
pub fn render_lut_gain_cpu(
    rgb8: &[u8],
    w: usize,
    h: usize,
    lut: &Lut3D,
    gain: &[f32],
    vignette: f32,
    show_clip: bool,
) -> Vec<u8> {
    let n = w * h;
    let mut out = vec![0u8; n * 3];
    for y in 0..h {
        let ny = (y as f32 / h.max(1) as f32) - 0.5;
        for x in 0..w {
            let nx = (x as f32 / w.max(1) as f32) - 0.5;
            let idx = y * w + x;
            let s = [
                rgb8[idx * 3] as f32 / 255.0,
                rgb8[idx * 3 + 1] as f32 / 255.0,
                rgb8[idx * 3 + 2] as f32 / 255.0,
            ];
            let mut c = lut.sample_tetra(s);
            let mut g = 1.0 + gain[idx];
            if vignette.abs() > 1e-4 {
                let r2 = 2.2 * (nx * nx + ny * ny);
                g *= (1.0 + 0.55 * vignette * r2 * r2).clamp(0.25, 1.75);
            }
            if (g - 1.0).abs() > 1e-4 {
                for ch in &mut c {
                    let v_lin = *ch * g;
                    if g <= 1.0 || v_lin <= 0.72 {
                        *ch = v_lin.clamp(0.0, 1.0);
                    } else {
                        let k = ch.max(0.72);
                        let d = v_lin - k;
                        let hr = (1.0 - k).max(0.0);
                        *ch = (k + (d * hr) / (d + hr + 1e-6)).clamp(0.0, 1.0);
                    }
                }
            }
            let mut ro = (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let mut go = (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            let mut bo = (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            if show_clip {
                if ro >= 254 || go >= 254 || bo >= 254 {
                    ro = 245; go = 55; bo = 55;
                } else if ro <= 1 && go <= 1 && bo <= 1 {
                    ro = 55; go = 135; bo = 250;
                }
            }
            out[idx * 3] = ro;
            out[idx * 3 + 1] = go;
            out[idx * 3 + 2] = bo;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_or_cpu_lut_shader_pipeline() {
        let (w, h) = (32usize, 24usize);
        let mut src = vec![0u8; w * h * 3];
        for i in 0..(w * h) {
            src[i * 3] = ((i * 7) % 256) as u8;
            src[i * 3 + 1] = ((i * 13) % 256) as u8;
            src[i * 3 + 2] = ((i * 19) % 256) as u8;
        }
        let lut = Lut3D::identity(33);
        let gain = vec![0.05f32; w * h];
        let cpu_out = render_lut_gain_cpu(&src, w, h, &lut, &gain, -0.1, false);
        assert_eq!(cpu_out.len(), src.len());

        if let Some(gpu) = GpuRenderer::try_new() {
            let gpu_out = gpu.render_lut_gain(&src, w, h, &lut, &gain, -0.1, false);
            for i in 0..src.len() {
                assert!(
                    (cpu_out[i] as i32 - gpu_out[i] as i32).abs() <= 1,
                    "mismatch at {i}: cpu={} gpu={}",
                    cpu_out[i],
                    gpu_out[i]
                );
            }
        }
    }
}
