#[macro_use]
extern crate cfg_if;
extern crate js_sys;
extern crate wasm_bindgen;
extern crate web_sys;
#[macro_use]
extern crate lazy_static;

use js_sys::Object;
use wasm_bindgen::prelude::*;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

cfg_if! {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function to get better error messages if we ever panic.
    if #[cfg(feature = "console_error_panic_hook")] {
        extern crate console_error_panic_hook;
        use console_error_panic_hook::set_once as set_panic_hook;
    } else {
        #[inline]
        fn set_panic_hook() {}
    }
}

cfg_if! {
    // When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
    // allocator.
    if #[cfg(feature = "wee_alloc")] {
        extern crate wee_alloc;
        #[global_allocator]
        static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
    }
}

// Called by our JS entry point to run the example.
#[wasm_bindgen]
pub fn run() -> Result<(), JsValue> {
    set_panic_hook();

    let window = web_sys::window().expect("should have a Window");
    let document = window.document().expect("should have a Document");

    let p: web_sys::Node = document.create_element("p")?.into();
    p.set_text_content(Some("Hello from Rust, WebAssembly, and Webpack!"));

    let body = document.body().expect("should have a body");
    let body: &web_sys::Node = body.as_ref();
    body.append_child(&p)?;

    Ok(())
}

#[wasm_bindgen]
pub struct Kernel {
    size: isize,
    content: Vec<f32>,
}

#[wasm_bindgen]
impl Kernel {
    pub fn new(size: isize) -> Kernel {
        assert!(size % 2 == 1);
        let content = vec![0.0; (size * size) as usize];
        Kernel { size, content }
    }

    pub fn content(&self) -> *const f32 {
        self.content.as_ptr()
    }
}

pub fn convolve(source: Vec<u8>, w: u32, h: u32, kernel: &Kernel) -> Vec<u8> {
    let mut target = Vec::with_capacity(source.len());
    let (w, h) = (w as isize, h as isize);
    let offset = kernel.size / 2;

    for y in 0..h {
        for x in 0..w {
            let mut rgb = [0.0; 3];
            for dy in 0..kernel.size {
                let ny = y + dy - offset;
                if ny < 0 || ny >= h {
                    continue;
                }
                for dx in 0..kernel.size {
                    let nx = x + dx - offset;
                    if nx < 0 || nx >= w {
                        continue;
                    }
                    let base_index = ((ny * w + nx) * 4) as usize;
                    let weight = kernel.content[(dy * kernel.size + dx) as usize];
                    for i in 0..3 {
                        rgb[i] += source[base_index + i] as f32 * weight;
                    }
                }
            }
            for i in 0..3 {
                target.push(rgb[i] as u8);
            }
            target.push(source[((y * w + x) * 4 + 3) as usize]);
        }
    }
    target
}

// c.f. https://en.wikipedia.org/wiki/Luma_%28video%29
pub fn luma_convert(source: Vec<u8>, w: u32, h: u32) -> Vec<u8> {
    let mut target = Vec::with_capacity(source.len());

    for y in 0..h {
        for x in 0..w {
            let base_index = ((y * w + x) * 4) as usize;
            let luma_value = ((source[base_index] as f32) * 0.2126
                + (source[base_index + 1] as f32) * 0.7152
                + (source[base_index + 2] as f32) * 0.0722) as u8;
            target.push(luma_value);
            target.push(luma_value);
            target.push(luma_value);
            target.push(source[base_index + 3]);
        }
    }
    target
}

lazy_static! {
    static ref DENSITY_PATTERNS: [u16; 17] = {
        let pattern_nums = [10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 6, 14, 12, 0];
        let mut ret = [0; 17];
        for i in 1..17 {
            ret[i] = ret[i - 1] + (1 << pattern_nums[i - 1]);
        }
        ret
    };
}

pub fn density_pattern_halftone(source: Vec<u8>, w: u32, h: u32) -> Vec<u8> {
    let mut target = vec![0; source.len() * 16];

    for y in 0..h {
        for x in 0..w {
            let source_base_index = ((y * w + x) * 4) as usize;
            let luma_value = ((source[source_base_index] as f32) * 0.2126
                + (source[source_base_index + 1] as f32) * 0.7152
                + (source[source_base_index + 2] as f32) * 0.0722)
                as u8;
            // this converts 0~255 to 0~17
            let density_pattern = DENSITY_PATTERNS[(luma_value as f32 / 16.0).round() as usize];
            for dy in 0..4 {
                for dx in 0..4 {
                    let color_value = if density_pattern & (1 << (dy * 4 + dx)) != 0 {
                        255
                    } else {
                        0
                    };
                    let target_base_index = (((y * 4 + dy) * (w * 4) + x * 4 + dx) * 4) as usize;
                    target[target_base_index] = color_value;
                    target[target_base_index + 1] = color_value;
                    target[target_base_index + 2] = color_value;
                    target[target_base_index + 3] = source[source_base_index + 3];
                }
            }
        }
    }
    target
}

#[wasm_bindgen]
pub fn init() {
    set_panic_hook();
}

fn get_context(canvas: &HtmlCanvasElement) -> Result<CanvasRenderingContext2d, Object> {
    canvas
        .get_context("2d")?
        .expect("Failed to unwrap CanvasRenderingContext2d")
        .dyn_into()
}

fn run_image_conversion(
    src_canvas: &HtmlCanvasElement,
    target_canvas: &HtmlCanvasElement,
    f: impl Fn(Vec<u8>, u32, u32) -> (Vec<u8>, u32, u32),
) -> Result<(), JsValue> {
    let (w, h) = (src_canvas.width(), src_canvas.height());
    let src_ctx = get_context(src_canvas)?;
    let image_vec = src_ctx
        .get_image_data(0.0, 0.0, w as f64, h as f64)?
        .data()
        .to_vec();

    let (mut new_image_vec, new_w, new_h) = f(image_vec, w, h);

    target_canvas.set_width(new_w);
    target_canvas.set_height(new_h);
    let target_ctx = get_context(target_canvas)?;
    let new_image_data =
        ImageData::new_with_u8_clamped_array_and_sh(Clamped(&mut new_image_vec), new_w, new_h)?;
    target_ctx.put_image_data(&new_image_data, 0.0, 0.0)
}

#[wasm_bindgen]
pub fn run_convolution(
    src_canvas: &HtmlCanvasElement,
    target_canvas: &HtmlCanvasElement,
    kernel: &Kernel,
) -> Result<(), JsValue> {
    run_image_conversion(src_canvas, target_canvas, |vec, w, h| {
        (convolve(vec, w, h, kernel), w, h)
    })
}

#[wasm_bindgen]
pub fn run_luma_conversion(
    src_canvas: &HtmlCanvasElement,
    target_canvas: &HtmlCanvasElement,
) -> Result<(), JsValue> {
    run_image_conversion(src_canvas, target_canvas, |vec, w, h| {
        (luma_convert(vec, w, h), w, h)
    })
}

#[wasm_bindgen]
pub fn run_density_pattern_halftone(
    src_canvas: &HtmlCanvasElement,
    target_canvas: &HtmlCanvasElement,
) -> Result<(), JsValue> {
    let result = run_image_conversion(src_canvas, target_canvas, |vec, w, h| {
        (density_pattern_halftone(vec, w, h), w * 4, h * 4)
    });
    result
}
