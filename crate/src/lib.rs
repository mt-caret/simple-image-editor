#[macro_use]
extern crate cfg_if;
extern crate web_sys;
extern crate wasm_bindgen;

use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, ImageData};

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

pub fn convolve(image_data: &ImageData, kernel: &Kernel) -> Vec<u8> {
    let source = image_data.data().to_vec();
    let mut target = Vec::with_capacity(source.len());
    let (w, h) = (image_data.width() as isize, image_data.height() as isize);
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
pub fn luma_convert(image_data: &ImageData) -> Vec<u8> {
    let source = image_data.data().to_vec();
    let mut target = Vec::with_capacity(source.len());
    let (w, h) = (image_data.width(), image_data.height());

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

//const DENSITY_PATTERN: [u8; 16] = [10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 6, 14, 12, 0];
//
//fn get_density_pattern(index: usize) -> u16 {
//    if index == 0 {
//        return 0;
//    }
//   (0..(index - 1)).map(|i| 1 << DENSITY_PATTERN[i]).sum()
//}

#[wasm_bindgen]
pub fn init() {
    set_panic_hook();
}

#[wasm_bindgen]
pub fn run_convolution(
    src_ctx: &CanvasRenderingContext2d,
    target_ctx: &CanvasRenderingContext2d,
    width: u32,
    height: u32,
    kernel: &Kernel,
) -> Result<(), JsValue> {
    let (w, h) = (width.into(), height.into());
    let image_data = src_ctx.get_image_data(0.0, 0.0, w, h)?;
    let mut result = convolve(&image_data, &kernel);
    let new_image_data =
        ImageData::new_with_u8_clamped_array_and_sh(Clamped(&mut result), width, height)?;
    target_ctx.put_image_data(&new_image_data, 0.0, 0.0)
}

#[wasm_bindgen]
pub fn run_luma_conversion(
    src_ctx: &CanvasRenderingContext2d,
    target_ctx: &CanvasRenderingContext2d,
    width: u32,
    height: u32,
) -> Result<(), JsValue> {
    let (w, h) = (width.into(), height.into());
    let image_data = src_ctx.get_image_data(0.0, 0.0, w, h)?;
    let mut result = luma_convert(&image_data);
    let new_image_data =
        ImageData::new_with_u8_clamped_array_and_sh(Clamped(&mut result), width, height)?;
    target_ctx.put_image_data(&new_image_data, 0.0, 0.0)
}
