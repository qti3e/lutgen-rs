use std::simd::{f32x4, num::SimdFloat, LaneCount, Simd, SupportedLaneCount};

use arrayvec::ArrayVec;
use image::Rgb;
use oklab::RGB;

use super::InterpolatedRemapper;
use crate::{GenerateLut, Image};

/// Simple remapper that doesn't do any interpolation. Mostly used internally by the other
/// algorithms.
pub struct NearestNeighborRemapper2<'a> {
    palette: &'a [[u8; 3]],

    palette_linearx4: Vec<[f32x4; 3]>,
    palette_linear: ArrayVec<[f32; 3], 3>,

    lum_factor: f64,
}

impl<'a> NearestNeighborRemapper2<'a> {
    pub fn new(palette: &'a [[u8; 3]], lum_factor: f64) -> Self {
        let mut palette_linear = ArrayVec::new();
        let mut palette_linearx4 = Vec::new();

        for p in palette.chunks(4) {
            // last chunk
            if p.len() != 4 {
                palette_linear.extend(p.iter().map(|c| c.map(to_linear8)));
                break;
            }

            let r = f32x4::from_array([p[0][0], p[1][0], p[2][0], p[3][0]].map(to_linear8));
            let g = f32x4::from_array([p[0][1], p[1][1], p[2][1], p[3][1]].map(to_linear8));
            let b = f32x4::from_array([p[0][2], p[1][2], p[2][2], p[3][2]].map(to_linear8));
            palette_linearx4.push([r, g, b]);
        }

        Self {
            palette,
            palette_linearx4,
            palette_linear,
            lum_factor,
        }
    }
}

impl<'a> GenerateLut<'a> for NearestNeighborRemapper2<'a> {}
impl<'a> InterpolatedRemapper<'a> for NearestNeighborRemapper2<'a> {
    fn remap_image(&self, image: &mut Image) {
        for pixel in image.pixels_mut() {
            self.remap_pixel(pixel)
        }
    }

    fn remap_pixel(&self, pixel: &mut Rgb<u8>) {
        let linear = [
            to_linear8(pixel[0]),
            to_linear8(pixel[1]),
            to_linear8(pixel[2]),
        ];

        let item = linear_rgb_find_nearest_neighbour(
            self.lum_factor as f32,
            &linear,
            &self.palette_linearx4,
            &self.palette_linear,
        );

        *pixel = Rgb(self.palette[item as usize]);
    }
}

#[inline(always)]
fn to_linear8(c: u8) -> f32 {
    to_linear(c as f32 / 255.0)
}

#[inline(always)]
fn to_linear(u: f32) -> f32 {
    if u >= 0.04045 {
        ((u + 0.055) / (1. + 0.055)).powf(2.4)
    } else {
        u / 12.92
    }
}

fn linear_rgb_find_nearest_neighbour(
    lum_factor: f32,
    pixel: &[f32; 3],
    palette_linearx4: &[[f32x4; 3]],
    palette_linear: &[[f32; 3]],
) -> usize {
    let mut simd_min_index = 0;
    let mut last_min_dist = f32x4::splat(0.0);

    let mut current_min_dist = f32::MAX;

    let a = [
        f32x4::splat(pixel[0]),
        f32x4::splat(pixel[1]),
        f32x4::splat(pixel[2]),
    ];

    for (i, p) in palette_linearx4.iter().enumerate() {
        let out = linear_srgb_oklab_distance_f32x4(lum_factor, &a, p);
        if out.reduce_min() < current_min_dist {
            simd_min_index = i;
            last_min_dist = out;
        }
    }

    let mut min_index = usize::MAX;
    for (i, p) in palette_linear.iter().enumerate() {
        let dist = linear_srgb_oklab_distance(lum_factor, pixel, p);
        if dist < current_min_dist {
            min_index = i + palette_linearx4.len() * 4;
            current_min_dist = dist;
        }
    }

    if min_index < usize::MAX {
        return min_index;
    }

    simd_min_index * 4
        + unsafe {
            last_min_dist
                .as_array()
                .iter()
                .position(|x| x == &current_min_dist)
                .unwrap_unchecked()
        }
}

pub fn linear_srgb_oklab_distance(lum_factor: f32, a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let c = RGB {
        r: a[0] - b[0],
        g: a[1] - b[1],
        b: a[2] - b[2],
    };

    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;

    let l_ = (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s) * lum_factor;
    let a_ = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    let b_ = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;

    l_ * l_ + a_ * a_ + b_ * b_
}

pub fn linear_srgb_oklab_distance_f32x4(lum_factor: f32, a: &[f32x4; 3], b: &[f32x4; 3]) -> f32x4 {
    let r = a[0] - b[0];
    let g = a[1] - b[1];
    let b = a[2] - b[2];

    let l = f32x4::splat(0.4122214708) * r
        + f32x4::splat(0.5363325363) * g
        + f32x4::splat(0.0514459929) * b;
    let m = f32x4::splat(0.2119034982) * r
        + f32x4::splat(0.6806995451) * g
        + f32x4::splat(0.1073969566) * b;
    let s = f32x4::splat(0.0883024619) * r
        + f32x4::splat(0.2817188376) * g
        + f32x4::splat(0.6299787005) * b;

    let l_ = (f32x4::splat(0.2104542553) * l + f32x4::splat(0.7936177850) * m
        - f32x4::splat(0.0040720468) * s)
        * f32x4::splat(lum_factor);
    let a_ = f32x4::splat(1.9779984951) * l - f32x4::splat(2.4285922050) * m
        + f32x4::splat(0.4505937099) * s;
    let b_ = f32x4::splat(0.0259040371) * l + f32x4::splat(0.7827717662) * m
        - f32x4::splat(0.8086757660) * s;

    l_ * l_ + a_ * a_ + b_ * b_
}

pub fn linear_srgb_oklab_distance_simd<const N: usize>(
    lum_factor: f32,
    a: &[Simd<f32, N>; 3],
    b: &[Simd<f32, N>; 3],
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    Simd::<f32, N>::splat(0.0)
}
