use std::simd::{
    cmp::SimdPartialEq, f32x16, f32x4, f32x8, num::SimdFloat, LaneCount, Mask, Simd, SimdElement,
    SupportedLaneCount,
};

use arrayvec::ArrayVec;
use image::Rgb;

use super::InterpolatedRemapper;
use crate::{GenerateLut, Image};

/// Simple remapper that doesn't do any interpolation. Mostly used internally by the other
/// algorithms.
pub struct NearestNeighborRemapper2<'a> {
    palette: &'a [[u8; 3]],
    linear_palette: Palette,
    lum_factor: f64,
}

impl<'a> NearestNeighborRemapper2<'a> {
    pub fn new(palette: &'a [[u8; 3]], lum_factor: f64) -> Self {
        Self {
            palette,
            linear_palette: Palette::from_srgb(palette),
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

        let item = self
            .linear_palette
            .get_nearest_position(self.lum_factor as f32, linear);

        *pixel = Rgb(self.palette[item]);
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

#[derive(Default)]
struct Palette {
    f32x16: Vec<[f32x16; 3]>,
    f32x8: Option<[f32x8; 3]>,
    f32x4: Option<[f32x4; 3]>,
    f32x1: ArrayVec<[f32; 3], 3>,

    // cache of indicies
    ix8: usize, // len(self.f32x16) * 16
    ix4: usize, // self.ix8 + len(self.f32x8) * 8
    ix1: usize, // self.ix4 + len(self.f32x4) * 4
}

impl Palette {
    pub fn from_srgb(mut palette: &[[u8; 3]]) -> Self {
        let mut result = Palette::default();

        fn transpose<const N: usize>(channel: usize, input: &[[u8; 3]]) -> [f32; N] {
            assert_eq!(input.len(), N);
            let mut output = [0.0; N];
            for (i, channels) in input.iter().enumerate() {
                output[i] = to_linear8(channels[channel]);
            }
            output
        }

        while palette.len() >= 16 {
            let items = &palette[..16];
            palette = &palette[16..];
            let r = f32x16::from_array(transpose::<16>(0, items));
            let g = f32x16::from_array(transpose::<16>(1, items));
            let b = f32x16::from_array(transpose::<16>(2, items));
            result.f32x16.push([r, g, b]);
        }

        if palette.len() >= 8 {
            let items = &palette[..8];
            palette = &palette[8..];
            let r = f32x8::from_array(transpose::<8>(0, items));
            let g = f32x8::from_array(transpose::<8>(1, items));
            let b = f32x8::from_array(transpose::<8>(2, items));
            result.f32x8 = Some([r, g, b]);
        }

        if palette.len() >= 4 {
            let items = &palette[..4];
            palette = &palette[4..];
            let r = f32x4::from_array(transpose::<4>(0, items));
            let g = f32x4::from_array(transpose::<4>(1, items));
            let b = f32x4::from_array(transpose::<4>(2, items));
            result.f32x4 = Some([r, g, b]);
        }

        for c in palette {
            result.f32x1.push(c.map(to_linear8));
        }

        result.ix8 = result.f32x16.len() * 16;
        result.ix4 = result.ix8 + if result.f32x8.is_some() { 8 } else { 0 };
        result.ix1 = result.ix4 + if result.f32x4.is_some() { 4 } else { 0 };

        result
    }

    #[inline]
    pub fn get_nearest_position(&self, lum_factor: f32, pixel: [f32; 3]) -> usize {
        enum CurrentMinState {
            Unset,
            X16(f32x16, usize),
            X8(f32x8),
            X4(f32x4),
            X1(usize),
        }

        let mut current_min = f32::MAX;
        let mut current_min_state = CurrentMinState::Unset;

        let pixel_f32x16 = [
            f32x16::splat(pixel[0]),
            f32x16::splat(pixel[1]),
            f32x16::splat(pixel[2]),
        ];

        for (i, x16) in self.f32x16.iter().enumerate() {
            let out = linear_srgb_oklab_distance_simd::<16>(lum_factor, &pixel_f32x16, x16);
            let min_dist = out.reduce_min();
            if min_dist < current_min {
                current_min = min_dist;
                current_min_state = CurrentMinState::X16(out, i);
            }
        }

        if let Some(x8) = &self.f32x8 {
            let pixel_f32x8 = [
                f32x8::splat(pixel[0]),
                f32x8::splat(pixel[1]),
                f32x8::splat(pixel[2]),
            ];
            let out = linear_srgb_oklab_distance_simd::<8>(lum_factor, &pixel_f32x8, x8);
            let min_dist = out.reduce_min();
            if min_dist < current_min {
                current_min = min_dist;
                current_min_state = CurrentMinState::X8(out);
            }
        }

        if let Some(x4) = &self.f32x4 {
            let pixel_f32x4 = [
                f32x4::splat(pixel[0]),
                f32x4::splat(pixel[1]),
                f32x4::splat(pixel[2]),
            ];
            let out = linear_srgb_oklab_distance_simd::<4>(lum_factor, &pixel_f32x4, x4);
            let min_dist = out.reduce_min();
            if min_dist < current_min {
                current_min = min_dist;
                current_min_state = CurrentMinState::X4(out);
            }
        }

        for (i, color) in self.f32x1.iter().enumerate() {
            let min_dist = linear_srgb_oklab_distance_scalar(lum_factor, &pixel, color);
            if min_dist < current_min {
                current_min = min_dist;
                current_min_state = CurrentMinState::X1(i);
            }
        }

        match current_min_state {
            CurrentMinState::Unset => unreachable!(),
            CurrentMinState::X16(v, n) => {
                n * 16 + unsafe { find(v, current_min).unwrap_unchecked() }
            },
            CurrentMinState::X8(v) => {
                //
                self.ix8 + unsafe { find(v, current_min).unwrap_unchecked() }
            },
            CurrentMinState::X4(v) => {
                //
                self.ix4 + unsafe { find(v, current_min).unwrap_unchecked() }
            },
            CurrentMinState::X1(n) => self.ix1 + n,
        }
    }
}

#[inline]
fn linear_srgb_oklab_distance_scalar(lum_factor: f32, a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let r = a[0] - b[0];
    let g = a[1] - b[1];
    let b = a[2] - b[2];

    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    let l_ = (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s) * lum_factor;
    let a_ = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    let b_ = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;

    l_ * l_ + a_ * a_ + b_ * b_
}

#[inline]
fn linear_srgb_oklab_distance_simd<const N: usize>(
    lum_factor: f32,
    a: &[Simd<f32, N>; 3],
    b: &[Simd<f32, N>; 3],
) -> Simd<f32, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let r = a[0] - b[0];
    let g = a[1] - b[1];
    let b = a[2] - b[2];

    let l = Simd::<f32, N>::splat(0.4122214708) * r
        + Simd::<f32, N>::splat(0.5363325363) * g
        + Simd::<f32, N>::splat(0.0514459929) * b;
    let m = Simd::<f32, N>::splat(0.2119034982) * r
        + Simd::<f32, N>::splat(0.6806995451) * g
        + Simd::<f32, N>::splat(0.1073969566) * b;
    let s = Simd::<f32, N>::splat(0.0883024619) * r
        + Simd::<f32, N>::splat(0.2817188376) * g
        + Simd::<f32, N>::splat(0.6299787005) * b;

    let l_ = (Simd::<f32, N>::splat(0.2104542553) * l + Simd::<f32, N>::splat(0.7936177850) * m
        - Simd::<f32, N>::splat(0.0040720468) * s)
        * Simd::<f32, N>::splat(lum_factor);
    let a_ = Simd::<f32, N>::splat(1.9779984951) * l - Simd::<f32, N>::splat(2.4285922050) * m
        + Simd::<f32, N>::splat(0.4505937099) * s;
    let b_ = Simd::<f32, N>::splat(0.0259040371) * l + Simd::<f32, N>::splat(0.7827717662) * m
        - Simd::<f32, N>::splat(0.8086757660) * s;

    l_ * l_ + a_ * a_ + b_ * b_
}

/// Find the index of value in a SIMD vec.
#[inline]
fn find<T, const N: usize>(v: Simd<T, N>, value: T) -> Option<usize>
where
    LaneCount<N>: SupportedLaneCount,
    T: SimdElement,
    Simd<T, N>: SimdPartialEq<Mask = Mask<<T as SimdElement>::Mask, N>>,
    // <Simd<T, N> as SimdPartialEq>::Mask: MaskElement,
{
    let out = Simd::<T, N>::splat(value).simd_eq(v);
    out.first_set()
}
