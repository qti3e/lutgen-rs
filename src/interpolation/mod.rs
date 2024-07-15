//! Interpolated remapping algorithms.

pub use gaussian_sample::GaussianSamplingRemapper;
use image::Rgb;
use kiddo::float::kdtree::KdTree;
pub use nearest_neighbor::NearestNeighborRemapper;
pub use nearest_neighbor2::NearestNeighborRemapper2;
use rayon::prelude::*;
pub use rbf::{GaussianRemapper, LinearRemapper, ShepardRemapper};

use crate::Image;

mod gaussian_sample;
mod nearest_neighbor;
mod nearest_neighbor2;
mod rbf;

/// Interpolated Remapper. Implements an algorithm with some initialization parameters.
pub trait InterpolatedRemapper<'a>: Sync {
    /// Remap a single pixel in place
    fn remap_pixel(&self, pixel: &mut Rgb<u8>);

    /// Remap an image in place. Default implementation uses `rayon` to iterate in parallel over
    /// the pixels.
    fn remap_image(&self, image: &mut Image) {
        image.par_pixels_mut().for_each(|pixel| {
            self.remap_pixel(pixel);
        });
    }
}

/// Type alias for our internal color tree for NN lookups
type ColorTree = KdTree<f64, u32, 3, 4, u32>;
