use oklab::{Oklab, RGB};
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rand_xoshiro::Xoshiro512PlusPlus;

#[inline(always)]
fn to_linear8(c: u8) -> f32 {
    let u = c as f32 / 255.0;
    if u >= 0.04045 {
        ((u + 0.055) / (1. + 0.055)).powf(2.4)
    } else {
        u / 12.92
    }
}

#[inline(always)]
fn to_linear8_(c: u8) -> f32 {
    let u = c as f32 / 255.0;
    ((u + 0.055) / (1. + 0.055)).powf(2.4)
}

#[test]
fn demo() {
    for u in 0..u8::MAX {
        let expected = to_linear8(u);
        let actual = to_linear8_(u);
        let diff = expected - actual;
        println!("{u}\t{expected}\t{actual}\t{diff}");
    }
}

pub fn linear_srgb_oklab_distance(a: RGB<f32>, b: RGB<f32>) -> f32 {
    let c = RGB {
        r: a.r as f64 - b.r as f64,
        g: a.g as f64 - b.g as f64,
        b: a.b as f64 - b.b as f64,
    };

    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;

    let l = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s;
    let a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
    let b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;

    (l * l + a * a + b * b) as f32
}

#[test]
fn rand_dist_demo() {
    let mut rng = Xoshiro512PlusPlus::seed_from_u64(0);
    let normal = Normal::<f64>::new(0.0, 20.0).unwrap();

    let mut min;
    let mut max;

    let z: f64 = rng.sample(rand_distr::StandardNormal);
    let n = normal.from_zscore(z);
    min = n;
    max = n;

    // for _ in 0..1_000_000_000u64 {
    for _ in 0..512u64 {
        let z: f64 = rng.sample(rand_distr::StandardNormal);
        let n = normal.from_zscore(z);

        if n < min {
            min = n;
        }

        if n > max {
            max = n;
        }
    }

    // 10B iterations -> -124.71844320913917 - 124.61000409928532
    // ~ (-2PI * stddev, +2PI * stddev)
    println!("{min} - {max}")
}
