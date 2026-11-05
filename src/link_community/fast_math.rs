//! Fast, approximate `ln` and `ln Γ` in `f32`: Paul Mineiro's fastapprox.
//!
//! Scoring is noise-dominated, so these trade digits for speed. The one thing
//! that must not be lost is the increment `ln Γ(x + y) − ln Γ(x)`, which is what
//! a collapsed conditional reads, and [`ln_gamma_increment`] keeps it.

/// `ln x` for a positive normal `x`: the float's bits read as its `log₂`, then
/// corrected. About `1e-4`.
#[inline(always)]
pub(crate) fn fast_ln(x: f32) -> f32 {
    let i = x.to_bits();
    let m = f32::from_bits((i & 0x007F_FFFF) | 0x3f00_0000);
    let y = i as i32 as f32 * 1.192_092_9e-7;
    std::f32::consts::LN_2 * (y - 124.225_52 - 1.498_030_3 * m - 1.725_88 / (0.352_088_7 + m))
}

/// `ln Γ(x)` for `x > 0`:
/// `−2.0810615 − x + 1/(12(x+3)) − ln(x (x+1) (x+2)) + (x + 2.5) ln(x+3)`.
#[inline(always)]
pub(crate) fn fast_ln_gamma(x: f32) -> f32 {
    let xp3 = 3.0 + x;
    -2.081_061_5 - x + 0.083_333_3 / xp3 - fast_ln(x * (1.0 + x) * (2.0 + x))
        + (2.5 + x) * fast_ln(xp3)
}

/// From here up, the slope of `ln Γ` is close enough to `ln(x − ½)` for the
/// midpoint rule. Below, `ln Γ` is small, so its difference cancels little.
const MIDPOINT_FROM: f32 = 8.0;

/// `ln Γ(x + y) − ln Γ(x)` for `x, y > 0`, off by about a percent of `y`.
///
/// When `y` is small against `x`, subtracting two `ln Γ` cancels nearly all of
/// their `f32` digits, and past `x ≈ 2e7` `x + y` no longer even differs from
/// `x`. There `∫ ln(t − ½) dt` over `[x, x + y]` needs no subtraction: with
/// `mid = x + (y − 1)/2` and `r = y/mid`, it is `y · (ln(mid) − r²/24 − r⁴/320)`
/// to the order that keeps the two forms close where they meet, at `x = y`.
/// When `y` is not below `x`, the difference cancels nothing and is used as is.
#[inline(always)]
pub(crate) fn ln_gamma_increment(x: f32, y: f32) -> f32 {
    if midpoint_holds(x, y) {
        midpoint_increment(x, y)
    } else {
        fast_ln_gamma(x + y) - fast_ln_gamma(x)
    }
}

/// [`ln_gamma_increment`] at each of `xs`, into `out`, bit for bit: the midpoint
/// form across the run, which vectorizes, then the other form where it does not
/// hold. One fold across `xs`, so that a run needing none pays no branch per cell.
#[inline(always)]
pub(crate) fn ln_gamma_increments(xs: &[f32], y: f32, out: &mut [f32]) {
    for (inc, &x) in out.iter_mut().zip(xs) {
        *inc = midpoint_increment(x, y);
    }
    if xs.iter().fold(false, |any, &x| any | !midpoint_holds(x, y)) {
        for (inc, &x) in out.iter_mut().zip(xs) {
            if !midpoint_holds(x, y) {
                *inc = fast_ln_gamma(x + y) - fast_ln_gamma(x);
            }
        }
    }
}

/// Whether [`ln_gamma_increment`] takes the midpoint form at these arguments.
#[inline(always)]
fn midpoint_holds(x: f32, y: f32) -> bool {
    y < x && x >= MIDPOINT_FROM
}

/// The midpoint form alone: branch-free, so a loop over `x` can vectorize.
#[inline(always)]
fn midpoint_increment(x: f32, y: f32) -> f32 {
    let mid = x + 0.5 * (y - 1.0);
    let ratio = y / mid;
    let squared = ratio * ratio;
    y * (fast_ln(mid) - squared / 24.0 - squared * squared / 320.0)
}
