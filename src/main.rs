use core::{
    ops::{ Add, AddAssign, Mul, MulAssign, Div, DivAssign, Neg  },
    cmp::{ PartialEq, PartialOrd }
};
use fixed::types::I18F14;

pub trait OneZero {
    fn one() -> Self;
    fn zero() -> Self;
    fn abs(self) -> Self;
    fn from_usize(num_u: usize) -> Self;
    fn from_f32(num_f: f32) -> Self;
}

impl OneZero for f32 {
    fn one() -> Self { 1.0 }
    fn zero() -> Self { 0.0 }
    fn abs(self) -> Self { self.abs() }
    fn from_usize(num_u: usize) -> Self { num_u as f32 }

    fn from_f32(num_f: f32) -> Self { num_f }
}

impl OneZero for I18F14 {
    fn one() -> Self { I18F14::from_num(1.0) }

    fn zero() -> Self { I18F14::from_num(0.0) }
    fn abs(self) -> Self { self.abs() }

    fn from_usize(num_u: usize) -> Self { I18F14::from_num(num_u) }
    fn from_f32(num_f: f32) -> Self { I18F14::from_num(num_f) }

}


fn fast_power<T>(base: T, exp: usize, cache: &mut [T]) -> T
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero{
    assert!(exp < cache.len());
    if exp == 0 {
        cache[exp] = T::one();
        return T::one();
    }
    if exp == 1 {
        cache[exp] = base;
        return base;
    }
    if cache[exp] == T::zero() {
        cache[exp] = if exp & 1 == 1 {
            base * fast_power(base, exp - 1, cache)
        } else {
            let half = fast_power(base, exp >> 1, cache);
            half * half
        };
    }
    cache[exp]
}


pub fn arctan_approximation<T>(num: T, degree_: Option<usize>, small_angle: bool) -> T
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    if small_angle || num == T::zero() {
        return num;
    }
    const MAX_DEGREE: usize = 20;
    let degree = degree_.unwrap_or(2);
    assert!(degree <= MAX_DEGREE);
    let mut eval: [T; MAX_DEGREE + 1] = [T::zero(); MAX_DEGREE + 1];
    for idx in (0..=degree).rev() {
        if idx % 2 == 1 {
            fast_power(num, idx, &mut eval);
        }
    }
    let mut arctan: T = T::zero();
    for idx in (1..=degree).step_by(2) {
        let co_eff: T = if (idx / 2) % 2 == 1 { -T::one() } else { T::one() };
        arctan += co_eff * eval[idx] / T::from_usize(idx);
    }
    arctan
}




fn sqrt_approximation<T>(num: T, err_: Option<T>) -> T
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    let err: T = err_.unwrap_or(T::from_f32(1e-6));
    let mut sol: T = num / (T::one() + T::one());
    while (sol * sol + - num).abs() > err {
        sol = T::from_f32(0.5)  * (sol + num / sol);
    }
    sol
}


macro_rules! pow2 {
    ($x: expr) => { (($x) * ($x)) };
}

pub fn row_pitch<T>(a_x: T, a_y: T, a_z: T, degree: Option<usize>, small_angle_: Option<bool>) -> (T, T)
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    let small_angle = small_angle_.unwrap_or(true);
    let (denom_roll, denom_pitch): (T, T) = (pow2!(a_x) + pow2!(a_z), pow2!(a_y) + pow2!(a_z));
    let (num_roll, num_pitch): (T, T) = (a_y / sqrt_approximation(denom_roll, None), a_x / sqrt_approximation(denom_pitch, None));
    (arctan_approximation(num_roll, degree, small_angle), arctan_approximation(num_pitch, degree, small_angle))
}



pub struct ComplementaryFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    a0: T,
    theta: T
}

impl<T> ComplementaryFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    pub(crate) fn new(a0: T) -> Self {
        ComplementaryFilter { a0, theta: T::zero() }
    }


    pub(crate) fn predict(&mut self, sax: T, sq: T, timestep: Option<T>) -> T {
        assert!(timestep.is_some());
        self.theta = self.a0 * sax + (T::one() + (- self.a0)) * (self.theta + sq * timestep.unwrap());
        self.theta
    }
}



pub struct KalmanFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    c1: T,
    c2: T,
    p2phi: T,
    bias: T,
    phi: T
}



impl<T> KalmanFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero{
    pub(crate) fn new(c1: T, c2: T, p2phi: T) -> Self {
        KalmanFilter { c1, c2, p2phi, bias: T::zero(), phi: T::zero() }
    }

    pub(crate) fn predict(&mut self, sp: T, sphi: T) -> (T, T, T) {
        let p: T = sp + (- self.bias);
        self.phi += p * self.p2phi;
        let err: T = self.phi + (- sphi);
        self.phi += - (err / self.c1);
        self.bias += (err / self.p2phi) / self.c2;
        (p, self.phi, err)
    }
}

fn main() {}