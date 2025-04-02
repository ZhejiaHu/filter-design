use core::option::Option;
use core::{
    ops::{ Add, AddAssign, Mul, MulAssign, Div, DivAssign, Neg  },
    cmp::{ PartialEq, PartialOrd }
};
use std::{ error::Error, path::Path};
use std::alloc::System;
use std::fmt::Display;
use std::fs::File;
use fixed::types::{I16F16, I18F14, I64F64};
use csv::{Reader, Writer};
use csv::Trim::Headers;
use fixed::{FixedI32, FixedI64};
use fixed::types::extra::{U32, U8};
use serde::{Deserialize, Serialize};

type Fixed = FixedI64<U32>;
type HighResolution = I64F64;

pub trait OneZero {
    fn one() -> Self;
    fn zero() -> Self;
    fn abs(self) -> Self;

    fn to_f32(self) -> f32;

    fn from_usize(num_u: usize) -> Self;
    fn from_f32(num_f: f32) -> Self;

    fn to_high_resolution(self) -> HighResolution;

    fn from_high_resolution(num_f: HighResolution) -> Self;
}

impl OneZero for f32 {
    fn one() -> Self { 1.0 }
    fn zero() -> Self { 0.0 }
    fn abs(self) -> Self { if self < 0.0 { -self } else { self } }

    fn to_f32(self) -> f32 { self }

    fn from_usize(num_u: usize) -> Self { num_u as f32 }

    fn from_f32(num_f: f32) -> Self { num_f }

    fn to_high_resolution(self) -> HighResolution { HighResolution::from_num(self) }

    fn from_high_resolution(num_f: HighResolution) -> Self { num_f.to_num::<f32>()}
}

impl OneZero for Fixed {
    fn one() -> Self { Fixed::from_num(1.0) }

    fn zero() -> Self { Fixed::from_num(0.0) }
    fn abs(self) -> Self { if self < Fixed::from_num(0) { -self } else { self } }

    fn to_f32(self) -> f32 { self.to_num() }

    fn from_usize(num_u: usize) -> Self { Fixed::from_num(num_u) }
    fn from_f32(num_f: f32) -> Self { Fixed::from_num(num_f) }

    fn to_high_resolution(self) -> HighResolution { HighResolution::from_num(self) }

    fn from_high_resolution(num_f: HighResolution) -> Self { num_f.to_num::<Fixed>() }
}



fn fast_power(base: HighResolution, exp: usize, cache: &mut [HighResolution]) -> HighResolution {
    assert!(exp < cache.len());
    if exp == 0 {
        cache[exp] = HighResolution::from_num(1.0);
        return cache[exp];
    }
    if exp == 1 {
        cache[exp] = base;
        return base;
    }
    if cache[exp] == HighResolution::from_num(0.0) {
        cache[exp] = if exp & 1 == 1 {
            base * fast_power(base, exp - 1, cache)
        } else {
            let half = fast_power(base, exp >> 1, cache);
            half * half
        };
    }
    cache[exp]
}

pub fn arctan_approximation(num: HighResolution, degree_: Option<usize>, small_angle: bool) -> HighResolution {
    if small_angle || num == HighResolution::from_num(0.0) {
        return num;
    }
    if num.abs() > HighResolution::from_num(1.0) {
        return if num > HighResolution::from_num(1.0) {
            HighResolution::from_num(core::f32::consts::PI / 2.0) + - arctan_approximation(HighResolution::from_num(1.0) / num, degree_, small_angle)
        } else {
            -HighResolution::from_num(core::f32::consts::PI / 2.0) + - arctan_approximation(HighResolution::from_num(1.0) / num, degree_, small_angle)
        }
    }
    const MAX_DEGREE: usize = 10;
    let degree = degree_.unwrap_or(MAX_DEGREE);
    assert!(degree <= MAX_DEGREE);
    let mut eval: [HighResolution; MAX_DEGREE + 1] = [HighResolution::from_num(0); MAX_DEGREE + 1];
    for idx in (0..=degree).rev() {
        if idx % 2 == 1 {
            fast_power(num, idx, &mut eval);
        }
    }
    let mut arctan: HighResolution = HighResolution::from_num(0.0);
    for idx in (1..=degree).step_by(2) {
        let co_eff: HighResolution = if (idx / 2) % 2 == 1 { -HighResolution::from_num(1.0) } else { HighResolution::from_num(1.0) };
        arctan += co_eff * eval[idx] / HighResolution::from_num(idx);
    }
    arctan
}


pub fn arcsin_approximation(num: HighResolution, degree_: Option<usize>) -> HighResolution {
    const MAX_DEGREE: usize = 10;
    let degree = degree_.unwrap_or(MAX_DEGREE);
    assert!(degree <= MAX_DEGREE && degree % 2 == 1);
    let mut eval: [HighResolution; MAX_DEGREE + 1] = [HighResolution::from_num(0.0); MAX_DEGREE + 1];
    for idx in (0..=degree).rev() {
        if idx % 2 == 1 {
            fast_power(num, idx, &mut eval);
        }
    }
    const FACT_LEN: usize = (MAX_DEGREE >> 1) + 1;
    let mut factorials: [[usize; 2]; FACT_LEN] = [[1; 2]; FACT_LEN];
    let mut sol: HighResolution = num;
    for idx in 1..=FACT_LEN.min(degree >> 1) {
        let cur_power: usize = (idx << 1) + 1;
        factorials[idx][0] = factorials[idx - 1][0] * ((idx << 1) - 1);
        factorials[idx][1] = factorials[idx - 1][1] * (idx << 1);
        sol += (HighResolution::from_num(factorials[idx][0]) * eval[cur_power]) / (HighResolution::from_num(factorials[idx][1]) * HighResolution::from_num(cur_power))
    }
    sol
}




fn sqrt_approximation(num: HighResolution, err_: Option<HighResolution>) -> HighResolution {
    assert!(num > HighResolution::from_num(0));
    let err: HighResolution = err_.unwrap_or(HighResolution::from_num(1e-3));
    let mut sol: HighResolution = num / (HighResolution::from_num(2));
    let mut prv_sol: Option<HighResolution> = None;
    while (sol * sol + - num).abs() > err  {
        //println!("sol: {}, num: {}, sol * sol + - num: {}", sol, num, sol * sol + - num);
        sol = HighResolution::from_num(0.5)  * (sol + num / sol);
        if prv_sol.is_some() && (prv_sol.unwrap() + - sol).abs() < err {
            break;
        }
        prv_sol = Some(sol);
    }
    sol
}


macro_rules! pow2 {
    ($x: expr) => { (($x.to_high_resolution()) * ($x.to_high_resolution())) };
}

pub fn row_pitch<T>(a_x: T, a_y: T, a_z: T, degree: Option<usize>, small_angle_: Option<bool>) -> (T, T)
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    let small_angle = small_angle_.unwrap_or(true);
    let (denom_roll, denom_pitch): (HighResolution, HighResolution) = (pow2!(a_x) + pow2!(a_z), pow2!(a_y) + pow2!(a_z));
    println!("a_y: {}, a_y * a_y: {} || a_z: {}, a_z * a_z: {}", a_y, pow2!(a_y), a_z, pow2!(a_z));
    let (num_roll, num_pitch): (HighResolution, HighResolution) = (a_y.to_high_resolution() / a_z.to_high_resolution(), a_x.to_high_resolution() / sqrt_approximation(denom_pitch, None));
    //println!("a_x: {}, a_x * a_x: {} | a_y: {}, a_y * a_y: {} | a_z : {}, a_z * a_z: {}", a_x, pow2!(a_x), a_y, pow2!(a_y), a_z, pow2!(a_z));
    //println!("[num_roll] num_roll: {}, arctan(num_roll): {}", num_roll, arctan_approximation(num_roll, degree, small_angle));
    //println!("[num_pitch] num_pitch: {}, arctan(num_pitch): {}", num_pitch, arctan_approximation(num_pitch, degree, small_angle));
    (
        T::from_high_resolution(arctan_approximation(num_roll, degree, small_angle)),
        T::from_high_resolution(arctan_approximation(num_pitch, degree, small_angle))
    )
}



pub struct ComplementaryFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    a0: T,
    theta: T
}

impl<T> ComplementaryFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
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
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    c1: T,
    c2: T,
    p2phi: T,
    bias: T,
    phi: T
}


impl<T> KalmanFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
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

#[derive(Clone)]
pub(crate) struct OffsetsTyped<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    pub yaw: T,
    pub pitch: T,
    pub roll: T,
}

macro_rules! arctan_highest_approximation {
    ($num: expr) => { arctan_approximation(($num), Some(10), false) };
}


macro_rules! arctan2 {
    ($numer: expr, $denom: expr) => { arctan_highest_approximation!(($numer) / ($denom)) };
}

macro_rules! arcsin_highest_approximation {
    ($num: expr) => { arcsin_approximation(($num), Some(9)) };
}

impl<T> OffsetsTyped<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    pub(crate) fn new() -> Self {
        OffsetsTyped { yaw: T::zero(), pitch: T::zero(), roll: T::zero(), }
    }

    pub(crate) fn from_quaternion(&mut self, w_: T, x_: T, y_: T, z_: T) {
        let (w, x, y, z): (HighResolution, HighResolution, HighResolution, HighResolution) = (w_.to_high_resolution(), x_.to_high_resolution(), y_.to_high_resolution(), z_.to_high_resolution());
        let two = (T::one() + T::one()).to_high_resolution();
        let one = (T::one()).to_high_resolution();
        let gx = two * (x * z + (- w * y));
        let gy = two * (w * x + y * z);
        let gz = w * w + (- x * x) + (- y * y) + z * z;
        let yaw = arctan2!((two * x * y + (- two * w * z)), two * w * w + two * x * x + (- one));
        let pitch = arcsin_highest_approximation!(two * (w * y + (- z * x)));
        let roll = arctan2!(two * (w * x + y * z), one + (- two * (x * x + y * y)));
        self.yaw = T::from_high_resolution(yaw);
        self.pitch = T::from_high_resolution(pitch);
        self.roll = T::from_high_resolution(roll);
    }
}
#[derive(Copy, Clone)]
pub(crate) struct ButterworthFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    num_sample: u8,
    pub prv_x: T,
    pub prv_y: T,
    co_eff1: T,
    co_eff2: T
}



impl<T> ButterworthFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    // let co_eff1 = fix!(self.num_sample - 1) / fix!(self.num_sample);
    // let co_eff2 = fix!(1) / (fix!(self.num_sample) * 2);
    pub(crate) fn new(num_sample: u8) -> Self {
        let (co_eff1, co_eff2): (T, T) = (T::from_usize((num_sample - 1) as usize) / T::from_usize(num_sample as usize), T::one() / T::from_usize((num_sample as usize) << 1));
        ButterworthFilter { num_sample, prv_x: T::zero(), prv_y: T::zero(), co_eff1, co_eff2}
    }

    pub(crate) fn filter(&mut self, cur_x: T) -> T {
        let cur_y = self.co_eff1 * self.prv_y + self.co_eff2 * (cur_x + self.prv_x);
        self.prv_x = cur_x;
        self.prv_y = cur_y;
        cur_y
    }
}


struct FilterSystem<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    bwf_theta: ButterworthFilter<T>, kf_theta: KalmanFilter<T>,
    bwf_phi: ButterworthFilter<T>, kf_phi: KalmanFilter<T>,
    bwf_psi: ButterworthFilter<T>, kf_psi: KalmanFilter<T>,
    bwf_x_dd: ButterworthFilter<T>,
    bwf_y_dd: ButterworthFilter<T>,
    bwf_z_dd: ButterworthFilter<T>
}


#[inline]
fn normalize_raw_rpy<T>(raw_roll: T, raw_pitch: T, raw_yaw: T) -> (T, T, T)
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    const CONVERSION: f32 = 0.0007;
    const PI: f32 = core::f64::consts::PI as f32;
    let (conversion, pi): (T, T) = (T::from_f32(CONVERSION), T::from_f32(PI));
    (raw_roll * conversion, raw_pitch * conversion, raw_yaw * conversion)
}


type Debug<T> = (T, T, T, T, T);

impl<T> FilterSystem<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    const DEGREE: usize = 9;
    const SMALL_ANGLE: bool = false;
    pub fn new(bwf_config_: Option<[u8; 6]>, theta_kf_config_: Option<(T, T, T)>, phi_kf_config_: Option<(T, T, T)>, psi_kf_config_: Option<(T, T, T)>) -> Self {
        let bwf_config: [u8; 6] = bwf_config_.unwrap_or([4; 6]);
        let (theta_kf_config, phi_kf_config, psi_kf_config): ((T, T, T), (T, T, T), (T, T, T)) = (
            theta_kf_config_.unwrap_or((T::zero(), T::zero(), T::zero())), phi_kf_config_.unwrap_or((T::zero(), T::zero(), T::zero())), psi_kf_config_.unwrap_or((T::zero(), T::zero(), T::zero()))
        );
        FilterSystem {
            bwf_theta: ButterworthFilter::new(bwf_config[0]), kf_theta: KalmanFilter::new(theta_kf_config.0, theta_kf_config.1, theta_kf_config.2),
            bwf_phi: ButterworthFilter::new(bwf_config[1]), kf_phi: KalmanFilter::new(phi_kf_config.0, phi_kf_config.1, phi_kf_config.2),
            bwf_psi: ButterworthFilter::new(bwf_config[2]), kf_psi: KalmanFilter::new(psi_kf_config.0, psi_kf_config.1, psi_kf_config.2),
            bwf_x_dd: ButterworthFilter::new(bwf_config[3]),
            bwf_y_dd: ButterworthFilter::new(bwf_config[4]),
            bwf_z_dd: ButterworthFilter::new(bwf_config[5])
        }
    }

    pub fn filter(&mut self, theta_: T, phi_: T, psi_: T, x_dd: T, y_dd: T, z_dd: T) -> (T, T, T, Option<Debug<T>>) {
        let (theta, phi, psi): (T, T, T) = normalize_raw_rpy(theta_, phi_, psi_);
        let (theta_tilde, phi_tilde , psi_tilde, x_dd_tilde, y_dd_tilde, z_dd_tilde): (T, T, T, T, T, T) = (
            self.bwf_theta.filter(theta), self.bwf_phi.filter(phi), self.bwf_psi.filter(psi),
            self.bwf_x_dd.filter(x_dd), self.bwf_y_dd.filter(y_dd), self.bwf_z_dd.filter(z_dd)
        );
        // println!("[FilterSystem::filter] theta_input: {}, theta_tilde: {} | phi_input: {}, phi_tilde: {} | psi_input: {}, psi_tilde: {}", theta, theta_tilde, phi, phi_tilde, psi, psi_tilde);
        // println!("[FilterSystem::filter] x_dd_input: {}, x_dd_tile: {} | y_dd_input: {}, y_dd_tilde: {} | z_dd_input: {}, z_dd_tilde: {}", x_dd, x_dd_tilde, y_dd, y_dd_tilde, z_dd, z_dd_tilde);
        let (theta_hat, phi_hat): (T, T) = row_pitch(x_dd_tilde, y_dd_tilde, z_dd_tilde, Some(Self::DEGREE), Some(Self::SMALL_ANGLE));
        println!("[FilterSystem::filter] theta_raw: {}, theta_input: {}, theta_tilde: {}, theta_hat: {}", theta_, theta, theta_tilde, theta_hat);
        println!("[FilterSystem::filter] phi_raw: {}, phi_input: {}, phi_tilde: {}, phi_hat: {}", phi_, phi, phi_tilde, phi_hat);
        println!("[FilterSystem::filter] psi: {}, psi_norm: {}, psi_tilde: {}", psi_, psi, psi_tilde);
        // todo: check wire connection of Kalman Filter and output
        (
            self.kf_theta.predict(theta_tilde, theta_hat).1,
            self.kf_phi.predict(phi_tilde, phi_hat).1,
            psi_tilde,
            Some((theta_tilde, phi_tilde, psi_tilde, theta_hat, phi_hat))
        )
    }

}


#[derive(Debug, Serialize, Deserialize)]
struct Data<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display
{
    #[serde(rename = "Raw Roll")]
    raw_roll: T,
    #[serde(rename = "Raw Pitch")]
    raw_pitch: T,
    #[serde(rename = "Raw Yar")]
    raw_yar: T,
    #[serde(rename = "Raw X")]
    raw_x: T,
    #[serde(rename = "Raw Y")]
    raw_y: T,
    #[serde(rename = "Raw Z")]
    raw_z: T,
    #[serde(rename = "DMP Roll")]
    dmp_roll: T,
    #[serde(rename = "DMP Pitch")]
    dmp_pitch: T,
    #[serde(rename = "DMP Yaw")]
    dmp_yaw: T
}


impl<T> Data<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    fn from(f_data: Data<f32>) -> Self {
        Data {
            raw_roll: T::from_f32(f_data.raw_roll), raw_pitch: T::from_f32(f_data.raw_pitch), raw_yar: T::from_f32(f_data.raw_yar),
            raw_x: T::from_f32(f_data.raw_x), raw_y: T::from_f32(f_data.raw_y), raw_z: T::from_f32(f_data.raw_z),
            dmp_roll: T::from_f32(f_data.dmp_roll), dmp_pitch: T::from_f32(f_data.dmp_pitch), dmp_yaw: T::from_f32(f_data.dmp_yaw)
        }
    }
}



fn process_data<T>(dataset: &Path) -> Result<Vec<Data<T>>, Box<dyn Error>>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    let mut reader = csv::Reader::from_path(dataset)?;
    let mut data_vec: Vec<Data<T>> = Vec::new();
    for result in reader.deserialize() {
        let data_raw: Data<f32> = result?;
        data_vec.push(Data::from(data_raw));
    }
    Ok(data_vec)
}


macro_rules! stringify {
    ($num: ident) => { $num.to_f32().to_string() };
}


fn testbench_data<T>(dataset: &Path, output_path: &Path, row_kf_config: Option<(T, T, T)>, pitch_kf_config: Option<(T, T, T)>, yaw_kf_config: Option<(T, T, T)>)
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero + Display {
    let dataset: Vec<Data<T>> = process_data(dataset).unwrap();
    let mut writer = Writer::from_path(output_path).unwrap();
    writer.write_record(&["Filtered Roll", "Filtered Pitch", "Filtered Yaw", "Butterworth Roll", "Butterworth Pitch", "Butterworth Yaw", "Estimated Roll", "Estimated Pitch"]).expect("No error writing header");
    let mut filter_system: FilterSystem<T> = FilterSystem::new(Some([10u8; 6]), row_kf_config, pitch_kf_config, yaw_kf_config);
    for data in dataset {
        println!("================================================================================================================");
        let (filtered_roll, filtered_pitch, filtered_yaw, debug_): (T, T, T, Option<Debug<T>>) = filter_system.filter(
            data.raw_roll, data.raw_pitch, data.raw_yar, data.raw_x, data.raw_y, data.raw_z
        );
        let (norm_roll, norm_pitch, norm_yaw): (T, T, T) = normalize_raw_rpy(data.raw_roll, data.raw_pitch, data.raw_yar);
        println!("[testbench_data] raw_roll: {}, norm_roll: {}, filtered_roll: {} || raw_pitch: {}, norm_pitch: {}, filtered_pitch: {} || raw_yaw: {}, norm_yaw: {}, filtered_yaw: {}", data.raw_roll, norm_roll, filtered_roll, data.raw_pitch, norm_pitch, filtered_pitch, data.raw_yar, norm_yaw, filtered_yaw);
        let (bw_roll, bw_pitch, bw_yaw, estimate_roll, estimate_pitch) = debug_.unwrap_or((T::from_f32(0.0), T::from_f32(0.0), T::from_f32(0.0), T::from_f32(0.0), T::from_f32(0.0)));
        writer.write_record(&[
            stringify!(filtered_roll), stringify!(filtered_pitch), stringify!(filtered_yaw),
            stringify!(bw_roll), stringify!(bw_pitch),stringify!(bw_yaw),
            stringify!(estimate_roll), stringify!(estimate_pitch)
        ]).expect("No error writing line");
    }
}


type FixedKfConfig = (Fixed, Fixed, Fixed);


fn main() {
    let (dataset_pth, output_pth): (&Path, &Path) = (Path::new("data/roll.csv"), Path::new("output/roll.csv"));
    let kf_config_: FixedKfConfig = (Fixed::from_num(50.0), Fixed::from_num(1000.0), Fixed::from_num(0.005));
    let row_kf_config: Option<FixedKfConfig> = Some(kf_config_.clone());
    let pitch_kf_config: Option<FixedKfConfig> = Some(kf_config_.clone());
    let yaw_kf_config: Option<FixedKfConfig> = Some(kf_config_);
    testbench_data(dataset_pth, output_pth, row_kf_config, pitch_kf_config, yaw_kf_config);
}


#[cfg(test)]
mod test {
    use rand::Rng;
    use crate::{arcsin_approximation, arctan_approximation, sqrt_approximation, HighResolution};
    const ITER: usize = 100;
    const ERR: f32 = 2e-1;

    macro_rules! assert_close {
        ($ans: expr, $sol: expr) => { assert!(($ans - $sol).abs() < ERR) };
    }

#[test]
    fn test_arcsin() {
        let mut rng = rand::rng();
        let num: f32 = rng.random_range(-1.0 + ERR..1.0);
        for idx in 0..ITER {
            let num: f32 = rng.random_range(-1.0 + ERR..1.0);
            let arcsin = arcsin_highest_approximation!(HighResolution::from_num(num)).to_num::<f32>();
            let arctan = arctan_highest_approximation!(HighResolution::from_num(num)).to_num::<f32>();
            let sqrt = sqrt_approximation(HighResolution::from_num(num.abs()), None).to_num::<f32>();
            println!("idx: {}, num: {}, arcsin: {} (original: {}), arctan: {} (original: {}), sqrt: {} (original: {})", idx, num, arcsin, num.asin(), arctan, num.atan(), sqrt, num.abs().sqrt());
            assert_close!(num.asin(), arcsin);
            assert_close!(num.atan(), arctan);
            assert_close!(num.abs().sqrt(), sqrt);
        }
    }
}