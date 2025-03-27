use core::option::Option;
use core::{
    ops::{ Add, AddAssign, Mul, MulAssign, Div, DivAssign, Neg  },
    cmp::{ PartialEq, PartialOrd }
};
use std::{ error::Error, path::Path};
use std::alloc::System;
use std::fs::File;
use fixed::types::{I16F16, I18F14};
use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};

type Fixed = I16F16;

pub trait OneZero {
    fn one() -> Self;
    fn zero() -> Self;
    fn abs(self) -> Self;

    fn to_f32(self) -> f32;

    fn from_usize(num_u: usize) -> Self;
    fn from_f32(num_f: f32) -> Self;
}

impl OneZero for f32 {
    fn one() -> Self { 1.0 }
    fn zero() -> Self { 0.0 }
    fn abs(self) -> Self { self.abs() }

    fn to_f32(self) -> f32 { self }

    fn from_usize(num_u: usize) -> Self { num_u as f32 }

    fn from_f32(num_f: f32) -> Self { num_f }
}

impl OneZero for Fixed {
    fn one() -> Self { Fixed::from_num(1.0) }

    fn zero() -> Self { Fixed::from_num(0.0) }
    fn abs(self) -> Self { self.abs() }

    fn to_f32(self) -> f32 { self.to_num() }

    fn from_usize(num_u: usize) -> Self { Fixed::from_num(num_u) }
    fn from_f32(num_f: f32) -> Self { Fixed::from_num(num_f) }

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
    let degree = degree_.unwrap_or(MAX_DEGREE);
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


pub fn arcsin_approximation<T>(num: T, degree_: Option<usize>) -> T
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    const MAX_DEGREE: usize = 20;
    let degree = degree_.unwrap_or(MAX_DEGREE);
    assert!(degree <= MAX_DEGREE && degree % 2 == 1);
    let mut eval: [T; MAX_DEGREE + 1] = [T::zero(); MAX_DEGREE + 1];
    for idx in (0..=degree).rev() {
        if idx % 2 == 1 {
            fast_power(num, idx, &mut eval);
        }
    }
    const FACT_LEN: usize = (MAX_DEGREE >> 1) + 1;
    let mut factorials: [[usize; 2]; FACT_LEN] = [[1; 2]; FACT_LEN];
    let mut sol: T = num;
    for idx in 1..FACT_LEN.min(degree >> 1) {
        let cur_power: usize = (idx << 1) + 1;
        factorials[idx][0] = factorials[idx - 1][0] * ((idx << 1) - 1);
        factorials[idx][1] = factorials[idx - 1][1] * (idx << 1);
        sol += (T::from_usize(factorials[idx][0]) * eval[cur_power]) / (T::from_usize(factorials[idx][1]) * T::from_usize(cur_power))
    }
    sol
}




fn sqrt_approximation<T>(num: T, err_: Option<T>) -> T
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    let err: T = err_.unwrap_or(T::from_f32(1e-3));
    let mut sol: T = num / (T::one() + T::one());
    let mut prv_sol: Option<T> = None;
    while (sol * sol + - num).abs() > err  {
        if prv_sol.is_some() && prv_sol.unwrap() == sol {
            break;
        }
        sol = T::from_f32(0.5)  * (sol + num / sol);
        prv_sol = Some(sol);
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
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
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
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    pub yaw: T,
    pub pitch: T,
    pub roll: T,
}

macro_rules! arctan_highest_approximation {
    ($num: expr) => { arctan_approximation(($num), Some(20), false) };
}


macro_rules! arctan2 {
    ($numer: expr, $denom: expr) => { arctan_highest_approximation!(($numer) / ($denom)) };
}

macro_rules! arcsin_highest_approximation {
    ($num: expr) => { arcsin_approximation(($num), Some(20)) };
}

impl<T> OffsetsTyped<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    pub(crate) fn new() -> Self {
        OffsetsTyped { yaw: T::zero(), pitch: T::zero(), roll: T::zero(), }
    }

    pub(crate) fn from_quaternion(&mut self, w: T, x: T, y: T, z: T) {
        let two = T::one() + T::one();
        let one = T::one();
        let gx = two * (x * z + (- w * y));
        let gy = two * (w * x + y * z);
        let gz = w * w + (- x * x) + (- y * y) + z * z;
        let yaw = arctan2!((two * x * y + (- two * w * z)), two * w * w + two * x * x + (- one));
        let pitch = arcsin_highest_approximation!(two * (w * y + (- z * x)));
        let roll = arctan2!(two * (w * x + y * z), one + (- two * (x * x + y * y)));
        self.yaw = yaw;
        self.pitch = pitch;
        self.roll = roll;
    }
}
#[derive(Copy, Clone)]
pub(crate) struct ButterworthFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    num_sample: u8,
    pub prv_x: T,
    pub prv_y: T,
    co_eff1: T,
    co_eff2: T
}



impl<T> ButterworthFilter<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
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
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    bwf_theta: ButterworthFilter<T>, kf_theta: KalmanFilter<T>,
    bwf_phi: ButterworthFilter<T>, kf_phi: KalmanFilter<T>,
    bwf_psi: ButterworthFilter<T>, kf_psi: KalmanFilter<T>,
    bwf_x_dd: ButterworthFilter<T>,
    bwf_y_dd: ButterworthFilter<T>,
    bwf_z_dd: ButterworthFilter<T>
}



impl<T> FilterSystem<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    const DEGREE: usize = 20;
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

    pub fn filter(&mut self, theta: T, phi: T, psi: T, x_dd: T, y_dd: T, z_dd: T) -> (T, T, T) {
        let (theta_tilde, phi_tilde , psi_tilde, x_dd_tilde, y_dd_tilde, z_dd_tilde): (T, T, T, T, T, T) = (
            self.bwf_theta.filter(theta), self.bwf_phi.filter(phi), self.bwf_psi.filter(phi),
            self.bwf_x_dd.filter(x_dd), self.bwf_y_dd.filter(y_dd), self.bwf_z_dd.filter(z_dd)
        );
        let (theta_hat, phi_hat): (T, T) = row_pitch(x_dd_tilde, y_dd_tilde, z_dd_tilde, Some(Self::DEGREE), Some(Self::SMALL_ANGLE));
        // todo: check wire connection of Kalman Filter and output
        (
            self.kf_theta.predict(theta_tilde, theta_hat).1,
            self.kf_phi.predict(phi_tilde, phi_hat).1,
            psi_tilde
        )
    }

}


#[derive(Debug, Serialize, Deserialize)]
struct Data<T>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero
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
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    fn from(f_data: Data<f32>) -> Self {
        Data {
            raw_roll: T::from_f32(f_data.raw_roll), raw_pitch: T::from_f32(f_data.raw_pitch), raw_yar: T::from_f32(f_data.raw_yar),
            raw_x: T::from_f32(f_data.raw_x), raw_y: T::from_f32(f_data.raw_y), raw_z: T::from_f32(f_data.raw_z),
            dmp_roll: T::from_f32(f_data.dmp_roll), dmp_pitch: T::from_f32(f_data.dmp_pitch), dmp_yaw: T::from_f32(f_data.dmp_yaw) }
    }
}



fn process_data<T>(dataset: &Path) -> Result<Vec<Data<T>>, Box<dyn Error>>
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    let mut reader = csv::Reader::from_path(dataset)?;
    let mut data_vec: Vec<Data<T>> = Vec::new();
    for result in reader.deserialize() {
        let data_raw: Data<f32> = result?;
        data_vec.push(Data::from(data_raw));
    }
    Ok(data_vec)
}



fn testbench_data<T>(dataset: &Path, output_path: &Path, row_kf_config: Option<(T, T, T)>, pitch_kf_config: Option<(T, T, T)>, yaw_kf_config: Option<(T, T, T)>)
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    let dataset: Vec<Data<T>> = process_data(dataset).unwrap();
    let mut writer = Writer::from_path(output_path).unwrap();
    writer.write_record(&["Filtered Roll", "Filtered Pitch", "Filtered Yaw"]).expect("No error writing header");
    let mut filter_system: FilterSystem<T> = FilterSystem::new(Some([5u8; 6]), row_kf_config, pitch_kf_config, yaw_kf_config);
    for data in dataset {
        let (filtered_roll, filtered_pitch, filtered_yaw): (T, T, T) = filter_system.filter(
            data.raw_roll, data.dmp_pitch, data.dmp_yaw, data.raw_x, data.raw_y, data.raw_z
        );
        writer.write_record(&[
            filtered_roll.to_f32().to_string(),
            filtered_pitch.to_f32().to_string(),
            filtered_yaw.to_f32().to_string()]
        ).expect("No error writing line");
    }
}



fn main() {
    let (dataset_pth, output_pth): (&Path, &Path) = (Path::new("data/roll.txt"), Path::new("output/roll.csv"));
    let row_kf_config: Option<(f32, f32, f32)> = Some((1.0, 1.0, 0.0));
    let pitch_kf_config: Option<(f32, f32, f32)> = Some((1.0, 1.0, 0.0));
    let yaw_kf_config: Option<(f32, f32, f32)> = Some((1.0, 1.0, 0.0));
    testbench_data(dataset_pth, output_pth, row_kf_config, pitch_kf_config, yaw_kf_config);
}