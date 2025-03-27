use std::option::Option;
use core::{
    ops::{ Add, AddAssign, Mul, MulAssign, Div, DivAssign, Neg  },
    cmp::{ PartialEq, PartialOrd }
};
use std::{ error::Error, path::Path};
use std::fs::File;
use fixed::types::I18F14;
use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};

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

impl OneZero for I18F14 {
    fn one() -> Self { I18F14::from_num(1.0) }

    fn zero() -> Self { I18F14::from_num(0.0) }
    fn abs(self) -> Self { self.abs() }

    fn to_f32(self) -> f32 { self.to_num() }

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



fn system_without_butterworth<T>(
    theta_slash: T, phi_slash: T, psi_slash: T, x_dd_slash: T, y_dd_slash: T, z_dd_slash: T, degree: Option<usize>, small_angle: Option<bool>,
    theta_kf_config_: Option<(T, T, T)>, phi_kf_config_: Option<(T, T, T)>, psi_kf_config_: Option<(T, T, T)>) -> (T, T, T)
where T: Add<Output = T> + AddAssign + Mul<Output = T> + MulAssign + Div<Output = T> + DivAssign + Neg<Output = T> + Copy + PartialEq + PartialOrd + OneZero {
    let (theta_kf_config, phi_kf_config, psi_kf_config): ((T, T, T), (T, T, T), (T, T, T)) = (
        theta_kf_config_.unwrap_or((T::zero(), T::zero(), T::zero())), phi_kf_config_.unwrap_or((T::zero(), T::zero(), T::zero())), psi_kf_config_.unwrap_or((T::zero(), T::zero(), T::zero()))
    );
    let (theta_tilde, phi_tilde): (T, T) = row_pitch(x_dd_slash, y_dd_slash, z_dd_slash, degree, small_angle);
    let (mut kalman_theta, mut kalman_phi, mut kalman_psi): (KalmanFilter<T>, KalmanFilter<T>, KalmanFilter<T>) = (
        KalmanFilter::new(theta_kf_config.0, theta_kf_config.1, theta_kf_config.2),
        KalmanFilter::new(phi_kf_config.0, phi_kf_config.1, phi_kf_config.2),
        KalmanFilter::new(psi_kf_config.0, psi_kf_config.1, psi_kf_config.2)
    );
    (
        kalman_theta.predict(theta_slash, theta_tilde).0,
        kalman_phi.predict(phi_slash, phi_tilde).0,
        psi_slash
    )
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
    for data in dataset {
        let (filtered_roll, filtered_pitch, filtered_raw): (T, T, T) = system_without_butterworth(
            data.raw_roll, data.dmp_pitch, data.dmp_yaw, data.raw_x, data.raw_y, data.raw_z,
            Some(10), Some(false), row_kf_config, pitch_kf_config, yaw_kf_config
        );
        writer.write_record(&[
            filtered_roll.to_f32().to_string(),
            filtered_pitch.to_f32().to_string(),
            filtered_roll.to_f32().to_string()]
        ).unwrap();
    }
}



fn main() {
    let (dataset_pth, output_pth): (&Path, &Path) = (Path::new("data/pitch.txt"), Path::new("output/pitch.csv"));
    let row_kf_config: Option<(f32, f32, f32)> = Some((1.0, 1.0, 0.0));
    let pitch_kf_config: Option<(f32, f32, f32)> = Some((1.0, 1.0, 0.0));
    let yaw_kf_config: Option<(f32, f32, f32)> = Some((1.0, 1.0, 0.0));
    testbench_data(dataset_pth, output_pth, row_kf_config, pitch_kf_config, yaw_kf_config);
}