use tch::{Device, Tensor};

fn main() {
    let device = Device::cuda_if_available();
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]).to(device);
    let t = t * 2;
    t.print();
}
