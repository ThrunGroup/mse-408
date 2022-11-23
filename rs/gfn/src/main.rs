use std::ops::Index;
use tch::Tensor;

pub trait Flows<F, Idx>: Index<Idx, Output = F> + Iterator<Item = F> {}

impl Flows<Tensor, Tensor> for Tensor {
    // TODO
}

fn main() {
    let t: Box<dyn Flows<Tensor, Tensor>> =
        Box::new(Tensor::of_slice(&[3, 1, 4, 1, 5]));
    println!("{}", t);
}
