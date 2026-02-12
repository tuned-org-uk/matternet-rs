use burn::tensor::{Tensor, backend::Backend};

pub struct DataMatrix<B: Backend> {
    pub tensor: Tensor<B, 2>, // [N, F]
}

impl<B: Backend> DataMatrix<B> {
    pub fn new(data: Vec<f32>, n: usize, f: usize, device: &B::Device) -> Self {
        let tensor = Tensor::<B, 2>::from_data(data.as_slice(), device).reshape([n, f]);
        Self { tensor }
    }
}
