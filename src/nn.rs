use std::iter::zip;

use rand::{distributions::Uniform, prelude::Distribution};

use crate::engine::Value;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let between = Uniform::new_inclusive(-1.0, 1.0);
        Self {
            weights: between
                .sample_iter(&mut rng)
                .take(nin)
                .map(|x| Value::new(x, vec![], None))
                .collect(),
            bias: Value::new(between.sample(&mut rng), vec![], None),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Value {
        let act = zip(self.weights.iter(), x.iter())
            .map(|(wi, xi)| wi * xi)
            .fold(self.bias.clone(), |a, b| a + b);

        act.tanh()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.weights.clone()
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        Self {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x.clone())).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|n| n.parameters())
            .flatten()
            .collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut szs = vec![nin];
        szs.extend_from_slice(nouts);

        MLP {
            layers: (0..szs.len() - 1)
                .map(|i| Layer::new(szs[i], szs[i + 1]))
                .collect(),
        }
    }

    pub fn forward(&self, x: Vec<Value>) -> Vec<Value> {
        let mut v = x.clone();

        for layer in &self.layers {
            v = layer.forward(v);
        }

        v
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .map(|l| l.parameters())
            .flatten()
            .collect()
    }
}
