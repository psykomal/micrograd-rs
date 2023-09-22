use std::iter::zip;

use micrograd_rs::engine::{Op, Value};
use micrograd_rs::nn::{Layer, Neuron, ZeroGrad, MLP};

fn main() {
    let x = Value::new(1.0, vec![], None);
    let y = Value::new(2.0, vec![], None);

    let z = x + y;

    println!("{:?}", z);

    z.backward();

    // Inputs.
    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    // Desired targets.
    let ys = [1.0, -1.0, -1.0, 1.0];

    // Convert inputs of f64 to Value.
    let inputs: Vec<Vec<Value>> = xs
        .iter()
        .map(|xrow| {
            vec![
                Value::new(xrow[0], vec![], None),
                Value::new(xrow[1], vec![], None),
                Value::new(xrow[2], vec![], None),
            ]
        })
        .collect();

    // MLP with three inputs, two 4-size layers, and single output.
    let mlp = MLP::new(3, &[4, 4, 1]);

    let mut ypred: Vec<Value> = Vec::new();
    for _ in 0..100 {
        // Forward pass.
        ypred = Vec::new();
        for x in inputs.clone() {
            ypred.push(mlp.forward(x)[0].clone());
        }
        println!("{:#?}", ypred);
        let loss = zip(ys, ypred.iter())
            .map(|(ygt, yout)| (yout - ygt).pow(2.0))
            .fold(Value::new(0.0, vec![], None), |a, b| a + b);

        // Backward pass. Don't forget to reset grads.
        mlp.zero_grad();
        loss.backward();

        // Update.
        for p in mlp.parameters() {
            p.set_data(p.data() + (-0.01 * p.grad()));
        }
    }

    // Values data should be close to [1.0, -1.0, -1.0, 1.0].
    println!("{:#?}", ypred);
}
