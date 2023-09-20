use micrograd_rs::engine::{Op, Value};

fn main() {
    let x = Value::new(1.0, vec![], None);
    let y = Value::new(2.0, vec![], None);

    let z = x + y;

    println!("{:?}", z)
}
