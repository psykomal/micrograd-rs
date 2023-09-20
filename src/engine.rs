use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Debug)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Pow,
    ReLU,
    TanH,
}

#[derive(Clone, Debug)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub _prev: Vec<Value>,
    pub _op: Option<Op>,
}

impl Value {
    pub fn new(data: f64, _children: Vec<Value>, _op: Option<Op>) -> Self {
        Value {
            data,
            grad: 0.0,
            _prev: _children,
            _op: _op,
        }
    }

    pub fn exp(&self) -> Self {
        let result = Self::new(self.data.exp(), vec![self.clone()], Some(Op::Exp));
        result
    }

    pub fn pow(&self, exponent: f64) -> Self {
        let result = Self::new(self.data.powf(exponent), vec![self.clone()], Some(Op::Pow));
        result
    }

    pub fn relu(&self) -> Self {
        let result = Self::new(self.data.max(0.0), vec![self.clone()], Some(Op::ReLU));
        result
    }

    pub fn tanh(&self) -> Self {
        let result = Self::new(self.data.tanh(), vec![self.clone()], Some(Op::TanH));
        result
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let res = Value::new(
            self.data + other.data,
            vec![self.clone(), other.clone()],
            Some(Op::Add),
        );
        res
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Value {
        Value::new(
            self.data - other.data,
            vec![self.clone(), other.clone()],
            Some(Op::Sub),
        )
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, other: Value) -> Value {
        Value::new(
            self.data / other.data,
            vec![self.clone(), other.clone()],
            Some(Op::Div),
        )
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        Value::new(
            self.data * other.data,
            vec![self.clone(), other.clone()],
            Some(Op::Mul),
        )
    }
}
