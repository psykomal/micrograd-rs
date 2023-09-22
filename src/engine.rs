use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::{
    cell::RefCell,
    collections::HashSet,
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Op {
    Add,
    Mul,
    Exp,
    Pow,
    ReLU,
    TanH,
}

impl Display for Op {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        match self {
            Op::Add => {
                write!(f, "+")
            }
            Op::Mul => {
                write!(f, "*")
            }
            Op::Exp => {
                write!(f, "exp")
            }
            Op::Pow => {
                write!(f, "pow")
            }
            Op::ReLU => {
                write!(f, "relu")
            }
            Op::TanH => {
                write!(f, "tanh")
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct InnerValue {
    pub data: f64,
    pub grad: f64,
    pub _prev: Vec<Value>,
    pub _op: Option<Op>,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<InnerValue>>);

impl Value {
    pub fn new(data: f64, _children: Vec<Value>, _op: Option<Op>) -> Self {
        Self(Rc::new(RefCell::new(InnerValue {
            data,
            grad: 0.0,
            _prev: _children,
            _op,
        })))
    }

    pub fn exp(&self) -> Self {
        let result = Self::new(
            self.0.borrow().data.exp(),
            vec![self.clone()],
            Some(Op::Exp),
        );
        result
    }

    pub fn pow(&self, exponent: f64) -> Self {
        let result = Self::new(
            self.0.borrow().data.powf(exponent),
            vec![self.clone()],
            Some(Op::Pow),
        );
        result
    }

    pub fn relu(&self) -> Self {
        let result = Self::new(
            self.0.borrow().data.max(0.0),
            vec![self.clone()],
            Some(Op::ReLU),
        );
        result
    }

    pub fn tanh(&self) -> Self {
        let result = Self::new(
            self.0.borrow().data.tanh(),
            vec![self.clone()],
            Some(Op::TanH),
        );
        result
    }

    pub fn op(&self) -> Option<Op> {
        self.0.borrow()._op.clone()
    }

    pub fn lvalue(&self) -> Self {
        self.0.borrow()._prev[0].clone()
    }

    pub fn rvalue(&self) -> Self {
        self.0.borrow()._prev[1].clone()
    }

    pub fn add_grad(&self, grad: f64) {
        self.0.borrow_mut().grad += grad
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    pub fn children(&self) -> Vec<Self> {
        self.0.borrow()._prev.clone()
    }

    pub fn backward(&self) {
        fn backward_prev(value: &Value) {
            match value.0.borrow()._op {
                Some(Op::Add) => {
                    value.lvalue().add_grad(value.grad());
                    value.rvalue().add_grad(value.grad());
                }
                Some(Op::Mul) => {
                    let lval = value.lvalue();
                    let rval = value.rvalue();

                    lval.add_grad(value.grad() * rval.data());
                    rval.add_grad(value.grad() * lval.data());
                }
                Some(Op::Exp) => {
                    value.lvalue().add_grad(value.grad() * value.data());
                }
                Some(Op::Pow) => {
                    let lvalue = value.lvalue();
                    let lval = lvalue.data();
                    lvalue.add_grad(value.grad() * (lval * value.data().powf(lval - 1.0)));
                }
                Some(Op::ReLU) => {
                    let lvalue = value.lvalue();
                    if value.grad() > 0.0 {
                        lvalue.add_grad(value.grad());
                    }
                }
                Some(Op::TanH) => {
                    let lvalue = value.lvalue();
                    let t = value.data();
                    lvalue.add_grad(value.grad() * (1.0 - t.powf(2.0)));
                }
                None => {}
            }
        }

        fn topological_sort(v: &Value, topo: &mut Vec<Value>, set: &mut HashSet<Value>) {
            // Topological sort algorithm to determine order of backward pass

            if !set.contains(v) {
                set.insert(v.clone());

                for child in &v.0.borrow()._prev {
                    topological_sort(child, topo, set);
                }
                topo.push(v.clone());
            }
        }

        let mut topo = Vec::new();
        let mut set = HashSet::new();
        topological_sort(self, &mut topo, &mut set);

        // set self grad to 1.0
        self.0.borrow_mut().grad = 1.0;
        for value in topo.iter().rev() {
            backward_prev(value);
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        Display::fmt(&self, f)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(
            f,
            "Value(data={:#?}, grad={:#?} )",
            self.data(),
            self.grad()
        )
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let res = Value::new(
            self.0.borrow().data + other.0.borrow().data,
            vec![self.clone(), other.clone()],
            Some(Op::Add),
        );
        res
    }
}

// x + 1.0
impl Add<f64> for Value {
    type Output = Value;
    fn add(self, other: f64) -> Value {
        self + Value::new(other, vec![], None)
    }
}

// 1.0 + x
impl Add<Value> for f64 {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        Value::new(self, vec![], None) + other
    }
}

// &x + &y
impl Add for &Value {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        self.clone() + other.clone()
    }
}

// &x + 1.0
impl Add<f64> for &Value {
    type Output = Value;
    fn add(self, other: f64) -> Value {
        self + &Value::new(other, vec![], None)
    }
}

// 1.0 + &x
impl Add<&Value> for f64 {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        other + self
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        Value::new(
            self.0.borrow().data * other.0.borrow().data,
            vec![self.clone(), other.clone()],
            Some(Op::Mul),
        )
    }
}

impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, other: f64) -> Value {
        self * Value::new(other, vec![], None)
    }
}

impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        other * self
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, other: Self) -> Value {
        self.clone() * other.clone()
    }
}

// &x * 1.0
impl Mul<f64> for &Value {
    type Output = Value;
    fn mul(self, other: f64) -> Value {
        self * &Value::new(other, vec![], None)
    }
}

impl Mul<&Value> for f64 {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        other * self
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * -1.0
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * -1.0
    }
}

// x - y
impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

// x - 1.0
impl Sub<f64> for Value {
    type Output = Value;
    fn sub(self, other: f64) -> Value {
        self + (-other)
    }
}

// 1.0 - x
impl Sub<Value> for f64 {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

// &x - &y
impl Sub for &Value {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        self + &(-other)
    }
}

// &x - 1.0
impl Sub<f64> for &Value {
    type Output = Value;
    fn sub(self, other: f64) -> Value {
        self + (-other)
    }
}

// 1.0 - &x
impl Sub<&Value> for f64 {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        self + (-other)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, other: Value) -> Value {
        self * other.pow(-1.0)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}
