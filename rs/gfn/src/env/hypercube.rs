use crate::{
    Action, Environment, InvalidTransitionError, State, Step, Transition,
};
use ndarray::{Array, Array1, Axis};

// TODO(danj): rewards function
// TODO(danj): loss function
// TODO(danj): training loop

#[derive(Debug, Clone)]
pub struct HypercubeState {
    coordinate: Array1<usize>,
    n_per_dim: usize,
}

impl State for HypercubeState {
    fn is_initial(&self) -> bool {
        self.coordinate.sum() == 0
    }

    fn is_terminal(&self) -> bool {
        self.coordinate.iter().any(|x| *x == self.n_per_dim - 1)
    }
}

impl HypercubeState {
    pub fn new(n_dims: usize, n_per_dim: usize) -> Self {
        Self {
            coordinate: Array1::zeros(n_dims),
            n_per_dim,
        }
    }

    fn apply(
        &self,
        action: &HypercubeAction,
    ) -> Result<Self, InvalidTransitionError> {
        if action.is_terminal() {
            return Ok(self.clone());
        }
        let mut coordinate = self.coordinate.clone();
        coordinate[action.direction] += 1;
        if coordinate[action.direction] < self.n_per_dim {
            Ok(Self {
                coordinate,
                ..*self
            })
        } else {
            Err(InvalidTransitionError {})
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct HypercubeAction {
    direction: usize,
    terminal: usize,
}

impl Action for HypercubeAction {
    fn is_terminal(&self) -> bool {
        self.direction == self.terminal
    }

    fn terminate(&self) -> Self {
        Self {
            direction: self.terminal,
            terminal: self.terminal,
        }
    }
}

#[derive(Clone)]
pub struct Hypercube {
    s0: HypercubeState,
    x_space: Array1<f64>,
    r: fn(x: &Array1<f64>) -> f64,
}

impl Environment for Hypercube {
    type S = HypercubeState;
    type A = HypercubeAction;
    type R = f64;

    fn s0(&self) -> Self::S {
        self.s0.clone()
    }

    fn step(
        &self,
        state: &Self::S,
        action: &Self::A,
    ) -> Result<Step<Self::S, Self::A, Self::R>, InvalidTransitionError> {
        let transition = Transition {
            state: state.clone(),
            action: action.clone(),
            next_state: state.apply(action)?,
        };
        let reward = self.reward(&transition);
        Ok(Step { transition, reward })
    }

    fn sample(&self, state: &Self::S) -> Self::A {
        // TODO(danj): use model to sample state
    }

    fn reward(
        &self,
        transition: &Transition<HypercubeState, HypercubeAction>,
    ) -> f64 {
        let x = self.x_space.select(
            Axis(0),
            transition.next_state.coordinate.as_slice().unwrap(),
        );
        (self.r)(&x)
    }
}

impl Hypercube {
    pub fn new(
        n_dims: usize,
        n_per_dim: usize,
        x_min: f64,
        x_max: f64,
        r: fn(&Array1<f64>) -> f64,
    ) -> Self {
        Self {
            s0: HypercubeState::new(n_dims, n_per_dim),
            x_space: Array::linspace(x_min, x_max, n_per_dim),
            r,
        }
    }
}

fn corners(x: &Array1<f64>) -> f64 {
    // return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
    // + r_0
    // x.gt(&0.5).all().into() as f64 * 0.5
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypercube() {
        let (n_dims, n_per_dim, x_min, x_max) = (2, 8, -1.0, 1.0);
        let grid = Hypercube::new(n_dims, n_per_dim, x_min, x_max, corners);
        let s0 = grid.s0();
        let a0 = grid.sample(&s0);
        let step = grid.step(&s0, &a0).expect("simple step error!");
        let r = grid.reward(&step.transition);
        assert_eq!(r, -1.0);
    }
}
