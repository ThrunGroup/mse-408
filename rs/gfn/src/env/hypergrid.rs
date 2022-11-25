use crate::{
    Action, Environment, InvalidTransitionError, State, Step, Transition,
};
use ndarray::{Array, Array1, ArrayView1, Ix1};

// TODO(danj): rewards function
// TODO(danj): loss function
// TODO(danj): training loop

#[derive(Debug)]
pub struct HypergridState {
    coordinate: Array1<usize>,
    n_per_dim: usize,
}

impl Clone for HypergridState {
    fn clone(&self) -> Self {
        Self {
            coordinate: self.coordinate.clone(),
            ..*self
        }
    }
}

impl State for HypergridState {
    fn is_initial(&self) -> bool {
        self.coordinate.sum() == 0
    }

    fn is_terminal(&self) -> bool {
        self.coordinate.iter().any(|x| *x == self.n_per_dim - 1)
    }
}

impl HypergridState {
    pub fn new(n_dims: usize, n_per_dim: usize) -> Self {
        Self {
            coordinate: Array1::zeros(n_dims),
            n_per_dim,
        }
    }

    fn apply(
        &self,
        action: &HypergridAction,
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
pub struct HypergridAction {
    direction: usize,
    terminal: usize,
}

impl Action for HypergridAction {
    fn is_terminal(&self) -> bool {
        self.direction == self.terminal
    }
}

#[derive(Debug, Clone)]
pub struct Hypergrid {
    s0: HypergridState,
    x_space: Array1<f64>,
    r: fn(x: &ArrayView1<f64>) -> f64,
}

impl Environment for Hypergrid {
    type S = HypergridState;
    type A = HypergridAction;
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

    fn reward(
        &self,
        transition: &Transition<HypergridState, HypergridAction>,
    ) -> f64 {
        let x = self.x(&transition.next_state);
        (self.r)(&x)
    }
}

impl Hypergrid {
    pub fn new(
        n_dims: usize,
        n_per_dim: usize,
        x_min: f64,
        x_max: f64,
        r: fn(&ArrayView1<f64>) -> f64,
    ) -> Self {
        Self {
            s0: HypergridState::new(n_dims, n_per_dim),
            x_space: Array::linspace(x_min, x_max, n_per_dim),
            r,
        }
    }

    fn x(&self, state: &HypergridState) -> ArrayView1<f64> {
        self.x_space.slice::<Ix1>(state.coordinate)
    }
}

fn corners(x: &ArrayView1<f64>) -> f64 {
    // return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
    // + r_0
    // x.gt(&0.5).all().into() as f64 * 0.5
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypergrid() {
        let grid = Hypergrid::new(2, 8, -1.0, 1.0, corners);
        let x = grid.x(&grid.s0);
        assert_eq!(x[0], -1.0);
    }
}
