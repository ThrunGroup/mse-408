use crate::{
    Action, Environment, InvalidTransitionError, State, Step, Transition,
};
use ndarray::{Array, Array1, Axis};

// TODO(danj): rewards function
// TODO(danj): API
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
    r0: f64,
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
        if !transition.is_terminal() {
            return 0.0;
        }
        let s_t = transition.next_state;
        let x_abs =
            (s_t.coordinate / (s_t.n_per_dim - 1) * 2 - 1).mapv(f64::abs);
        self.r0 + 0.5
    }
}

impl Hypercube {
    pub fn new(n_dims: usize, n_per_dim: usize, r_0: f64) -> Self {
        Self {
            s0: HypercubeState::new(n_dims, n_per_dim),
            r0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypercube() {
        let (n_dims, n_per_dim, r0) = (2, 8, 0.01);
        let grid = Hypercube::new(n_dims, n_per_dim, r0);
        let s0 = grid.s0();
        let a0 = grid.sample(&s0);
        let step = grid.step(&s0, &a0).expect("simple step error!");
        let r = grid.reward(&step.transition);
        assert_eq!(r, -1.0);
    }
}
