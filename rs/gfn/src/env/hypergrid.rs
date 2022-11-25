use crate::{
    Action, Environment, InvalidTransitionError, State, Step, Transition,
};
use tch::{Device, Tensor};

// TODO(danj): rewards function
// TODO(danj): loss function
// TODO(danj): training loop

pub struct HypergridState {
    coordinate: Tensor,
    n_per_dim: usize,
}

impl Clone for HypergridState {
    fn clone(&self) -> Self {
        Self {
            coordinate: self.coordinate.copy(),
            ..*self
        }
    }
}

impl State for HypergridState {
    fn is_initial(&self) -> bool {
        self.coordinate.sum(self.coordinate.kind()) == Tensor::from(0)
    }

    fn is_terminal(&self) -> bool {
        self.coordinate.eq((self.n_per_dim - 1) as i64).any().into()
    }
}

impl HypergridState {
    pub fn new(n_dims: usize, n_per_dim: usize) -> Self {
        let device = Device::cuda_if_available();
        Self {
            coordinate: Tensor::zeros(
                &[n_dims as i64],
                (tch::Kind::Int64, device),
            ),
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
        let coordinate = self.coordinate.index_add(
            0,
            &Tensor::from(action.direction as i64),
            &Tensor::from(1),
        );
        if i64::from(coordinate.get(action.direction as i64))
            < self.n_per_dim as i64
        {
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

pub struct Hypergrid {
    s0: HypergridState,
    x_space: Tensor,
    r: fn(&Tensor) -> Tensor,
}

impl Clone for Hypergrid {
    fn clone(&self) -> Self {
        Self {
            s0: self.s0.clone(),
            x_space: self.x_space.copy(),
            ..*self
        }
    }
}

impl Environment for Hypergrid {
    type S = HypergridState;
    type A = HypergridAction;
    type R = Tensor;

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
    ) -> Tensor {
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
        r: fn(&Tensor) -> Tensor,
    ) -> Self {
        let device = Device::cuda_if_available();
        let kind = tch::Kind::Int64;
        Self {
            s0: HypergridState::new(n_dims, n_per_dim),
            x_space: Tensor::linspace(
                x_min,
                x_max,
                n_per_dim as i64,
                (kind, device),
            ),
            r,
        }
    }

    fn x(&self, state: &HypergridState) -> Tensor {
        self.x_space.gather(0, &state.coordinate, false)
    }
}

fn corners(x: &Tensor) -> Tensor {
    // return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
    // + r_0
    // x.gt(&0.5).all().into() as f64 * 0.5
    Tensor::from(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypergrid() {
        let grid = Hypergrid::new(2, 8, -1.0, 1.0, corners);
        let x = grid.x(&grid.s0);
        let x0: f64 = x.get(0).into();
        assert_eq!(x0, -1.0);
    }
}
