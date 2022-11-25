use crate::{
    Action, Environment, InvalidTransitionError, State, Step, Transition,
};
use tch::Tensor;

// TODO(danj): tensor index -> get value
// TODO(danj): tensor reward
// TODO(danj): map tensor to x space

#[derive(Debug)]
struct HypergridState {
    coordinate: Tensor,
    n_per_dim: usize,
    x_min: f64,
    x_max: f64,
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
        self.coordinate.contains(&(&self.n_per_dim - 1))
    }
}

impl HypergridState {
    fn apply(
        &self,
        action: &HypergridAction,
    ) -> Result<Self, InvalidTransitionError> {
        if action.is_terminal() {
            return Ok(self.clone());
        }
        let coordinate = self.coordinate.copy().index_add_(
            0,
            &Tensor::from(action.direction as i64),
            &Tensor::from(1),
        );
        if coordinate[action.direction] < self.n_per_dim {
            Ok(Self {
                coordinate,
                ..*self
            })
        } else {
            Err(InvalidTransitionError {})
        }
    }

    fn x(&self) -> f64 {
        0.0
    }
}

#[derive(Debug, Copy, Clone)]
struct HypergridAction {
    direction: usize,
    terminal: usize,
}

impl Action for HypergridAction {
    fn is_terminal(&self) -> bool {
        self.direction == self.terminal
    }
}

#[derive(Debug, Clone)]
struct Hypergrid {
    s0: HypergridState,
    r: fn(&Transition<HypergridState, HypergridAction>) -> f64,
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
        (self.r)(transition)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_tensor_api() {}
}
