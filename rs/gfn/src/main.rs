use gfn::{Action, Environment, State, Step, Transition};
use tch::{Scalar, Tensor};

#[derive(Debug)]
struct HypergridState {
    coordinate: Tensor,
    n_per_dim: usize,
    x_min: f64,
    x_max: f64,
}

#[derive(Debug, Copy, Clone)]
struct HypergridAction {
    direction: usize,
    terminal: usize,
}

struct Hypergrid {}

impl HypergridState {
    fn apply(&self, action: usize) -> Result<Self, &str> {
        if action.is_terminal() {
            return self.clone();
        }
        let mut coordinate = self.coordinate.copy();
        coordinate[action.direction] += 1;
        if coordinate[action.direction] < self.n_per_dim {
            Ok(Self {
                coordinate,
                ..*self
            })
        } else {
            Err("Attempt to advance to an invalid state.")
        }
    }

    fn x(&self) -> Tensor {}

    fn initial(&self) -> Self {
        Self {
            coordinate: self.coordinate.zeros_like(),
            ..*self
        }
    }
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
        self.coordinate.sum(self.coordinate.kind()) == Tensor::zeros(1.0)
    }

    fn is_terminal(&self) -> bool {
        self.coordinate.contains(&(&self.n_per_dim - 1))
    }
}

impl Action for HypergridAction {
    fn is_terminal(&self) -> bool {
        self.action.contains(&self.terminal)
    }
}

impl<S: State, A: Action, R> Environment<S, A, R> for Hypergrid {
    fn step(
        &self,
        state: &HypergridState,
        action: &HypergridAction,
    ) -> Result<Step<S, A, R>, &str> {
        let transition = Transition {
            state: state.copy(),
            action: action.copy(),
            next_state: state.apply(action)?,
        };
        let mut reward = 0;
        if transition.is_terminal() {
            reward = self.reward(state);
        }
        Ok(Step { transition, reward })
    }

    fn reward(state: &HypergridState) -> f64 {
        5.0
    }
}

fn main() {
    println!("Hello, world!");
}
