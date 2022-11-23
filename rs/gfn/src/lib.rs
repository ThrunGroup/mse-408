use std::collections::HashMap;
use tch::Tensor;

/// This defines the contract for all states.
pub trait State {
    fn is_initial(&self) -> bool;
    fn is_terminal(&self) -> bool;
}

/// This defines the contract for all actions.
pub trait Action {
    fn is_terminal(&self) -> bool;
}

pub trait Environment<S, A: Action, R, Idx> {
    fn step(&self, state: S, action: A) -> Step<S, A, R>;
    fn reward(&self, transition: Transition<S, A>) -> R;
}

pub struct Transition<S, A> {
    pub state: S,
    pub action: A,
    pub next_state: S,
}

pub struct Step<S, A, R> {
    pub transition: Transition<S, A>,
    pub reward: R,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_balance_loss() {}
}
