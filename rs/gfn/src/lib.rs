/// This defines the contract for all states.
pub trait State {
    fn is_initial(&self) -> bool;
    fn is_terminal(&self) -> bool;
}

/// This defines the contract for all actions.
pub trait Action {
    fn is_terminal(&self) -> bool;
}

pub trait Environment<S: State, A: Action, R> {
    fn step(&self, state: &S, action: &A) -> Step<S, A, R>;
    fn reward(&self, transition: &Transition<S, A>) -> R;
}

pub struct Transition<S: State, A: Action> {
    pub state: S,
    pub action: A,
    pub next_state: S,
}

pub struct Step<S: State, A: Action, R> {
    pub transition: Transition<S, A>,
    pub reward: R,
}

impl<S: State, A: Action> Transition<S, A> {
    pub fn is_terminal(&self) -> bool {
        self.next_state.is_terminal() || self.action.is_terminal()
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_trajectory_balance_loss() {}
}
