use std::fmt;

pub mod env;

pub trait State: Clone {
    fn is_initial(&self) -> bool;
    fn is_terminal(&self) -> bool;
}

pub trait Action: Clone {
    fn is_terminal(&self) -> bool;
    fn terminate(&self) -> Self;
}

pub trait Environment: Clone {
    type S: State;
    type A: Action;
    type R;

    fn s0(&self) -> Self::S;
    fn step(
        &self,
        state: &Self::S,
        action: &Self::A,
    ) -> Result<Step<Self::S, Self::A, Self::R>, InvalidTransitionError>;
    fn sample(&self, state: &Self::S) -> Self::A;
    fn reward(&self, transition: &Transition<Self::S, Self::A>) -> Self::R;
}

pub struct Transition<S: State, A: Action> {
    pub state: S,
    pub action: A,
    pub next_state: S,
}

#[derive(Debug, Clone)]
pub struct InvalidTransitionError {}

pub struct Step<S: State, A: Action, R> {
    pub transition: Transition<S, A>,
    pub reward: R,
}

impl fmt::Display for InvalidTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid transition!")
    }
}

impl<S: State, A: Action> Transition<S, A> {
    pub fn is_terminal(&self) -> bool {
        self.next_state.is_terminal() || self.action.is_terminal()
    }
}
