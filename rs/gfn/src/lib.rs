use std::ops::Index;

pub trait Flows<F, Idx>: Index<Idx, Output = F> + Iterator<Item = F> {}

pub trait Transitions<S, A, Idx>:
    Index<Idx, Output = Transition<S, A>> + Iterator<Item = Transition<S, A>>
{
}

pub trait StateFlow {
    fn in_flows<F, Idx>(&self) -> dyn Flows<F, Idx>;
    fn out_flows<F, Idx>(&self) -> dyn Flows<F, Idx>;
}

pub trait Action {
    fn is_terminal(&self) -> bool;
    fn is_stop(&self) -> bool;
}

pub trait Environment<S, A: Action, F, Idx> {
    fn step(&self, state: S, action: A) -> Step<S, A, F>;
    fn flow(&self, transition: Transition<S, A>) -> F;
    fn flows(
        &self,
        transitions: dyn Transitions<S, A, Idx>,
    ) -> dyn Flows<F, Idx>;
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
