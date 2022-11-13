use std::ops::Index;

trait Flows<F, Idx>: Index<Idx, Output = F> + Iterator<Item = F> {}

trait Transitions<S, A, Idx>:
    Index<Idx, Output = Transition<S, A>> + Iterator<Item = Transition<S, A>>
{
}

trait StateFlow {
    fn in_flows<F, Idx>(&self) -> dyn Flows<F, Idx>;
    fn out_flows<F, Idx>(&self) -> dyn Flows<F, Idx>;
}

trait Action {
    fn is_terminal(&self) -> bool;
    fn is_stop(&self) -> bool;
}

trait Environment<S, A: Action, F, Idx> {
    fn step(&self, state: S, action: A) -> Step<S, A, F>;
    fn flow(&self, transition: Transition<S, A>) -> F;
    fn flows(
        &self,
        transitions: dyn Transitions<S, A, Idx>,
    ) -> dyn Flows<F, Idx>;
}

struct Transition<S, A> {
    state: S,
    action: A,
    next_state: S,
}

struct Step<S, A, R> {
    transition: Transition<S, A>,
    reward: R,
}
