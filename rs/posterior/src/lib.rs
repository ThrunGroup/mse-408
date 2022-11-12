// helper functions for environment in rust

trait Flow<F> {
    fn flow(&self) -> F;
}

trait Action {
    fn is_terminal(&self) -> bool;
    fn is_stop(&self) -> bool;
}

trait DiscreteState<F: Sized>: Sized {
    fn in_flows(&self) -> Vec<Box<dyn Flow<F>>>;
    fn out_flows(&self) -> Vec<Box<dyn Flow<F>>>;
}

trait ContinuousState: Sized {}

trait Environment<S, A: Action, R> {
    fn step(&self, state: S, action: A) -> Step<S, A, R>;
    fn flow(&self, transition: Transition<S, A>) -> Box<dyn Flow<R>>;
    fn flows(
        &self,
        transitions: Vec<Transition<S, A>>,
    ) -> Vec<Box<dyn Flow<R>>>;
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
