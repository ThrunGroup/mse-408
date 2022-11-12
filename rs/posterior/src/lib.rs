// helper functions for environment in rust
struct Step<S, A, R> {
    state: S,
    action: A,
    reward: R,
    next_state: S,
}

trait Action {
    fn is_terminal(&self) -> bool;
    fn is_stop(&self) -> bool;
}

struct Transition<S: Sized, A: Sized> {
    state: S,
    action: A,
}

trait State<A: Sized>: Sized {
    fn in_flows(&self) -> Vec<Transition<Self, A>>;
    fn out_flows(&self) -> Vec<Transition<Self, A>>;
}

trait Environment<S: State<A>, A: Action, R> {
    fn step(&self, state: &S, action: &A) -> Step<S, A, R> {}
}
