use super::context::Env;
use crate::env::{context::Perspective, step::Step};
use burn::{
    prelude::ToElement,
    tensor::{Int, Tensor, backend::Backend},
};
use std::{array::from_fn, marker::PhantomData};

#[derive(Clone)]
pub struct VecEnv<B: Backend, E: Env<B>, const N: usize> {
    pub envs: [E; N],
    _marker: PhantomData<B>,
}

impl<B: Backend, E: Env<B> + Clone, const N: usize> VecEnv<B, E, N> {
    pub fn new(base_env: E) -> Self {
        Self {
            envs: from_fn(|_| base_env.clone()),
            _marker: PhantomData,
        }
    }

    pub fn reset(&mut self) {
        self.envs.iter_mut().for_each(|e| e.reset());
    }

    pub fn state_tensor(&mut self, perspective: Perspective, device: &B::Device) -> Tensor<B, 2> {
        let states = self
            .envs
            .iter_mut()
            .map(|env| env.state_tensor(perspective, device).0)
            .collect();
        Tensor::cat(states, 0)
    }

    pub fn step_simultaneous(
        &mut self,
        p_actions: Tensor<B, 2, Int>,
        t_actions: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> (Step<B>, Step<B>) {
        let p_acts = p_actions.into_data().to_vec::<i32>().unwrap();
        let t_acts = t_actions.into_data().to_vec::<i32>().unwrap();

        let mut p_steps = Vec::with_capacity(N);
        let mut t_steps = Vec::with_capacity(N);

        for i in 0..N {
            let (p_s, t_s) = self.envs[i].step_simultaneous(
                Tensor::from_data([[p_acts[i]]], device),
                Tensor::from_data([[t_acts[i]]], device),
                device,
            );

            if p_s.done.clone().any().into_scalar().to_bool() {
                self.envs[i].reset();
            }

            p_steps.push(p_s);
            t_steps.push(t_s);
        }

        (self.combine(p_steps), self.combine(t_steps))
    }

    fn combine(&self, s: Vec<Step<B>>) -> Step<B> {
        Step::new(
            Tensor::cat(s.iter().map(|x| x.next_state.clone()).collect(), 0),
            Tensor::cat(s.iter().map(|x| x.reward.clone()).collect(), 0),
            Tensor::cat(s.iter().map(|x| x.done.clone()).collect(), 0),
        )
    }
}
