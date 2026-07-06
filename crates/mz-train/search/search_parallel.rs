use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use burn::{
    Tensor,
    tensor::{Int, backend::Backend},
};
use crossbeam::channel::{Receiver, Sender, bounded};
use rand_distr::{Distribution, multi::Dirichlet};

use crate::mz_config::MuZeroConfig;
use crate::networks::MuZeroNets;

struct Node<B: Backend> {
    visits: usize,
    action: usize,
    hidden_state: Option<Tensor<B, 1>>,
    cumulative_value: f32,
    reward: f32,
    children: Vec<usize>,
    policy: f32,
}

struct QNormalization {
    q_max: f32,
    q_min: f32,
}

impl QNormalization {
    fn get_q(&mut self, q_value: f32) -> f32 {
        self.q_max = self.q_max.max(q_value);
        self.q_min = self.q_min.min(q_value);
        let epsilon = 0.001;

        (q_value - self.q_min) / (self.q_max - self.q_min + epsilon)
    }
}

impl Default for QNormalization {
    fn default() -> Self {
        QNormalization {
            q_max: f32::NEG_INFINITY,
            q_min: f32::INFINITY,
        }
    }
}

/// (hidden_state, reward, value, policy) 
type InferResult<B> = (Tensor<B, 1>, f32, f32, Vec<f32>);
type RespondTx<B> = Sender<InferResult<B>>;

pub struct InitRequest<B: Backend> {
    obs: Tensor<B, 2>,
    respond_to: RespondTx<B>,
}

pub struct RecurrentRequest<B: Backend> {
    hidden_state: Tensor<B, 1>,
    action: usize,
    respond_to: RespondTx<B>,
}

#[derive(Clone)]
pub struct InferenceHandles<B: Backend> {
    init_tx: Sender<InitRequest<B>>,
    rec_tx: Sender<RecurrentRequest<B>>,
}

impl<B: Backend> InferenceHandles<B> {
    fn request_init(&self, obs: Tensor<B, 2>) -> InferResult<B> {
        let (respond_to, response) = bounded(1);
        self.init_tx
            .send(InitRequest { obs, respond_to })
            .expect("inference master thread is gone");
        response.recv().expect("inference master dropped request")
    }

    fn request_recurrent(&self, hidden_state: Tensor<B, 1>, action: usize) -> InferResult<B> {
        let (respond_to, response) = bounded(1);
        self.rec_tx
            .send(RecurrentRequest {
                hidden_state,
                action,
                respond_to,
            })
            .expect("inference master thread is gone");
        response.recv().expect("inference master dropped request")
    }
}

pub struct InferenceChannels<B: Backend> {
    pub handles: InferenceHandles<B>,
    pub init_rx: Receiver<InitRequest<B>>,
    pub rec_rx: Receiver<RecurrentRequest<B>>,
}

pub fn inference_channels<B: Backend>() -> InferenceChannels<B> {
    let (init_tx, init_rx) = crossbeam::channel::unbounded();
    let (rec_tx, rec_rx) = crossbeam::channel::unbounded();
    InferenceChannels {
        handles: InferenceHandles { init_tx, rec_tx },
        init_rx,
        rec_rx,
    }
}

pub fn inference_master<B: Backend, N: MuZeroNets<B>>(
    init_rx: Receiver<InitRequest<B>>,
    rec_rx: Receiver<RecurrentRequest<B>>,
    agent: Arc<Mutex<N>>,
    action_space: usize,
    init_batch_size: usize,
    rec_batch_size: usize,
    max_wait: Duration,
) {
    let mut init_pending: Vec<InitRequest<B>> = Vec::new();
    let mut rec_pending: Vec<RecurrentRequest<B>> = Vec::new();

    let mut init_alive = true;
    let mut rec_alive = true;

    while init_alive || rec_alive {
        if init_pending.is_empty() && rec_pending.is_empty() {
            // Nothing buffered: block until either queue produces something.
            match (init_alive, rec_alive) {
                (true, true) => crossbeam::select! {
                    recv(init_rx) -> msg => match msg {
                        Ok(r) => init_pending.push(r),
                        Err(_) => init_alive = false,
                    },
                    recv(rec_rx) -> msg => match msg {
                        Ok(r) => rec_pending.push(r),
                        Err(_) => rec_alive = false,
                    },
                },
                (true, false) => match init_rx.recv() {
                    Ok(r) => init_pending.push(r),
                    Err(_) => init_alive = false,
                },
                (false, true) => match rec_rx.recv() {
                    Ok(r) => rec_pending.push(r),
                    Err(_) => rec_alive = false,
                },
                (false, false) => break,
            }
            continue;
        }

        let deadline = Instant::now() + max_wait;
        loop {
            let init_full = init_pending.len() >= init_batch_size;
            let rec_full = rec_pending.len() >= rec_batch_size;
            if init_full || rec_full || Instant::now() >= deadline {
                break;
            }

            let mut progressed = false;
            if let Ok(r) = init_rx.try_recv() {
                init_pending.push(r);
                progressed = true;
            }
            if let Ok(r) = rec_rx.try_recv() {
                rec_pending.push(r);
                progressed = true;
            }
            if !progressed {
                std::thread::yield_now();
            }
        }

        if !init_pending.is_empty() {
            let batch = std::mem::take(&mut init_pending);
            flush_init(batch, &agent);
        }
        if !rec_pending.is_empty() {
            let batch = std::mem::take(&mut rec_pending);
            flush_rec(batch, &agent, action_space);
        }
    }
}

fn distribute<B: Backend>(
    respond_to: Vec<RespondTx<B>>,
    hidden_state: Tensor<B, 2>,
    reward: Tensor<B, 2>,
    value: Tensor<B, 2>,
    policy: Tensor<B, 2>,
) {
    for (i, tx) in respond_to.into_iter().enumerate() {
        let hs_i = hidden_state.clone().narrow(0, i, 1).squeeze_dim(0);
        let r_i = read_scalar(reward.clone().narrow(0, i, 1));
        let v_i = read_scalar(value.clone().narrow(0, i, 1));
        let p_i = read_vec(policy.clone().narrow(0, i, 1));
        let _ = tx.send((hs_i, r_i, v_i, p_i));
    }
}

fn flush_init<B: Backend, N: MuZeroNets<B>>(batch: Vec<InitRequest<B>>, agent: &Arc<Mutex<N>>) {
    let mut obs = Vec::with_capacity(batch.len());
    let mut respond_to = Vec::with_capacity(batch.len());
    for req in batch {
        obs.push(req.obs);
        respond_to.push(req.respond_to);
    }

    let obs_batch = Tensor::cat(obs, 0);
    let (hidden_state, reward, value, policy) = {
        let agent = agent.lock().unwrap();
        agent.initial_inference(obs_batch)
    };
    distribute(respond_to, hidden_state, reward, value, policy);
}

fn flush_rec<B: Backend, N: MuZeroNets<B>>(
    batch: Vec<RecurrentRequest<B>>,
    agent: &Arc<Mutex<N>>,
    action_space: usize,
) {
    let device = batch[0].hidden_state.device();
    let mut hidden_states = Vec::with_capacity(batch.len());
    let mut actions = Vec::with_capacity(batch.len());
    let mut respond_to = Vec::with_capacity(batch.len());
    for req in batch {
        hidden_states.push(req.hidden_state);
        actions.push(req.action as i32);
        respond_to.push(req.respond_to);
    }

    let hs_batch: Tensor<B, 2> = Tensor::stack(hidden_states, 0);
    let action_batch = Tensor::<B, 1, Int>::from_data(actions.as_slice(), &device);
    let (hidden_state, reward, value, policy) = {
        let agent = agent.lock().unwrap();
        agent.recurrent_inference(hs_batch, action_batch, action_space)
    };
    distribute(respond_to, hidden_state, reward, value, policy);
}

pub fn search<B: Backend>(
    obs: Tensor<B, 2>,
    mz_conf: &MuZeroConfig,
    tau: f32,
    inference: &InferenceHandles<B>,
) -> (Vec<f32>, f32, usize) {
    let discount: f32 = mz_conf.discount;
    let mut norm = QNormalization::default();

    let mut nodes =
        Vec::<Node<B>>::with_capacity((mz_conf.num_simulations + 1) * mz_conf.action_space);

    // Initialize and expand root (node 0)
    let (root_hidden_state, root_reward_val, root_value_val, root_policy_vals) =
        inference.request_init(obs);
    let dirichlet = Dirichlet::new(&vec![mz_conf.dirichlet_noise; mz_conf.action_space]).unwrap();
    let noise = dirichlet.sample(&mut rand::rng());
    let frac = mz_conf.root_exploration_fraction;

    let root_node = Node {
        visits: 1,
        action: 0, // This action is irelevant
        hidden_state: Some(root_hidden_state),
        children: (1..=mz_conf.action_space).collect(),
        cumulative_value: root_value_val,
        reward: root_reward_val,
        policy: 0.,
    };

    nodes.push(root_node);

    for (idx, policy) in root_policy_vals.into_iter().enumerate() {
        nodes.push(Node {
            visits: 0,
            action: idx,
            hidden_state: None,
            cumulative_value: 0.,
            reward: 0.,
            children: Vec::new(),
            policy: (1.0 - frac) * policy + frac * noise[idx],
        });
    }

    for _sim_step in 0..mz_conf.num_simulations {
        // Selection: PUCT
        let mut curr_node_idx = 0;
        let mut path = vec![0usize];
        loop {
            let curr_node = &nodes[curr_node_idx];
            if curr_node.children.is_empty() {
                // Terminal node, go to expansion
                break;
            }

            let parent_visits = curr_node.visits;

            // Find the best child
            let mut best_puct = f32::NEG_INFINITY;
            let mut best_node = 0usize;
            for child_idx in &curr_node.children {
                let child = &nodes[*child_idx];
                let q_value = match child.visits {
                    0 => 0.,
                    _ => norm.get_q(child.cumulative_value / child.visits as f32),
                };
                let puct_value = puct(
                    q_value,
                    child.policy,
                    parent_visits as i32,
                    child.visits as i32,
                );

                if puct_value > best_puct {
                    best_puct = puct_value;
                    best_node = *child_idx;
                }
            }

            curr_node_idx = best_node;
            path.push(curr_node_idx);
        }

        // Expansion: run recurrent_inference from parent with action that led to leaf
        let [.., parent_idx, leaf_idx] = path.as_slice() else {
            unreachable!()
        };
        let parent_hs = match &nodes[*parent_idx].hidden_state {
            Some(hs) => hs.clone(),
            None => panic!("Parent node has no hidden state!"),
        };
        let action = nodes[*leaf_idx].action;
        let (new_hs, new_reward_val, new_value_val, new_policy_vals) =
            inference.request_recurrent(parent_hs, action);

        let new_policy_len = new_policy_vals.len();
        let nodes_len = nodes.len();

        for (idx, policy) in new_policy_vals.into_iter().enumerate() {
            nodes.push(Node {
                visits: 0,
                action: idx,
                hidden_state: None,
                cumulative_value: 0.,
                reward: 0.,
                children: Vec::new(),
                policy,
            });
        }

        let curr_node = &mut nodes[*leaf_idx];

        curr_node.hidden_state = Some(new_hs);
        curr_node.reward = new_reward_val;

        for idx in 0..new_policy_len {
            curr_node.children.push(nodes_len + idx);
        }

        // Backprop: walk path from leaf to root, accumulate discounted returns
        let mut back_value = new_value_val;
        for &node_idx in path.iter().rev() {
            let curr_node = &mut nodes[node_idx];
            curr_node.visits += 1;
            curr_node.cumulative_value += back_value;
            back_value = curr_node.reward + discount * back_value;
        }
    }

    let root_node = &nodes[0];
    let value = nodes[0].cumulative_value / (nodes[0].visits as f32);
    let mut visit_distribution = vec![0.0f32; mz_conf.action_space];
    if tau == 0.0 {
        let best_child_idx = root_node
            .children
            .iter()
            .max_by_key(|child_idx| nodes[**child_idx].visits);
        let best_action = match best_child_idx {
            Some(child_idx) => nodes[*child_idx].action,
            None => panic!("There are no child nodes."),
        };
        visit_distribution[best_action] = 1.0;
        (visit_distribution, value, best_action)
    } else {
        let visit_sum: f32 = root_node
            .children
            .iter()
            .map(|child_idx| (nodes[*child_idx].visits as f32).powf(1.0 / tau))
            .sum();
        let mut highest_visits = 0;
        let mut best_action = 0;
        for child_idx in &root_node.children {
            let action = nodes[*child_idx].action;
            let child_visits = nodes[*child_idx].visits;
            if child_visits > highest_visits {
                highest_visits = child_visits;
                best_action = action;
            }
            visit_distribution[action] = (child_visits as f32).powf(1.0 / tau) / visit_sum;
        }
        (visit_distribution, value, best_action)
    }
}

// Single-element [1, 1] tensor -> f32, one host transfer.
fn read_scalar<B: Backend>(t: Tensor<B, 2>) -> f32 {
    t.into_data().to_vec::<f32>().unwrap()[0]
}

// [1, N] tensor -> Vec<f32>, one host transfer instead of N.
fn read_vec<B: Backend>(t: Tensor<B, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

fn puct(q_value: f32, prior: f32, parent_visits: i32, child_visits: i32) -> f32 {
    let c1: f32 = 1.25;
    let c2: i32 = 19652;
    q_value
        + prior * (parent_visits as f32).sqrt() / (1 + child_visits) as f32
            * (c1 + ((parent_visits + c2 + 1) as f32 / c2 as f32).ln())
}
