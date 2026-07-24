use burn::{
    Tensor,
    tensor::{Int, backend::Backend},
};
use rand_distr::{Distribution, multi::Dirichlet};

use crate::agent::{MuZeroAgent};
use crate::mz_config::MuZeroConfig;

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

pub fn search<B: Backend>(
    obs: Tensor<B, 2>,
    mz_conf: &MuZeroConfig,
    mz_agent: &MuZeroAgent<B>,
    tau: f32,
) -> (Vec<f32>, f32, usize) {
    let device = obs.device();
    let discount: f32 = mz_conf.discount;
    let mut norm = QNormalization::default();

    let mut nodes =
        Vec::<Node<B>>::with_capacity((mz_conf.num_simulations + 1) * mz_conf.action_space);

    // Initialize and expand root (node 0)
    let (root_hidden_state, root_reward, root_value, root_policy) = mz_agent.initial_forward(obs);
    let dirichlet = Dirichlet::new(&vec![mz_conf.dirichlet_alpha; mz_conf.action_space]).unwrap();
    let noise = dirichlet.sample(&mut rand::rng());
    let frac = mz_conf.root_exploration_fraction;

    let root_reward_val = read_scalar(root_reward);
    let root_value_val = read_scalar(root_value);
    let root_policy_vals = read_vec(root_policy);

    let root_node = Node {
        visits: 1,
        action: 0, // This action is irelevant
        hidden_state: Some(root_hidden_state.squeeze_dim(0)),
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

        // Expansion: run recurrent_forward from parent with action that led to leaf
        let [.., parent_idx, leaf_idx] = path.as_slice() else {
            unreachable!()
        };
        let parent_hs = match &nodes[*parent_idx].hidden_state {
            Some(hs) => hs,
            None => panic!("Parent node has no hidden state!"),
        };
        let action_tensor = Tensor::<B, 1, Int>::from_data([nodes[*leaf_idx].action], &device);
        let (new_hs, new_reward, new_value, new_policy) = mz_agent.recurrent_forward(
            parent_hs.clone().unsqueeze(),
            action_tensor,
            mz_conf.action_space,
        );

        let new_policy_len = new_policy.dims()[1];

        let nodes_len = nodes.len();

        let new_reward_val = read_scalar(new_reward);
        let new_value_val = read_scalar(new_value);
        let new_policy_vals = read_vec(new_policy);
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

        curr_node.hidden_state = Some(new_hs.squeeze_dim(0));
        // println!("Used {}ms for nn inference.", el.as_millis());
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

// TODO: Only use tensor operations?
fn puct(q_value: f32, prior: f32, parent_visits: i32, child_visits: i32) -> f32 {
    let c1: f32 = 1.25;
    let c2: i32 = 19652;
    q_value
        + prior * (parent_visits as f32).sqrt() / (1 + child_visits) as f32
            * (c1 + ((parent_visits + c2 + 1) as f32 / c2 as f32).ln())
}
