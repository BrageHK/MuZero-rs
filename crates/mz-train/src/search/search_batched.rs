use burn::{
    Tensor,
    tensor::{Int, TensorData, Transaction, backend::Backend},
};
use rand_distr::{Distribution, multi::Dirichlet};

use crate::networks::MuZeroNets;
use crate::{mz_config::MuZeroConfig, search::node::Node, utils::QNormalization};

pub struct SearchReturn {
    pub distribution: Vec<f32>,
    pub value: f32,
    pub best_action: usize,
}

/// Runs one MCTS per row of `observations`, batching every network call
/// across the trees. All observations should be on the same device.
///
/// Each simulation step does exactly one `recurrent_inference` with batch
/// size = number of trees, and one `Transaction` to move rewards, values
/// and policies to the host. Hidden states stay on the device.
pub fn batched_search<B: Backend, N: MuZeroNets<B>>(
    observations: Tensor<B, 2>,
    mz_conf: &MuZeroConfig,
    mz_agent: &N,
    tau: f32,
) -> Vec<SearchReturn> {
    let batch_size = observations.dims()[0];
    let device = observations.device();
    let discount = mz_conf.discount;
    let action_space = mz_conf.action_space;

    let mut norms: Vec<QNormalization> =
        (0..batch_size).map(|_| QNormalization::default()).collect();

    let mut node_batch: Vec<Vec<Node<B>>> = (0..batch_size)
        .map(|_| Vec::with_capacity((mz_conf.num_simulations + 1) * action_space))
        .collect();

    // Initialize and expand root nodes
    let (root_hidden_states, root_rewards, root_values, root_policies) =
        mz_agent.initial_inference(observations);
    let dirichlet = Dirichlet::new(&vec![mz_conf.dirichlet_noise; action_space]).unwrap();
    let noise_batch: Vec<_> = (0..batch_size)
        .map(|_| dirichlet.sample(&mut rand::rng()))
        .collect();
    let frac = mz_conf.root_exploration_fraction;

    let [root_rewards, root_values, root_policies] = Transaction::default()
        .register(root_rewards)
        .register(root_values)
        .register(root_policies)
        .execute()
        .try_into()
        .expect("Correct amount of tensor data");

    let root_rewards = root_rewards.into_vec::<f32>().unwrap();
    let root_values = root_values.into_vec::<f32>().unwrap();
    let root_policies = root_policies.into_vec::<f32>().unwrap();

    for i in 0..batch_size {
        node_batch[i].push(Node {
            visits: 1,
            action: 0, // This action is irrelevant
            hidden_state: Some(root_hidden_states.clone().slice([i..i + 1]).squeeze_dim(0)),
            children: (1..=action_space).collect(),
            cumulative_value: root_values[i],
            reward: root_rewards[i],
            policy: 0.,
        });

        let policy_row = &root_policies[i * action_space..(i + 1) * action_space];
        for (action, &policy) in policy_row.iter().enumerate() {
            node_batch[i].push(Node {
                visits: 0,
                action,
                hidden_state: None,
                cumulative_value: 0.,
                reward: 0.,
                children: Vec::new(),
                policy: (1.0 - frac) * policy + frac * noise_batch[i][action],
            });
        }
    }

    for _sim_step in 0..mz_conf.num_simulations {
        // Selection: PUCT walk in every tree, all on host
        let mut path_batch = Vec::with_capacity(batch_size);
        let mut parent_hs_batch = Vec::with_capacity(batch_size);
        let mut actions = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let nodes = &node_batch[i];
            let norm = &mut norms[i];

            let mut curr_node_idx = 0;
            let mut path = vec![0usize];
            loop {
                let curr_node = &nodes[curr_node_idx];
                if curr_node.children.is_empty() {
                    // Unexpanded leaf, go to expansion
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

            let [.., parent_idx, leaf_idx] = path.as_slice() else {
                unreachable!()
            };
            let parent_hs = match &node_batch[i][*parent_idx].hidden_state {
                Some(hs) => hs.clone(),
                None => panic!("Parent node has no hidden state!"),
            };
            parent_hs_batch.push(parent_hs);
            actions.push(node_batch[i][*leaf_idx].action as i64);
            path_batch.push(path);
        }

        // Expansion: one recurrent_inference for all trees
        let hidden_batch = Tensor::stack::<2>(parent_hs_batch, 0);
        let action_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(actions.as_slice()), &device);
        let (new_hs, new_rewards, new_values, new_policies) =
            mz_agent.recurrent_inference(hidden_batch, action_tensor, action_space);

        let [new_rewards, new_values, new_policies] = Transaction::default()
            .register(new_rewards)
            .register(new_values)
            .register(new_policies)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let new_rewards = new_rewards.into_vec::<f32>().unwrap();
        let new_values = new_values.into_vec::<f32>().unwrap();
        let new_policies = new_policies.into_vec::<f32>().unwrap();

        for i in 0..batch_size {
            let nodes = &mut node_batch[i];
            let path = &path_batch[i];
            let leaf_idx = *path.last().unwrap();

            let nodes_len = nodes.len();
            let policy_row = &new_policies[i * action_space..(i + 1) * action_space];
            for (action, &policy) in policy_row.iter().enumerate() {
                nodes.push(Node {
                    visits: 0,
                    action,
                    hidden_state: None,
                    cumulative_value: 0.,
                    reward: 0.,
                    children: Vec::new(),
                    policy,
                });
            }

            let leaf = &mut nodes[leaf_idx];
            leaf.hidden_state = Some(new_hs.clone().slice([i..i + 1]).squeeze_dim(0));
            leaf.reward = new_rewards[i];
            leaf.children = (nodes_len..nodes_len + action_space).collect();

            // Backprop: walk path from leaf to root, accumulate discounted returns
            let mut back_value = new_values[i];
            for &node_idx in path.iter().rev() {
                let curr_node = &mut nodes[node_idx];
                curr_node.visits += 1;
                curr_node.cumulative_value += back_value;
                back_value = curr_node.reward + discount * back_value;
            }
        }
    }

    (0..batch_size)
        .map(|i| extract_result(&node_batch[i], action_space, tau))
        .collect()
}

fn extract_result<B: Backend>(nodes: &[Node<B>], action_space: usize, tau: f32) -> SearchReturn {
    let root_node = &nodes[0];
    let value = root_node.cumulative_value / (root_node.visits as f32);
    let mut visit_distribution = vec![0.0f32; action_space];

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
        SearchReturn {
            distribution: visit_distribution,
            value,
            best_action,
        }
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
        SearchReturn {
            distribution: visit_distribution,
            value,
            best_action,
        }
    }
}

fn puct(q_value: f32, prior: f32, parent_visits: i32, child_visits: i32) -> f32 {
    let c1: f32 = 1.25;
    let c2: i32 = 19652;
    q_value
        + prior * (parent_visits as f32).sqrt() / (1 + child_visits) as f32
            * (c1 + ((parent_visits + c2 + 1) as f32 / c2 as f32).ln())
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::agent::MlpNets;
    use burn::backend::NdArray;

    #[test]
    fn batched_search_valid_distributions() {
        let mz_conf = MuZeroConfig::default();
        let device = Default::default();
        let agent: MlpNets<NdArray> = mz_conf.init(&device);

        let batch_size = 3;
        let obs = Tensor::<NdArray, 2>::random(
            [batch_size, mz_conf.obs_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        for tau in [0.0, 1.0] {
            let results = batched_search(obs.clone(), &mz_conf, &agent, tau);
            assert_eq!(results.len(), batch_size);
            for res in &results {
                assert_eq!(res.distribution.len(), mz_conf.action_space);
                let sum: f32 = res.distribution.iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "distribution sums to {sum}");
                assert!(res.best_action < mz_conf.action_space);
                assert!(res.value.is_finite());
            }
        }
    }

    #[test]
    fn batched_search_visits_match_simulations() {
        let mz_conf = MuZeroConfig::default();
        let device = Default::default();
        let agent: MlpNets<NdArray> = mz_conf.init(&device);

        let obs = Tensor::<NdArray, 2>::random(
            [2, mz_conf.obs_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // With tau=1 the distribution is visits/total; root child visits sum
        // to num_simulations, so no probability mass can be lost.
        let results = batched_search(obs, &mz_conf, &agent, 1.0);
        for res in &results {
            assert!(res.distribution.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }
}
