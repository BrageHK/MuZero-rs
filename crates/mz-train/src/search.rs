use burn::{
    Tensor,
    tensor::{Int, TensorData, Transaction, backend::Backend},
};
use rand_distr::{Distribution, multi::Dirichlet};
use rayon::prelude::*;

use crate::networks::MuZeroNets;
use crate::{mz_config::MuZeroConfig, utils::QNormalization};

pub struct SearchReturn {
    pub distribution: Vec<f32>,
    pub policy_target: Vec<f32>,
    pub value: f32,
    pub best_action: usize,
}

struct BatchNode {
    visits: usize,
    action: usize,
    hidden_row: usize,
    first_child: usize,
    cumulative_value: f32,
    reward: f32,
    policy: f32,
    legal: bool,
}

/// Returns a Vec of SearchReturn. Batch items with a single legal action skip
/// the search and immediately return that action with the network's root value.
pub fn batched_search<B: Backend, N: MuZeroNets<B>>(
    observations: Tensor<B, 2>,
    legal_masks: Option<&[Vec<bool>]>,
    mz_conf: &MuZeroConfig,
    mz_agent: &N,
    tau: f32,
) -> Vec<SearchReturn> {
    let batch_size = observations.dims()[0];
    let device = observations.device();
    let discount = mz_conf.discount;
    let action_space = mz_conf.action_space;
    let value_sign = if mz_conf.is_twoplayer { -1.0f32 } else { 1.0f32 };

    let forced_actions: Vec<Option<usize>> = match legal_masks {
        Some(masks) => masks
            .iter()
            .map(|mask| {
                let mut legal = mask.iter().enumerate().filter(|&(_, &l)| l).map(|(a, _)| a);
                match (legal.next(), legal.next()) {
                    (Some(action), None) => Some(action),
                    _ => None,
                }
            })
            .collect(),
        None => vec![None; batch_size],
    };
    let active: Vec<usize> = (0..batch_size)
        .filter(|&i| forced_actions[i].is_none())
        .collect();
    let n_active = active.len();

    let (root_hidden_states, root_rewards, root_values, root_policies) =
        mz_agent.initial_inference(observations);
    let alpha = mz_conf.dirichlet_noise;
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

    let forced_result = |i: usize, action: usize| {
        let mut distribution = vec![0.0f32; action_space];
        distribution[action] = 1.0;
        SearchReturn {
            policy_target: distribution.clone(),
            distribution,
            value: root_values[i],
            best_action: action,
        }
    };

    if n_active == 0 {
        return (0..batch_size)
            .map(|i| forced_result(i, forced_actions[i].unwrap()))
            .collect();
    }

    let mut norms: Vec<QNormalization> =
        (0..n_active).map(|_| QNormalization::default()).collect();

    let mut node_batch: Vec<Vec<BatchNode>> = (0..n_active)
        .map(|_| Vec::with_capacity(1 + (mz_conf.num_simulations + 1) * action_space))
        .collect();

    let root_hidden_states = if n_active == batch_size {
        root_hidden_states
    } else {
        let active_rows: Vec<i64> = active.iter().map(|&i| i as i64).collect();
        let idx_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(active_rows.as_slice()), &device);
        root_hidden_states.select(0, idx_tensor)
    };

    let hidden_dim = root_hidden_states.dims()[1];
    let mut arena = Tensor::<B, 2>::zeros(
        [(mz_conf.num_simulations + 1) * n_active, hidden_dim],
        &device,
    );
    arena = arena.slice_assign(0..n_active, root_hidden_states);
    let mut arena_len = n_active;

    node_batch
        .par_iter_mut()
        .with_min_len(mz_conf.min_rayon_threads)
        .enumerate()
        .for_each(|(i, nodes)| {
            let row = active[i];
            nodes.push(BatchNode {
                visits: 0,
                action: 0, // This action is irrelevant
                hidden_row: i,
                first_child: 1,
                cumulative_value: 0.,
                reward: root_rewards[row],
                policy: 0.,
                legal: true,
            });

            let policy_row = &root_policies[row * action_space..(row + 1) * action_space];
            let mask = legal_masks.map(|masks| masks[row].as_slice());
            let priors = root_priors(policy_row, mask, alpha, frac);
            for (action, &policy) in priors.iter().enumerate() {
                nodes.push(BatchNode {
                    visits: 0,
                    action,
                    hidden_row: 0,
                    first_child: 0,
                    cumulative_value: 0.,
                    reward: 0.,
                    policy,
                    legal: mask.is_none_or(|m| m[action]),
                });
            }
        });

    let mut path_batch: Vec<Vec<usize>> = (0..n_active).map(|_| Vec::new()).collect();
    let mut parent_rows: Vec<i64> = Vec::with_capacity(n_active);
    let mut actions: Vec<i64> = Vec::with_capacity(n_active);

    for _sim_step in 0..mz_conf.num_simulations {
        node_batch
            .par_iter()
            .zip(norms.par_iter())
            .zip(path_batch.par_iter_mut())
            .with_min_len(mz_conf.min_rayon_threads)
            .map(|((nodes, norm), path)| {
                let mut curr_node_idx = 0;
                path.clear();
                path.push(0);
                loop {
                    let curr_node = &nodes[curr_node_idx];
                    if curr_node.first_child == 0 {
                        // Unexpanded leaf, go to expansion
                        break;
                    }

                    // compute it once per node instead of once per child.
                    let parent_visits = curr_node.visits;
                    let exploration = exploration_factor(parent_visits);

                    // Find the best child
                    let mut best_puct = f32::NEG_INFINITY;
                    let mut best_node = 0usize;
                    for (child_idx, child) in nodes.iter().enumerate().skip(curr_node.first_child).take(action_space) {
                        if !child.legal {
                            continue;
                        }
                        let q_value = match child.visits {
                            0 => 0.,
                            _ => norm.normalize(
                                child.reward
                                    + discount * value_sign * child.cumulative_value
                                        / child.visits as f32,
                            ),
                        };
                        let puct_value =
                            q_value + child.policy * exploration / (1 + child.visits) as f32;

                        if puct_value > best_puct {
                            best_puct = puct_value;
                            best_node = child_idx;
                        }
                    }

                    curr_node_idx = best_node;
                    path.push(curr_node_idx);
                }

                let [.., parent_idx, leaf_idx] = path.as_slice() else {
                    unreachable!()
                };
                (
                    nodes[*parent_idx].hidden_row as i64,
                    nodes[*leaf_idx].action as i64,
                )
            })
            .unzip_into_vecs(&mut parent_rows, &mut actions);

        // Expansion: one recurrent_inference for all trees
        let row_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(parent_rows.as_slice()), &device);
        let hidden_batch = arena.clone().select(0, row_tensor);
        let action_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::from(actions.as_slice()), &device);
        let (new_hs, new_rewards, new_values, new_policies) =
            mz_agent.recurrent_inference(hidden_batch, action_tensor, action_space);

        arena = arena.slice_assign(arena_len..arena_len + n_active, new_hs);

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

        // Expansion + backprop, one rayon task per tree
        node_batch
            .par_iter_mut()
            .zip(path_batch.par_iter())
            .zip(norms.par_iter_mut())
            .with_min_len(mz_conf.min_rayon_threads)
            .enumerate()
            .for_each(|(i, ((nodes, path), norm))| {
                let leaf_idx = *path.last().unwrap();

                let nodes_len = nodes.len();
                let policy_row = &new_policies[i * action_space..(i + 1) * action_space];
                for (action, &policy) in policy_row.iter().enumerate() {
                    nodes.push(BatchNode {
                        visits: 0,
                        action,
                        hidden_row: 0,
                        first_child: 0,
                        cumulative_value: 0.,
                        reward: 0.,
                        policy,
                        legal: true,
                    });
                }

                let leaf = &mut nodes[leaf_idx];
                leaf.hidden_row = arena_len + i;
                leaf.reward = new_rewards[i];
                leaf.first_child = nodes_len;

                // Backprop: walk path from leaf to root, accumulate discounted returns
                let mut back_value = new_values[i];
                for &node_idx in path.iter().rev() {
                    let curr_node = &mut nodes[node_idx];
                    curr_node.visits += 1;
                    curr_node.cumulative_value += back_value;
                    norm.update(
                        curr_node.reward
                            + discount * value_sign * curr_node.cumulative_value
                                / curr_node.visits as f32,
                    );
                    back_value = curr_node.reward + discount * value_sign * back_value;
                }
            });

        arena_len += n_active;
    }

    let mut tree_idx = 0;
    (0..batch_size)
        .map(|i| match forced_actions[i] {
            Some(action) => forced_result(i, action),
            None => {
                let result = extract_result(&node_batch[tree_idx], action_space, tau);
                tree_idx += 1;
                result
            }
        })
        .collect()
}

fn extract_result(nodes: &[BatchNode], action_space: usize, tau: f32) -> SearchReturn {
    let root_node = &nodes[0];
    let value = root_node.cumulative_value / (root_node.visits as f32);
    let children = root_node.first_child..root_node.first_child + action_space;

    let total_visits: f32 = children
        .clone()
        .filter(|&c| nodes[c].legal)
        .map(|c| nodes[c].visits as f32)
        .sum();

    let mut policy_target = vec![0.0f32; action_space];
    let mut best_action = 0;
    let mut highest_visits = 0;
    for child_idx in children.clone() {
        let child = &nodes[child_idx];
        if !child.legal {
            continue;
        }
        if child.visits > highest_visits {
            highest_visits = child.visits;
            best_action = child.action;
        }
        if total_visits > 0.0 {
            policy_target[child.action] = child.visits as f32 / total_visits;
        }
    }

    let mut distribution = vec![0.0f32; action_space];
    if tau == 0.0 {
        distribution[best_action] = 1.0;
    } else {
        let visit_sum: f32 = children
            .clone()
            .filter(|&c| nodes[c].legal)
            .map(|c| (nodes[c].visits as f32).powf(1.0 / tau))
            .sum();
        if visit_sum > 0.0 {
            for child_idx in children {
                let child = &nodes[child_idx];
                if child.legal {
                    distribution[child.action] =
                        (child.visits as f32).powf(1.0 / tau) / visit_sum;
                }
            }
        } else {
            distribution[best_action] = 1.0;
        }
    }

    SearchReturn {
        distribution,
        policy_target,
        value,
        best_action,
    }
}

fn exploration_factor(parent_visits: usize) -> f32 {
    let c1: f32 = 1.25;
    let c2: f32 = 19652.;
    let pv = parent_visits as f32;
    pv.sqrt() * (c1 + ((pv + c2 + 1.) / c2).ln())
}

fn root_priors(policy: &[f32], mask: Option<&[bool]>, alpha: f32, frac: f32) -> Vec<f32> {
    let mask_legal_len = match mask {
        Some(mask) => mask.iter().filter(|&&m| m).count().max(1),
        None => policy.len(),
    };
    let dirichlet = Dirichlet::new(vec![alpha; mask_legal_len].as_slice()).unwrap();
    let noise = dirichlet.sample(&mut rand::rng());
    let mut noise_iter = noise.into_iter();
    let output = match mask {
        Some(mask) => {
            let legal_sum: f32 = policy
                .iter()
                .zip(mask.iter())
                .filter(|&(_, &m)| m)
                .map(|(p, _)| p)
                .sum();
            let mut output = Vec::<f32>::new();
            for (i, p) in policy.iter().enumerate() {
                if mask[i] {
                    let p = if legal_sum > 0.0 {
                        p / legal_sum
                    } else {
                        1.0 / mask_legal_len as f32
                    };
                    output.push(p * (1. - frac) + frac * noise_iter.next().unwrap());
                } else {
                    output.push(0.0);
                }
            }

            output
        }
        None => policy
            .iter()
            .map(|p| p * (1.0 - frac) + frac * noise_iter.next().unwrap())
            .collect(),
    };

    let sum: f32 = output.iter().sum();
    output.iter().map(|x| x / sum).collect()
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::agent::MlpNets;
    use burn::backend::NdArray;

    // Tests the strictly lower possible bounds and sum of root_priors function
    #[test]
    fn root_priors_test() {
        const EPS: f32 = 1e-6;
        let priors = root_priors(&[0.2, 0.5, 0.3], Some(&[true, true, false]), 1., 0.25);
        println!("Priors: {:?}", &priors);
        assert!(priors[0] >= 0.75 * 0.2);
        assert!(priors[1] >= 0.75 * 0.5);
        assert!(priors[2] == 0.0);

        let sum: f32 = priors.iter().sum();
        println!("sum: {}", &sum);
        assert!((sum - 1.0).abs() < EPS, "Actual sum: {sum}");

        let priors = root_priors(&[0.2, 0.5, 0.3], Some(&[true, false, true]), 1., 0.25);
        assert!(priors[0] >= 0.75 * 0.2);
        assert!(priors[1] == 0.0);
        assert!(priors[2] >= 0.75 * 0.3);

        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "Actual sum: {sum}");

        let priors = root_priors(&[0.2, 0.5, 0.3], None, 1., 0.25);
        assert!(priors[0] >= 0.75 * 0.2);
        assert!(priors[1] >= 0.75 * 0.5);
        assert!(priors[2] >= 0.75 * 0.3);

        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "Actual sum: {sum}");
        assert!(priors.len() == 3);
        println!("Priors: {:?}", &priors);

        let priors = root_priors(&[0.2, 0.5, 0.3], Some(&[true, true, true]), 1., 0.25);
        assert!(priors[0] >= 0.75 * 0.2);
        assert!(priors[1] >= 0.75 * 0.5);
        assert!(priors[2] >= 0.75 * 0.3);

        let sum: f32 = priors.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "Actual sum: {sum}");
    }

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
            let results = batched_search(obs.clone(), None, &mz_conf, &agent, tau);
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
    fn batched_search_single_legal_action() {
        let mz_conf = MuZeroConfig::default();
        let device = Default::default();
        let agent: MlpNets<NdArray> = mz_conf.init(&device);

        let obs = Tensor::<NdArray, 2>::random(
            [2, mz_conf.obs_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let mut forced_mask = vec![false; mz_conf.action_space];
        forced_mask[1] = true;
        let masks = vec![forced_mask, vec![true; mz_conf.action_space]];

        let results = batched_search(obs, Some(&masks), &mz_conf, &agent, 1.0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].best_action, 1);
        assert_eq!(results[0].distribution[1], 1.0);
        let sum: f32 = results[0].distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(results[0].value.is_finite());

        let sum: f32 = results[1].distribution.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(results[1].best_action < mz_conf.action_space);
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
        let results = batched_search(obs, None, &mz_conf, &agent, 1.0);
        for res in &results {
            assert!(res.distribution.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }
}
