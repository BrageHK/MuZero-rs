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
}

/// Returns a Vec of SearchReturn. This function will crash if there is only 1 legal action.
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

    let mut norms: Vec<QNormalization> =
        (0..batch_size).map(|_| QNormalization::default()).collect();

    let mut node_batch: Vec<Vec<BatchNode>> = (0..batch_size)
        .map(|_| Vec::with_capacity((mz_conf.num_simulations + 1) * action_space))
        .collect();

    let (root_hidden_states, root_rewards, root_values, root_policies) =
        mz_agent.initial_inference(observations);
    let alpha = mz_conf.dirichlet_noise;
    let frac = mz_conf.root_exploration_fraction;

    let hidden_dim = root_hidden_states.dims()[1];
    let mut arena = Tensor::<B, 2>::zeros(
        [(mz_conf.num_simulations + 1) * batch_size, hidden_dim],
        &device,
    );
    arena = arena.slice_assign([0..batch_size], root_hidden_states);
    let mut arena_len = batch_size;

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

    node_batch
        .par_iter_mut()
        .with_min_len(mz_conf.min_rayon_threads)
        .enumerate()
        .for_each(|(i, nodes)| {
            nodes.push(BatchNode {
                visits: 1,
                action: 0, // This action is irrelevant
                hidden_row: i,
                first_child: 1,
                cumulative_value: root_values[i],
                reward: root_rewards[i],
                policy: 0.,
            });

            let policy_row = &root_policies[i * action_space..(i + 1) * action_space];
            let mask = legal_masks.map(|masks| masks[i].as_slice());
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
                });
            }
        });

    let mut path_batch: Vec<Vec<usize>> = (0..batch_size).map(|_| Vec::new()).collect();
    let mut parent_rows: Vec<i64> = Vec::with_capacity(batch_size);
    let mut actions: Vec<i64> = Vec::with_capacity(batch_size);

    for _sim_step in 0..mz_conf.num_simulations {
        node_batch
            .par_iter()
            .zip(norms.par_iter_mut())
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
                    for child_idx in curr_node.first_child..curr_node.first_child + action_space {
                        let child = &nodes[child_idx];
                        let q_value = match child.visits {
                            0 => 0.,
                            _ => norm.get_q(child.cumulative_value / child.visits as f32),
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

        arena = arena.slice_assign([arena_len..arena_len + batch_size], new_hs);

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
            .with_min_len(mz_conf.min_rayon_threads)
            .enumerate()
            .for_each(|(i, (nodes, path))| {
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
                    back_value = curr_node.reward + discount * back_value;
                }
            });

        arena_len += batch_size;
    }

    (0..batch_size)
        .map(|i| extract_result(&node_batch[i], action_space, tau))
        .collect()
}

fn extract_result(nodes: &[BatchNode], action_space: usize, tau: f32) -> SearchReturn {
    let root_node = &nodes[0];
    let value = root_node.cumulative_value / (root_node.visits as f32);
    let mut visit_distribution = vec![0.0f32; action_space];
    let children = root_node.first_child..root_node.first_child + action_space;

    if tau == 0.0 {
        let best_child_idx = children
            .max_by_key(|child_idx| nodes[*child_idx].visits)
            .expect("There are no child nodes.");
        let best_action = nodes[best_child_idx].action;
        visit_distribution[best_action] = 1.0;
        SearchReturn {
            distribution: visit_distribution,
            value,
            best_action,
        }
    } else {
        let visit_sum: f32 = children
            .clone()
            .map(|child_idx| (nodes[child_idx].visits as f32).powf(1.0 / tau))
            .sum();
        let mut highest_visits = 0;
        let mut best_action = 0;
        for child_idx in children {
            let action = nodes[child_idx].action;
            let child_visits = nodes[child_idx].visits;
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

fn exploration_factor(parent_visits: usize) -> f32 {
    let c1: f32 = 1.25;
    let c2: f32 = 19652.;
    let pv = parent_visits as f32;
    pv.sqrt() * (c1 + ((pv + c2 + 1.) / c2).ln())
}

fn root_priors(policy: &[f32], mask: Option<&[bool]>, alpha: f32, frac: f32) -> Vec<f32> {
    let mask_legal_len = match mask {
        Some(mask) => mask.iter().filter(|&&m| m).count(),
        None => policy.len(),
    };
    let dirichlet = Dirichlet::new(vec![alpha; mask_legal_len].as_slice()).unwrap();
    let noise = dirichlet.sample(&mut rand::rng());
    let output = match mask {
        Some(mask) => {
            let mut output = Vec::<f32>::new();
            for (i, p) in policy.iter().enumerate() {
                if mask[i] {
                    output.push(p * (1. - frac) + frac * noise.iter().next().unwrap());
                } else {
                    output.push(0.0);
                }
            }

            output
        }
        None => policy
            .iter()
            .map(|p| p * (1.0 - frac) + frac * noise.iter().next().unwrap())
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
