use burn::{
    Tensor,
    tensor::{Int, TensorData, Transaction, backend::Backend},
};
use rand_distr::{Distribution, Gumbel, multi::Dirichlet};
use rayon::prelude::*;

use crate::networks::MuZeroNets;
use crate::support::logits_to_scalars;
use crate::{
    mz_config::{MuZeroConfig, SearchAlgorithm},
    utils::QNormalization,
};

const RESCALE_EPS: f32 = 1e-8;

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
    prior: f32,
    logit: f32,
    value: f32,
    legal: bool,
}

struct TreeScratch {
    gumbel: Vec<f32>,
    scores: Vec<f32>,
    num_considered: usize,
}

/// Returns a Vec of SearchReturn. Batch items with a single legal action skip
/// the search and immediately return that action with the network's root value.
pub fn batched_search<B: Backend, N: MuZeroNets<B>>(
    observations: Tensor<B, 2>,
    legal_masks: Option<&[Vec<bool>]>,
    mz_conf: &MuZeroConfig,
    mz_agent: &N,
    tau: f32,
    add_exploration_noise: bool,
) -> Vec<SearchReturn> {
    let batch_size = observations.dims()[0];
    let device = observations.device();
    let discount = mz_conf.discount;
    let action_space = mz_conf.action_space;
    let value_sign = if mz_conf.is_twoplayer { -1.0f32 } else { 1.0f32 };
    let algorithm = mz_conf.search_algorithm;
    let puct_conf = match algorithm {
        SearchAlgorithm::Puct => Some(mz_conf.puct()),
        SearchAlgorithm::Gumbel => None,
    };
    let gumbel_conf = match algorithm {
        SearchAlgorithm::Gumbel => Some(mz_conf.gumbel()),
        SearchAlgorithm::Puct => None,
    };
    let c_visit = gumbel_conf.map_or(0.0, |g| g.c_visit);
    let c_scale = gumbel_conf.map_or(0.0, |g| g.c_scale);
    let max_considered =
        gumbel_conf.map_or(0, |g| g.max_num_considered_actions.min(action_space));
    let alpha = puct_conf.map_or(0.0, |p| p.dirichlet_alpha);
    let frac = match (puct_conf, add_exploration_noise) {
        (Some(puct), true) => puct.root_exploration_fraction,
        _ => 0.0,
    };

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

    let [root_rewards, root_values, root_policies] = Transaction::default()
        .register(root_rewards)
        .register(root_values)
        .register(root_policies)
        .execute()
        .try_into()
        .expect("Correct amount of tensor data");

    let support_size = mz_conf.support_size;
    let root_rewards =
        logits_to_scalars(&root_rewards.into_vec::<f32>().unwrap(), batch_size, support_size);
    let root_values =
        logits_to_scalars(&root_values.into_vec::<f32>().unwrap(), batch_size, support_size);
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

    let considered_table = match algorithm {
        SearchAlgorithm::Gumbel => {
            table_of_considered_visits(max_considered, mz_conf.num_simulations)
        }
        SearchAlgorithm::Puct => Vec::new(),
    };

    let mut norms: Vec<QNormalization> =
        (0..n_active).map(|_| QNormalization::default()).collect();

    let mut tree_batch: Vec<TreeScratch> = (0..n_active)
        .map(|_| TreeScratch {
            gumbel: Vec::new(),
            scores: Vec::new(),
            num_considered: 0,
        })
        .collect();

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

    let gumbel_dist = Gumbel::new(0.0f32, 1.0f32).expect("valid Gumbel(0, 1)");

    node_batch
        .par_iter_mut()
        .zip(tree_batch.par_iter_mut())
        .with_min_len(mz_conf.rayon_min_chunk_len)
        .enumerate()
        .for_each(|(i, (nodes, tree))| {
            let row = active[i];
            nodes.push(BatchNode {
                visits: 0,
                action: 0, // This action is irrelevant
                hidden_row: i,
                first_child: 1,
                cumulative_value: 0.,
                reward: root_rewards[row],
                prior: 0.,
                logit: 0.,
                value: root_values[row],
                legal: true,
            });

            let logits = &root_policies[row * action_space..(row + 1) * action_space];
            let mask = legal_masks.map(|masks| masks[row].as_slice());

            let priors = match algorithm {
                SearchAlgorithm::Puct => root_priors(logits, mask, alpha, frac),
                SearchAlgorithm::Gumbel => masked_softmax(logits, mask),
            };

            if let SearchAlgorithm::Gumbel = algorithm {
                // Gumbel perturbation is the algorithm's exploration; zeroing it
                // makes the root deterministic for evaluation and reanalyze.
                tree.gumbel = if add_exploration_noise {
                    let mut rng = rand::rng();
                    (0..action_space)
                        .map(|_| gumbel_dist.sample(&mut rng))
                        .collect()
                } else {
                    vec![0.0; action_space]
                };
                let num_legal =
                    mask.map_or(action_space, |m| m.iter().filter(|&&l| l).count());
                tree.num_considered = num_legal.min(max_considered);
            }

            for (action, &prior) in priors.iter().enumerate() {
                let legal = mask.is_none_or(|m| m[action]);
                nodes.push(BatchNode {
                    visits: 0,
                    action,
                    hidden_row: 0,
                    first_child: 0,
                    cumulative_value: 0.,
                    reward: 0.,
                    prior,
                    logit: if legal { logits[action] } else { f32::NEG_INFINITY },
                    value: 0.,
                    legal,
                });
            }
        });

    let mut path_batch: Vec<Vec<usize>> = (0..n_active).map(|_| Vec::new()).collect();
    let mut parent_rows: Vec<i64> = Vec::with_capacity(n_active);
    let mut actions: Vec<i64> = Vec::with_capacity(n_active);

    for sim_step in 0..mz_conf.num_simulations {
        node_batch
            .par_iter()
            .zip(norms.par_iter())
            .zip(path_batch.par_iter_mut())
            .zip(tree_batch.par_iter_mut())
            .with_min_len(mz_conf.rayon_min_chunk_len)
            .map(|(((nodes, norm), path), tree)| {
                let mut curr_node_idx = 0;
                path.clear();
                path.push(0);
                loop {
                    if nodes[curr_node_idx].first_child == 0 {
                        // Unexpanded leaf, go to expansion
                        break;
                    }

                    curr_node_idx = match algorithm {
                        SearchAlgorithm::Puct => puct_select(
                            nodes,
                            curr_node_idx,
                            action_space,
                            discount,
                            value_sign,
                            norm,
                        ),
                        SearchAlgorithm::Gumbel => {
                            let (sum_visits, _) = completed_q_scaled(
                                nodes,
                                curr_node_idx,
                                action_space,
                                discount,
                                value_sign,
                                c_visit,
                                c_scale,
                                &mut tree.scores,
                            );
                            if curr_node_idx == 0 {
                                let threshold =
                                    considered_table[tree.num_considered][sim_step];
                                gumbel_root_select(
                                    nodes,
                                    action_space,
                                    &tree.gumbel,
                                    &tree.scores,
                                    threshold,
                                )
                            } else {
                                improved_policy_select(
                                    nodes,
                                    curr_node_idx,
                                    action_space,
                                    &mut tree.scores,
                                    sum_visits,
                                )
                            }
                        }
                    };
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

        let new_rewards =
            logits_to_scalars(&new_rewards.into_vec::<f32>().unwrap(), n_active, support_size);
        let new_values =
            logits_to_scalars(&new_values.into_vec::<f32>().unwrap(), n_active, support_size);
        let new_policies = new_policies.into_vec::<f32>().unwrap();

        // Expansion + backprop, one rayon task per tree
        node_batch
            .par_iter_mut()
            .zip(path_batch.par_iter())
            .zip(norms.par_iter_mut())
            .with_min_len(mz_conf.rayon_min_chunk_len)
            .enumerate()
            .for_each(|(i, ((nodes, path), norm))| {
                let leaf_idx = *path.last().unwrap();

                let nodes_len = nodes.len();
                let logits = &new_policies[i * action_space..(i + 1) * action_space];
                let priors = masked_softmax(logits, None);
                for (action, &prior) in priors.iter().enumerate() {
                    nodes.push(BatchNode {
                        visits: 0,
                        action,
                        hidden_row: 0,
                        first_child: 0,
                        cumulative_value: 0.,
                        reward: 0.,
                        prior,
                        logit: logits[action],
                        value: 0.,
                        legal: true,
                    });
                }

                let leaf = &mut nodes[leaf_idx];
                leaf.hidden_row = arena_len + i;
                leaf.reward = new_rewards[i];
                leaf.value = new_values[i];
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
                let result = match algorithm {
                    SearchAlgorithm::Puct => {
                        extract_result(&node_batch[tree_idx], action_space, tau)
                    }
                    SearchAlgorithm::Gumbel => extract_gumbel_result(
                        &node_batch[tree_idx],
                        action_space,
                        &mut tree_batch[tree_idx],
                        discount,
                        value_sign,
                        c_visit,
                        c_scale,
                    ),
                };
                tree_idx += 1;
                result
            }
        })
        .collect()
}

fn puct_select(
    nodes: &[BatchNode],
    node_idx: usize,
    action_space: usize,
    discount: f32,
    value_sign: f32,
    norm: &QNormalization,
) -> usize {
    let first_child = nodes[node_idx].first_child;
    // compute it once per node instead of once per child.
    let exploration = exploration_factor(nodes[node_idx].visits);

    let mut best_puct = f32::NEG_INFINITY;
    let mut best_node = first_child;
    for (child_idx, child) in nodes.iter().enumerate().skip(first_child).take(action_space) {
        if !child.legal {
            continue;
        }
        let q_value = match child.visits {
            0 => 0.,
            _ => norm.normalize(
                child.reward + discount * value_sign * child.cumulative_value / child.visits as f32,
            ),
        };
        let puct_value = q_value + child.prior * exploration / (1 + child.visits) as f32;

        if puct_value > best_puct {
            best_puct = puct_value;
            best_node = child_idx;
        }
    }
    best_node
}

fn completed_q_scaled(
    nodes: &[BatchNode],
    node_idx: usize,
    action_space: usize,
    discount: f32,
    value_sign: f32,
    c_visit: f32,
    c_scale: f32,
    out: &mut Vec<f32>,
) -> (usize, f32) {
    let first_child = nodes[node_idx].first_child;
    let children = &nodes[first_child..first_child + action_space];

    let child_q =
        |child: &BatchNode| child.reward + discount * value_sign * child.cumulative_value / child.visits as f32;

    let mut sum_visits = 0usize;
    let mut max_visits = 0usize;
    let mut sum_priors = 0.0f32;
    let mut weighted_q = 0.0f32;
    for child in children {
        if child.visits == 0 {
            continue;
        }
        sum_visits += child.visits;
        max_visits = max_visits.max(child.visits);
        sum_priors += child.prior;
        weighted_q += child.prior * child_q(child);
    }

    let raw_value = nodes[node_idx].value;
    let v_mix = if sum_visits == 0 || sum_priors <= 0.0 {
        raw_value
    } else {
        (raw_value + sum_visits as f32 * weighted_q / sum_priors) / (sum_visits as f32 + 1.0)
    };

    out.clear();
    let mut min_q = f32::INFINITY;
    let mut max_q = f32::NEG_INFINITY;
    for child in children {
        let q = if child.visits == 0 {
            v_mix
        } else {
            child_q(child)
        };
        min_q = min_q.min(q);
        max_q = max_q.max(q);
        out.push(q);
    }

    let visit_scale = (c_visit + max_visits as f32) * c_scale;
    let range = max_q - min_q;
    for q in out.iter_mut() {
        *q = visit_scale * (*q - min_q) / (range + RESCALE_EPS);
    }

    (sum_visits, v_mix)
}

fn gumbel_root_select(
    nodes: &[BatchNode],
    action_space: usize,
    gumbel: &[f32],
    scores: &[f32],
    threshold: usize,
) -> usize {
    let first_child = nodes[0].first_child;
    let mut best_node = usize::MAX;
    let mut best_score = f32::NEG_INFINITY;
    let mut fallback = first_child;
    let mut fewest_visits = usize::MAX;

    for (child_idx, child) in nodes.iter().enumerate().skip(first_child).take(action_space) {
        if !child.legal {
            continue;
        }
        if child.visits < fewest_visits {
            fewest_visits = child.visits;
            fallback = child_idx;
        }
        if child.visits != threshold {
            continue;
        }
        let score = gumbel[child.action] + child.logit + scores[child.action];
        if score > best_score {
            best_score = score;
            best_node = child_idx;
        }
    }

    if best_node == usize::MAX {
        fallback
    } else {
        best_node
    }
}

fn improved_policy_select(
    nodes: &[BatchNode],
    node_idx: usize,
    action_space: usize,
    scores: &mut [f32],
    sum_visits: usize,
) -> usize {
    let first_child = nodes[node_idx].first_child;
    let children = &nodes[first_child..first_child + action_space];

    for child in children {
        scores[child.action] = if child.legal {
            child.logit + scores[child.action]
        } else {
            f32::NEG_INFINITY
        };
    }
    softmax_in_place(scores);

    let denom = 1.0 + sum_visits as f32;
    let mut best_node = first_child;
    let mut best = f32::NEG_INFINITY;
    for (child_idx, child) in children.iter().enumerate() {
        if !child.legal {
            continue;
        }
        let score = scores[child.action] - child.visits as f32 / denom;
        if score > best {
            best = score;
            best_node = first_child + child_idx;
        }
    }
    best_node
}

fn extract_result(nodes: &[BatchNode], action_space: usize, tau: f32) -> SearchReturn {
    let root_node = &nodes[0];
    let value = if root_node.visits == 0 {
        0.0
    } else {
        root_node.cumulative_value / root_node.visits as f32
    };
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

fn extract_gumbel_result(
    nodes: &[BatchNode],
    action_space: usize,
    tree: &mut TreeScratch,
    discount: f32,
    value_sign: f32,
    c_visit: f32,
    c_scale: f32,
) -> SearchReturn {
    let (_, v_mix) = completed_q_scaled(
        nodes,
        0,
        action_space,
        discount,
        value_sign,
        c_visit,
        c_scale,
        &mut tree.scores,
    );

    let first_child = nodes[0].first_child;
    let children = &nodes[first_child..first_child + action_space];

    let mut policy_target: Vec<f32> = children
        .iter()
        .map(|child| {
            if child.legal {
                child.logit + tree.scores[child.action]
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    softmax_in_place(&mut policy_target);

    let max_visits = children
        .iter()
        .filter(|child| child.legal)
        .map(|child| child.visits)
        .max()
        .unwrap_or(0);

    let mut best_action = 0;
    let mut best_score = f32::NEG_INFINITY;
    for child in children {
        if !child.legal || child.visits != max_visits {
            continue;
        }
        let score = tree.gumbel[child.action] + child.logit + tree.scores[child.action];
        if score > best_score {
            best_score = score;
            best_action = child.action;
        }
    }

    let mut distribution = vec![0.0f32; action_space];
    distribution[best_action] = 1.0;

    SearchReturn {
        distribution,
        policy_target,
        value: v_mix,
        best_action,
    }
}

fn exploration_factor(parent_visits: usize) -> f32 {
    let c1: f32 = 1.25;
    let c2: f32 = 19652.;
    let pv = parent_visits as f32;
    pv.sqrt() * (c1 + ((pv + c2 + 1.) / c2).ln())
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in values.iter_mut() {
        *v /= sum;
    }
}

fn masked_softmax(logits: &[f32], mask: Option<&[bool]>) -> Vec<f32> {
    let mut out: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(action, &logit)| {
            if mask.is_none_or(|m| m[action]) {
                logit
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    softmax_in_place(&mut out);
    out
}

fn sequence_of_considered_visits(num_considered: usize, num_simulations: usize) -> Vec<usize> {
    if num_considered <= 1 {
        return (0..num_simulations).collect();
    }

    let log2max = usize::BITS as usize - (num_considered - 1).leading_zeros() as usize;
    let mut visits = vec![0usize; num_considered];
    let mut considered = num_considered;
    let mut sequence = Vec::with_capacity(num_simulations);

    while sequence.len() < num_simulations {
        let extra_visits = (num_simulations / (log2max * considered)).max(1);
        for _ in 0..extra_visits {
            sequence.extend_from_slice(&visits[..considered]);
            for visit in visits[..considered].iter_mut() {
                *visit += 1;
            }
        }
        considered = (considered / 2).max(2);
    }

    sequence.truncate(num_simulations);
    sequence
}

fn table_of_considered_visits(max_considered: usize, num_simulations: usize) -> Vec<Vec<usize>> {
    (0..=max_considered)
        .map(|m| sequence_of_considered_visits(m, num_simulations))
        .collect()
}

fn root_priors(logits: &[f32], mask: Option<&[bool]>, alpha: f32, frac: f32) -> Vec<f32> {
    let probs = masked_softmax(logits, mask);
    let legal_len = match mask {
        Some(mask) => mask.iter().filter(|&&m| m).count(),
        None => logits.len(),
    };
    if frac <= 0.0 || legal_len < 2 {
        return probs;
    }

    let dirichlet = Dirichlet::new(vec![alpha; legal_len].as_slice()).unwrap();
    let noise = dirichlet.sample(&mut rand::rng());
    let mut noise_iter = noise.into_iter();

    let mut output: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(action, &p)| match mask.is_none_or(|m| m[action]) {
            true => p * (1. - frac) + frac * noise_iter.next().unwrap(),
            false => 0.0,
        })
        .collect();

    let sum: f32 = output.iter().sum();
    for p in output.iter_mut() {
        *p /= sum;
    }
    output
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use crate::agent::MlpNets;
    use crate::mz_config::{GumbelSubConfig, PuctSubConfig, TemperatureSchedule};
    use burn::backend::NdArray;

    fn puct_conf() -> MuZeroConfig {
        MuZeroConfig {
            search_algorithm: SearchAlgorithm::Puct,
            puct: Some(PuctSubConfig {
                dirichlet_alpha: 0.25,
                root_exploration_fraction: 0.25,
                temperature_schedule: vec![TemperatureSchedule {
                    step: None,
                    tau: 1.0,
                }],
            }),
            ..Default::default()
        }
    }

    fn gumbel_conf(num_simulations: usize) -> MuZeroConfig {
        MuZeroConfig {
            search_algorithm: SearchAlgorithm::Gumbel,
            gumbel: Some(GumbelSubConfig {
                max_num_considered_actions: 16,
                c_visit: 50.0,
                c_scale: 0.1,
            }),
            num_simulations,
            ..Default::default()
        }
    }

    fn child(visits: usize, action: usize, prior: f32, reward: f32, cumulative: f32) -> BatchNode {
        BatchNode {
            visits,
            action,
            hidden_row: 0,
            first_child: 0,
            cumulative_value: cumulative,
            reward,
            prior,
            logit: 0.0,
            value: 0.0,
            legal: true,
        }
    }

    // Tests the strictly lower possible bounds and sum of root_priors function
    #[test]
    fn root_priors_test() {
        const EPS: f32 = 1e-6;
        let logits = [0.2f32.ln(), 0.5f32.ln(), 0.3f32.ln()];

        for mask in [
            Some(vec![true, true, false]),
            Some(vec![true, false, true]),
            Some(vec![true, true, true]),
            None,
        ] {
            let mask_ref = mask.as_deref();
            let base = masked_softmax(&logits, mask_ref);
            let priors = root_priors(&logits, mask_ref, 1., 0.25);

            for action in 0..logits.len() {
                match mask_ref.is_none_or(|m| m[action]) {
                    true => assert!(
                        priors[action] >= 0.75 * base[action] - EPS,
                        "action {action}: {} < 0.75 * {}",
                        priors[action],
                        base[action]
                    ),
                    false => assert_eq!(priors[action], 0.0),
                }
            }

            let sum: f32 = priors.iter().sum();
            assert!((sum - 1.0).abs() < EPS, "Actual sum: {sum}");
            assert_eq!(priors.len(), logits.len());
        }
    }

    #[test]
    fn root_priors_without_noise_is_masked_softmax() {
        let logits = [1.0, -2.0, 0.5];
        let mask = [true, false, true];
        let priors = root_priors(&logits, Some(&mask), 1.0, 0.0);
        let expected = masked_softmax(&logits, Some(&mask));
        for (p, e) in priors.iter().zip(expected.iter()) {
            assert!((p - e).abs() < 1e-6);
        }
    }

    #[test]
    fn considered_visit_sequences() {
        assert_eq!(sequence_of_considered_visits(4, 8), vec![0, 0, 0, 0, 1, 1, 2, 2]);
        assert_eq!(sequence_of_considered_visits(2, 4), vec![0, 0, 1, 1]);
        assert_eq!(sequence_of_considered_visits(1, 5), vec![0, 1, 2, 3, 4]);
        assert_eq!(sequence_of_considered_visits(0, 3), vec![0, 1, 2]);

        let table = table_of_considered_visits(16, 50);
        assert_eq!(table.len(), 17);
        for row in &table {
            assert_eq!(row.len(), 50);
        }
    }

    #[test]
    fn considered_visits_start_with_top_m_draw() {
        let sequence = sequence_of_considered_visits(8, 32);
        assert!(sequence[..8].iter().all(|&v| v == 0));
        assert_ne!(sequence[8], 0);
    }

    #[test]
    fn v_mix_is_raw_value_without_visits() {
        let mut nodes = vec![child(0, 0, 0.0, 0.0, 0.0)];
        nodes[0].value = 1.5;
        nodes[0].first_child = 1;
        nodes.push(child(0, 0, 0.5, 0.0, 0.0));
        nodes.push(child(0, 1, 0.5, 0.0, 0.0));

        let mut out = Vec::new();
        let (sum_visits, v_mix) =
            completed_q_scaled(&nodes, 0, 2, 1.0, 1.0, 50.0, 0.1, &mut out);

        assert_eq!(sum_visits, 0);
        assert!((v_mix - 1.5).abs() < 1e-6);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|q| q.abs() < 1e-6));
    }

    #[test]
    fn v_mix_mixes_raw_value_and_visited_q() {
        let visits = 4usize;
        let q = 2.0f32;
        let raw_value = 0.5f32;

        let mut nodes = vec![child(0, 0, 0.0, 0.0, 0.0)];
        nodes[0].value = raw_value;
        nodes[0].first_child = 1;
        nodes.push(child(visits, 0, 1.0, 0.0, q * visits as f32));
        nodes.push(child(0, 1, 0.0, 0.0, 0.0));

        let mut out = Vec::new();
        let (sum_visits, v_mix) =
            completed_q_scaled(&nodes, 0, 2, 1.0, 1.0, 50.0, 0.1, &mut out);

        assert_eq!(sum_visits, visits);
        let expected = (raw_value + visits as f32 * q) / (visits as f32 + 1.0);
        assert!((v_mix - expected).abs() < 1e-5, "{v_mix} != {expected}");

        let visit_scale = (50.0 + visits as f32) * 0.1;
        assert!((out[0] - visit_scale).abs() < 1e-3, "{}", out[0]);
        assert!(out[1].abs() < 1e-3, "{}", out[1]);
    }

    #[test]
    fn batched_search_valid_distributions() {
        let device = Default::default();

        for mz_conf in [puct_conf(), gumbel_conf(16)] {
            let agent: MlpNets<NdArray> = mz_conf.init(&device);
            let batch_size = 3;
            let obs = Tensor::<NdArray, 2>::random(
                [batch_size, mz_conf.obs_dim],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );

            for tau in [0.0, 1.0] {
                let results = batched_search(obs.clone(), None, &mz_conf, &agent, tau, false);
                assert_eq!(results.len(), batch_size);
                for res in &results {
                    assert_eq!(res.distribution.len(), mz_conf.action_space);
                    let sum: f32 = res.distribution.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-4, "distribution sums to {sum}");
                    let target_sum: f32 = res.policy_target.iter().sum();
                    assert!((target_sum - 1.0).abs() < 1e-4, "target sums to {target_sum}");
                    assert!(res.best_action < mz_conf.action_space);
                    assert!(res.value.is_finite());
                }
            }
        }
    }

    #[test]
    fn batched_search_single_legal_action() {
        let device = Default::default();

        for mz_conf in [puct_conf(), gumbel_conf(16)] {
            let agent: MlpNets<NdArray> = mz_conf.init(&device);
            let obs = Tensor::<NdArray, 2>::random(
                [2, mz_conf.obs_dim],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            );

            let mut forced_mask = vec![false; mz_conf.action_space];
            forced_mask[1] = true;
            let masks = vec![forced_mask, vec![true; mz_conf.action_space]];

            let results = batched_search(obs, Some(&masks), &mz_conf, &agent, 1.0, false);
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
    }

    #[test]
    fn batched_search_visits_match_simulations() {
        let mz_conf = puct_conf();
        let device = Default::default();
        let agent: MlpNets<NdArray> = mz_conf.init(&device);

        let obs = Tensor::<NdArray, 2>::random(
            [2, mz_conf.obs_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        // With tau=1 the distribution is visits/total; root child visits sum
        // to num_simulations, so no probability mass can be lost.
        let results = batched_search(obs, None, &mz_conf, &agent, 1.0, false);
        for res in &results {
            assert!(res.distribution.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }
    }

    #[test]
    fn gumbel_respects_legal_mask() {
        let device = Default::default();
        let mut mz_conf = gumbel_conf(8);
        mz_conf.action_space = 6;
        let agent: MlpNets<NdArray> = mz_conf.init(&device);

        let obs = Tensor::<NdArray, 2>::random(
            [4, mz_conf.obs_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let mut mask = vec![false; mz_conf.action_space];
        mask[2] = true;
        mask[5] = true;
        let masks = vec![mask; 4];

        let results = batched_search(obs, Some(&masks), &mz_conf, &agent, 1.0, false);
        for res in &results {
            assert!(res.best_action == 2 || res.best_action == 5);
            for (action, &p) in res.policy_target.iter().enumerate() {
                if action != 2 && action != 5 {
                    assert_eq!(p, 0.0, "illegal action {action} has mass {p}");
                }
            }
            let sum: f32 = res.policy_target.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "target sums to {sum}");
        }
    }

    #[test]
    fn gumbel_single_simulation_is_valid() {
        let device = Default::default();
        let mz_conf = gumbel_conf(1);
        let agent: MlpNets<NdArray> = mz_conf.init(&device);

        let obs = Tensor::<NdArray, 2>::random(
            [2, mz_conf.obs_dim],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let results = batched_search(obs, None, &mz_conf, &agent, 1.0, false);
        for res in &results {
            assert!(res.best_action < mz_conf.action_space);
            assert!(res.value.is_finite());
            assert!(res.policy_target.iter().all(|p| p.is_finite()));
            let sum: f32 = res.policy_target.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "target sums to {sum}");
        }
    }
}
